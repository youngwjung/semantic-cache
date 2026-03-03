terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "6.34.0"
    }
    opensearch = {
      source  = "opensearch-project/opensearch"
      version = "2.3.2"
    }
  }
}

# 환경 변수
locals {
  project    = "semantic-cache"
  account_id = data.aws_caller_identity.current.account_id
  aws_region = data.aws_region.current.id
}

# AWS 정보
data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

data "aws_availability_zones" "azs" {}

data "aws_bedrock_foundation_models" "embedding" {
  by_output_modality = "EMBEDDING"
  by_provider        = "Amazon"
}

data "aws_bedrock_foundation_model" "titan" {
  model_id = data.aws_bedrock_foundation_models.embedding.model_summaries[0].model_id
}

################################################################################
# VPC
################################################################################

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "6.6.0"

  name = local.project
  cidr = "10.0.0.0/16"

  azs            = data.aws_availability_zones.azs.names
  public_subnets = [for idx, _ in data.aws_availability_zones.azs.names : cidrsubnet("10.0.0.0/16", 8, idx)]
}

################################################################################
# Bedrock Knowledge Base
################################################################################
module "kb_bucket" {
  source  = "terraform-aws-modules/s3-bucket/aws"
  version = "5.10.0"

  force_destroy = true
}

resource "aws_s3vectors_vector_bucket" "this" {
  vector_bucket_name = "${local.account_id}-${local.project}"
}

resource "aws_s3vectors_index" "this" {
  index_name         = local.project
  vector_bucket_name = aws_s3vectors_vector_bucket.this.vector_bucket_name

  data_type       = "float32"
  dimension       = 1024
  distance_metric = "cosine"
}

resource "aws_iam_role" "kb" {
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = "AmazonBedrockKnowledgeBaseTrustPolicy"
        Principal = {
          Service = "bedrock.amazonaws.com"
        }
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = local.account_id
          }
          ArnLike = {
            "aws:SourceArn" = "arn:aws:bedrock:${local.aws_region}:${local.account_id}:knowledge-base/*"
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "kb_fm" {
  role = aws_iam_role.kb.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action   = ["bedrock:InvokeModel"]
        Effect   = "Allow"
        Resource = data.aws_bedrock_foundation_model.titan.model_arn
      }
    ]
  })
}

resource "aws_iam_role_policy" "kb_s3" {
  role = aws_iam_role.kb.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action   = ["s3:ListBucket"]
        Effect   = "Allow"
        Resource = module.kb_bucket.s3_bucket_arn
      },
      {
        Action   = ["s3:GetObject"]
        Effect   = "Allow"
        Resource = "${module.kb_bucket.s3_bucket_arn}/*"
      }
    ]
  })
}

resource "aws_iam_role_policy" "kb_s3_vector" {
  role = aws_iam_role.kb.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3vectors:PutVectors",
          "s3vectors:GetVectors",
          "s3vectors:DeleteVectors",
          "s3vectors:QueryVectors",
          "s3vectors:GetIndex"
        ]
        Effect   = "Allow"
        Resource = aws_s3vectors_index.this.index_arn
      }
    ]
  })
}

resource "aws_bedrockagent_knowledge_base" "this" {
  name     = local.project
  role_arn = aws_iam_role.kb.arn

  knowledge_base_configuration {
    vector_knowledge_base_configuration {
      embedding_model_arn = data.aws_bedrock_foundation_model.titan.model_arn
      embedding_model_configuration {
        bedrock_embedding_model_configuration {
          dimensions          = 1024
          embedding_data_type = "FLOAT32"
        }
      }
    }
    type = "VECTOR"
  }

  storage_configuration {
    type = "S3_VECTORS"
    s3_vectors_configuration {
      index_arn = aws_s3vectors_index.this.index_arn
    }
  }
}

resource "aws_bedrockagent_data_source" "this" {
  knowledge_base_id = aws_bedrockagent_knowledge_base.this.id
  name              = "s3"

  data_source_configuration {
    type = "S3"
    s3_configuration {
      bucket_arn = module.kb_bucket.s3_bucket_arn
    }
  }
}

################################################################################
# Valkey (ElastiCache)
################################################################################

module "valkey_sg" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "5.3.1"

  name        = local.project
  description = "${local.project} valkey security group"
  vpc_id      = module.vpc.vpc_id

  ingress_cidr_blocks = ["10.0.0.0/16"]
  ingress_rules       = ["redis-tcp"]
}

module "valkey" {
  source  = "terraform-aws-modules/elasticache/aws"
  version = "1.11.0"

  replication_group_id = local.project
  engine               = "valkey"
  engine_version       = "8.2"
  node_type            = "cache.t4g.micro"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.public_subnets
  security_group_ids = [
    module.valkey_sg.security_group_id
  ]
  transit_encryption_mode = "preferred"

  create_parameter_group = true
  parameter_group_family = "valkey8"

  log_delivery_configuration = {}
}

################################################################################
# ECS
################################################################################
module "alb" {
  source  = "terraform-aws-modules/alb/aws"
  version = "10.5.0"

  vpc_id  = module.vpc.vpc_id
  subnets = module.vpc.public_subnets

  enable_deletion_protection = false

  security_group_ingress_rules = {
    all_http = {
      from_port   = 80
      to_port     = 80
      ip_protocol = "tcp"
      cidr_ipv4   = "0.0.0.0/0"
    }
  }
  security_group_egress_rules = {
    all = {
      ip_protocol = "-1"
      cidr_ipv4   = module.vpc.vpc_cidr_block
    }
  }

  listeners = {
    http = {
      port     = 80
      protocol = "HTTP"

      forward = {
        target_group_key = "ecs"
      }
    }
  }

  target_groups = {
    ecs = {
      backend_protocol     = "HTTP"
      backend_port         = "8501"
      target_type          = "ip"
      deregistration_delay = 5
      create_attachment    = false
    }
  }
}

module "ecs" {
  source  = "terraform-aws-modules/ecs/aws"
  version = "7.3.1"

  cluster_name               = local.project
  cluster_capacity_providers = ["FARGATE"]

  services = {
    streamlit = {
      container_definitions = {
        streamlit = {
          image                  = "youngwjung/semantic-cache"
          essential              = true
          readonlyRootFilesystem = false
          portMappings = [
            {
              name          = "http"
              containerPort = 8501
              protocol      = "tcp"
            }
          ]
          environment = [
            {
              name  = "VALKEY_HOST"
              value = module.valkey.replication_group_primary_endpoint_address
            }
          ]
        }
      }

      load_balancer = {
        service = {
          target_group_arn = module.alb.target_groups["ecs"].arn
          container_name   = "streamlit"
          container_port   = "8501"
        }
      }

      tasks_iam_role_policies = {
        AmazonBedrockFullAccess = "arn:aws:iam::aws:policy/AmazonBedrockFullAccess"
        AmazonS3FullAccess      = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
      }

      enable_autoscaling = false

      subnet_ids       = module.vpc.public_subnets
      assign_public_ip = true

      security_group_ingress_rules = {
        all = {
          ip_protocol = "-1"
          cidr_ipv4   = "0.0.0.0/0"
        }
      }
      security_group_egress_rules = {
        all = {
          ip_protocol = "-1"
          cidr_ipv4   = "0.0.0.0/0"
        }
      }
    }
  }
}

################################################################################
# CloudFront
################################################################################

data "aws_cloudfront_cache_policy" "caching_disabled" {
  name = "Managed-CachingDisabled"
}

data "aws_cloudfront_origin_request_policy" "all_viewer" {
  name = "Managed-AllViewer"
}

module "cloudfront" {
  source  = "terraform-aws-modules/cloudfront/aws"
  version = "6.4.0"

  enabled = true

  origin = {
    alb = {
      domain_name = module.alb.dns_name
      custom_origin_config = {
        http_port              = 80
        https_port             = 443
        origin_protocol_policy = "http-only"
        origin_ssl_protocols   = ["TLSv1.2"]
      }
    }
  }

  default_cache_behavior = {
    target_origin_id       = "alb"
    viewer_protocol_policy = "redirect-to-https"

    allowed_methods = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods  = ["GET", "HEAD"]

    cache_policy_id          = data.aws_cloudfront_cache_policy.caching_disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer.id

    compress = true
  }

  origin_access_control = {}

  viewer_certificate = {
    cloudfront_default_certificate = true
    minimum_protocol_version       = "TLSv1"
  }
}

################################################################################
# Output
################################################################################

output "cloudfornt_dns" {
  value = module.cloudfront.cloudfront_distribution_domain_name
}
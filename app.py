import os
import time
import boto3
import streamlit as st

from valkey import Valkey
from langchain_aws import BedrockEmbeddings
from langgraph_checkpoint_aws import ValkeyStore
from hashlib import md5


# Streamlit 설정
st.set_page_config(page_title="Semantic Caching Demo", layout="wide")
st.title("Semantic Caching Demo")

# 로컬 세션
if "messages" not in st.session_state:
    st.session_state.messages = []

if "logs" not in st.session_state:
    st.session_state.logs = []

# 세션에 저장된 메세지를 화면에 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"][0]["text"])

# Amazon Bedrock 클라이언트
bedrock_client = boto3.client("bedrock")

# Amazon Bedrock Runtime 클라이언트
bedrock_runtime_client = boto3.client("bedrock-runtime")

# Amazon Bedrock Agent 클라이언트
bedrock_agent_client = boto3.client("bedrock-agent")

# Amazon Bedrock Agent Runtime 클라이언트
bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime")

# 모델 선택 UI
models = bedrock_client.list_inference_profiles()["inferenceProfileSummaries"]
model_options = {}
for model in models:
    # 채팅을 지원하는 모델만 추가
    if "Embed" not in model["inferenceProfileName"]:
        model_options[model["inferenceProfileName"]] = model["inferenceProfileId"]

selected_model_name = st.sidebar.selectbox(
    "모델 선택", options=list(model_options.keys())
)

# RAG
rag = st.sidebar.toggle("RAG", value=False)

if rag:
    # Knowledge Base 목록
    kbs = bedrock_agent_client.list_knowledge_bases()["knowledgeBaseSummaries"]
    kb_options = {}
    for kb in kbs:
        kb_options[kb["name"]] = kb["knowledgeBaseId"]

    selected_kb_name = st.sidebar.selectbox(
        "Knowledge Base 선택", options=list(kb_options.keys())
    )

    # 문서 관련성 점수 임계값 설정
    rag_score_threshold = st.sidebar.slider(
        "문서 관련성 점수 임계값", 0.0, 1.0, 0.8, 0.05
    )

    # 선택된 Knowledge Base의 Date Source 확인
    datasource_id = bedrock_agent_client.list_data_sources(
        knowledgeBaseId=kb_options[selected_kb_name]
    )["dataSourceSummaries"][0]["dataSourceId"]

    # Knowledge Base에 연동된 S3로 파일 업로드
    with st.sidebar.form(key="file_upload", clear_on_submit=True):
        uploaded_file = st.file_uploader("파일 선택")
        submitted = st.form_submit_button("업로드", use_container_width=True)

    if submitted and uploaded_file:
        with st.sidebar.spinner("파일 전송중.."):
            s3_client = boto3.client("s3")

            datasource = bedrock_agent_client.get_data_source(
                knowledgeBaseId=kb_options[selected_kb_name],
                dataSourceId=datasource_id,
            )

            bucket_name = datasource["dataSource"]["dataSourceConfiguration"][
                "s3Configuration"
            ]["bucketArn"].split(":")[-1]

            s3_client.upload_fileobj(
                uploaded_file,
                bucket_name,
                uploaded_file.name,
            )

            st.sidebar.success("업로드 완료")

    # Knowledge Base 동기화
    if st.sidebar.button("동기화", use_container_width=True):
        response = bedrock_agent_client.start_ingestion_job(
            knowledgeBaseId=kb_options[selected_kb_name], dataSourceId=datasource_id
        )

        with st.sidebar.status(response["ingestionJob"]["status"]) as status:
            ingestion_job_id = response["ingestionJob"]["ingestionJobId"]
            while True:
                response = bedrock_agent_client.get_ingestion_job(
                    knowledgeBaseId=kb_options[selected_kb_name],
                    dataSourceId=datasource_id,
                    ingestionJobId=ingestion_job_id,
                )

                st.write(response["ingestionJob"]["status"])
                if response["ingestionJob"]["status"] in (
                    "COMPLETE",
                    "FAILED",
                    "STOPPED",
                ):
                    status.update(label=response["ingestionJob"]["status"])
                    break

                time.sleep(5)

# Cache
cache = False
if rag:
    cache = st.sidebar.toggle("Cache", value=False)

if cache:
    # 유사도 임계값 설정
    cache_similarity_threshold = st.sidebar.slider("유사도 임계값", 0.0, 1.0, 0.9, 0.05)

    valkey_client = Valkey(
        host=os.getenv("VALKEY_HOST"),
        port=6379,
        decode_responses=False,
    )

    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

    # Valkey 벡터 저장소 설정
    store = ValkeyStore(
        client=valkey_client,
        index={
            "collection_name": "semantic_cache",
            "embed": embeddings,
            "fields": ["query"],
            "index_type": "FLAT",
            "distance_metric": "COSINE",
            "dims": 1024,
        },
    )
    store.setup()

    # 캐시 키 생성
    def cache_key_for_query(query: str):
        return md5(query.encode("utf-8")).hexdigest()

    # 캐시 검색
    def search_cache(user_message: str, k: int = 3) -> tuple[dict | None, float | None]:
        hits = store.search(
            namespace_prefix=("semantic-cache",), query=user_message, limit=k
        )
        if not hits:
            return None, None

        # 유사도가 높은순으로 정렬
        hits = sorted(hits, key=lambda h: h.score)
        for hit in hits:
            add_log(f"{hit.value['query']}, {hit.score}")

        # 유사도가 가장 높은 결과
        top_hit = hits[0]

        # 유사도 점수를 0 - 1 사이로 정규화
        score = 1 - top_hit.score

        return top_hit.value, score

    # 캐시 저장
    def store_cache(user_message: str, result_message: str) -> None:
        key = cache_key_for_query(user_message)
        store.put(
            namespace=("semantic-cache",),
            key=key,
            value={"query": user_message, "answer": result_message},
        )

    # 캐시 목록 조회
    def get_cache_items():
        items = []
        for key in valkey_client.scan_iter(match="langgraph:semantic-cache/*"):
            data = valkey_client.hgetall(key)
            query = data.get(b"query")
            if query:
                items.append(query.decode("utf-8"))
        return items

    # 캐시 목록 표시
    with st.sidebar.expander("캐시 목록"):
        items = get_cache_items()
        if items:
            to_delete = []
            for item in items:
                if st.checkbox(item):
                    to_delete.append(item)
            # 캐시 삭제
            if to_delete and st.button("🗑️ 삭제", use_container_width=True):
                keys = [
                    f"langgraph:semantic-cache/{cache_key_for_query(q)}"
                    for q in to_delete
                ]
                valkey_client.delete(*keys)
                st.rerun()
        else:
            st.caption("저장된 캐시 없음")


# 채팅 기록 삭제 버튼
if st.sidebar.button("🗑️ 채팅 기록 삭제", use_container_width=True):
    st.session_state.messages = []
    st.session_state.logs = []
    st.rerun()

# 로그 컨테이너
st.sidebar.markdown("로그")
log_container = st.sidebar.empty()


# 로그 렌더링
def render_logs():
    if st.session_state.logs:
        log_container.code("\n".join(st.session_state.logs), language=None)


def add_log(msg: str):
    st.session_state.logs.append(msg)
    render_logs()


# 초기 로그 렌더링
render_logs()

# 사용자가 메시지를 입력
if prompt := st.chat_input("무엇이든 물어보세요."):
    # 이전 로그 초기화
    st.session_state.logs = []

    # 세션에 사용자가 입력한 메시지 추가
    st.session_state.messages.append({"role": "user", "content": [{"text": prompt}]})

    # 채팅창에 사용자 입력 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # 채팅창에 에이전트 응답 메시지 표시
    with st.chat_message("assistant"):
        # LLM 응답을 출력할 빈 컨테이너 생성
        placeholder = st.empty()
        cache_hit = False
        try:
            # 스트리밍 데이터 임시 저장
            buffer = ""

            # RAG가 활성화 된 경우
            if rag:
                # 캐시가 활성화 된 경우
                if cache:
                    # 캐시 검색
                    add_log(f"[Cache] 캐시 검색 중... 'query: {prompt}'")
                    match, score = search_cache(prompt)

                    # 유사한 쿼리가 검색된 경우
                    if match and cache_similarity_threshold < score:
                        buffer = match["answer"]
                        cache_hit = True
                        add_log(f"[Cache] Hit | {match['query']} | 유사도: {score:.2f}")
                    # 유사한 쿼리가 없으면
                    else:
                        if score is not None:
                            add_log(
                                f"[Cache] Miss | {match['query']} | 유사도: {score:.2f} | 임계값: {cache_similarity_threshold})"
                            )
                        else:
                            add_log("[Cache] Miss | 저장된 캐시 없음")

                        add_log("[RAG] Knowledge Base 검색 중...")
                        # 문서 검색
                        retrieve_response = bedrock_agent_runtime_client.retrieve(
                            knowledgeBaseId=kb_options[selected_kb_name],
                            retrievalQuery={"text": prompt},
                            retrievalConfiguration={
                                "vectorSearchConfiguration": {"numberOfResults": 5}
                            },
                        )
                        results = retrieve_response.get("retrievalResults", [])

                        # 문서 관련성 로그
                        for r in results:
                            add_log(
                                f"  [{r.get('score', 0):.2f}] {r['content']['text'][:40]}..."
                            )

                        # 관련성 임계값 이상 문서만 필터링
                        relevant_docs = [
                            r
                            for r in results
                            if r.get("score", 0) >= rag_score_threshold
                        ]

                        if relevant_docs:
                            add_log(
                                f"[RAG] 관련 문서 {len(relevant_docs)}개 → LLM 호출"
                            )
                            context = "\n\n---\n\n".join(
                                [doc["content"]["text"] for doc in relevant_docs]
                            )
                            messages = st.session_state.messages + [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "text": f"참고 문서:\n\n{context}\n\n질문: {prompt}"
                                        }
                                    ],
                                }
                            ]

                            # 추론 요청
                            response = bedrock_runtime_client.converse_stream(
                                modelId=model_options[selected_model_name],
                                messages=messages,
                            )

                            # 이벤트가 수신되면
                            for event in response.get("stream", []):
                                # 이벤트에 응답 세그먼트가 존재하면
                                if "contentBlockDelta" in event:
                                    # 텍스트 축출
                                    delta = event["contentBlockDelta"].get("delta", {})
                                    chunk = delta.get("text", "")
                                    if chunk:
                                        # 버퍼에 응답 세그먼트 추가
                                        buffer += chunk
                                        # 현재까지 수신된 응답을 화면에 표시
                                        placeholder.markdown(buffer)

                            add_log("[Cache] 캐시 저장 완료")
                            store_cache(prompt, buffer.strip())
                        else:
                            add_log("[RAG] 관련 문서 없음 → 캐시 저장 안함")
                            buffer = "관련 문서를 찾을 수 없습니다."
                            placeholder.markdown(buffer)

                # 캐시가 활성화 안된 경우
                else:
                    add_log("[RAG] Knowledge Base 검색 중...")
                    # 문서 검색
                    retrieve_response = bedrock_agent_runtime_client.retrieve(
                        knowledgeBaseId=kb_options[selected_kb_name],
                        retrievalQuery={"text": prompt},
                        retrievalConfiguration={
                            "vectorSearchConfiguration": {"numberOfResults": 5}
                        },
                    )
                    results = retrieve_response.get("retrievalResults", [])

                    # 문서 관련성 로그
                    for r in results:
                        add_log(
                            f"  [{r.get('score', 0):.2f}] {r['content']['text'][:40]}..."
                        )

                    # 관련성 임계값 이상 문서만 필터링
                    relevant_docs = [
                        r for r in results if r.get("score", 0) >= rag_score_threshold
                    ]

                    if relevant_docs:
                        add_log(f"[RAG] 관련 문서 {len(relevant_docs)}개 → LLM 호출")
                        context = "\n\n---\n\n".join(
                            [doc["content"]["text"] for doc in relevant_docs]
                        )
                        messages = st.session_state.messages + [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": f"참고 문서:\n\n{context}\n\n질문: {prompt}"
                                    }
                                ],
                            }
                        ]
                    else:
                        add_log("[RAG] 관련 문서 없음 → 일반 LLM 호출")
                        messages = st.session_state.messages

                    response = bedrock_runtime_client.converse_stream(
                        modelId=model_options[selected_model_name],
                        messages=messages,
                    )

                    # 이벤트가 수신되면
                    for event in response.get("stream", []):
                        # 이벤트에 응답 세그먼트가 존재하면
                        if "contentBlockDelta" in event:
                            # 텍스트 축출
                            delta = event["contentBlockDelta"].get("delta", {})
                            chunk = delta.get("text", "")
                            if chunk:
                                # 버퍼에 응답 세그먼트 추가
                                buffer += chunk
                                # 현재까지 수신된 응답을 화면에 표시
                                placeholder.markdown(buffer)

            else:
                add_log("[LLM] 추론 요청 중...")
                # 추론 요청
                response = bedrock_runtime_client.converse_stream(
                    modelId=model_options[selected_model_name],
                    messages=st.session_state.messages,
                )

                # 이벤트가 수신되면
                for event in response.get("stream", []):
                    # 이벤트에 응답 세그먼트가 존재하면
                    if "contentBlockDelta" in event:
                        # 텍스트 축출
                        delta = event["contentBlockDelta"].get("delta", {})
                        chunk = delta.get("text", "")
                        if chunk:
                            # 버퍼에 응답 세그먼트 추가
                            buffer += chunk
                            # 현재까지 수신된 응답을 화면에 표시
                            placeholder.markdown(buffer)

            # 수신 완료된 응답
            response_text = buffer.strip() or ""
        # 에러가 발생하면 에러 메세지를 응답으로 화면에 표시
        except Exception as e:
            response_text = f"**Error:** {e}"

        # 수신된 완료된 응답을 화면에 표시
        if cache_hit:
            placeholder.markdown(
                response_text
                + """
                <div style='text-align: right; color: gray; font-size: 16px; margin-top: 4px;'>
                    Cache Hit
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            placeholder.markdown(response_text)

    # 세션에 수신된 완료된 응답 추가
    st.session_state.messages.append(
        {"role": "assistant", "content": [{"text": response_text}]}
    )

    # 캐시 미스로 새 항목이 저장된 경우 목록 즉시 갱신
    if cache and not cache_hit:
        st.rerun()

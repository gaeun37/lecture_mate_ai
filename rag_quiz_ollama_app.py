
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import chromadb
import pandas as pd
import pymupdf
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from json_repair import repair_json


# =========================
# 기본 설정
# =========================

st.set_page_config(
    page_title="General PDF RAG Quiz Generator",
    page_icon="🧠",
    layout="wide"
)


# =========================
# 1. 모델 / DB
# =========================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_lecture_db")


# =========================
# 2. PDF 처리
# =========================

def make_safe_id(text: str) -> str:
    """
    파일명/자료명을 ChromaDB id에 안전하게 넣기 위한 보조 함수.
    """
    safe = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", str(text)).strip("_")
    return safe[:80] if safe else "material"


def extract_text_from_pdf(
    file_bytes: bytes,
    file_name: str = "uploaded.pdf",
    material_id: str = "material"
) -> List[Dict[str, Any]]:
    pages = []

    with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            pages.append({
                "material_id": material_id,
                "file_name": file_name,
                "page": page_index,
                "text": text,
                "char_count": len(text)
            })

    return pages


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        next_start = end - chunk_overlap

        if next_start <= start:
            break

        start = next_start

    return chunks


def make_chunks_from_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> List[Dict[str, Any]]:
    chunks = []

    for page in pages:
        page_num = page["page"]
        text = page["text"]
        file_name = str(page.get("file_name", "uploaded.pdf"))
        material_id = str(page.get("material_id", make_safe_id(file_name)))

        if not text:
            continue

        for idx, chunk_text in enumerate(split_text(text, chunk_size, chunk_overlap), start=1):
            chunks.append({
                "chunk_id": f"{material_id}_p{page_num}_c{idx}",
                "material_id": material_id,
                "file_name": file_name,
                "page": page_num,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "quality_score": estimate_chunk_quality(chunk_text)
            })

    return chunks


# =========================
# 3. 범용 검색 보조
# =========================

GENERIC_STOPWORDS = {
    "문제", "만들어줘", "내줘", "출제", "묻는", "대한", "관련", "가장", "적절한",
    "설명", "다음", "객관식", "보기", "정답", "이후", "과정", "역할", "의미",
    "배경", "성격", "차이", "비교", "무엇인지", "무엇", "어떤", "어떻게",
    "왜", "핵심", "개념", "question", "quiz", "make", "about", "explain",
    "compare", "what", "why", "how"
}


def normalize_for_match(text: Any) -> str:
    text = str(text).lower()
    text = text.replace("·", "․")
    text = re.sub(r"\s+", "", text)
    return text


def normalize_blank_marker(question: Any) -> str:
    """
    빈칸 주관식에서 모델이 ___, ____, _____처럼 서로 다른 길이의 밑줄을 출력해도
    화면과 검증에서는 표준 빈칸 표시인 _____로 통일한다.
    """
    question = str(question).strip()
    question = re.sub(r"_{3,}", "_____", question)
    return question


def has_blank_marker(question: Any) -> bool:
    """
    빈칸 표시가 3개 이상의 밑줄로 들어오면 빈칸 문제로 인정한다.
    """
    return re.search(r"_{3,}", str(question)) is not None


def tokenize_query(text: str) -> List[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9_+./·․%-]+", str(text))
    result = []

    for token in tokens:
        token = token.strip()

        if len(token) < 2:
            continue

        if token.lower() in GENERIC_STOPWORDS or token in GENERIC_STOPWORDS:
            continue

        if token not in result:
            result.append(token)

    return result[:30]


def is_table_of_contents_like(text: str) -> bool:
    if not text:
        return True

    compact = normalize_for_match(text)

    if len(compact) < 80:
        return True

    meta_words = [
        "목차", "차례", "contents", "index", "참고문헌", "references",
        "bibliography", "copyright", "allrightsreserved", "표지", "cover"
    ]

    if any(word in compact for word in meta_words) and len(compact) < 900:
        return True

    digit_ratio = sum(ch.isdigit() for ch in text) / max(1, len(text))

    if digit_ratio > 0.25 and len(compact) < 700:
        return True

    return False


def estimate_chunk_quality(text: str) -> int:
    if not text:
        return 0

    compact = normalize_for_match(text)
    score = 50

    if len(compact) > 250:
        score += 20

    if len(compact) > 500:
        score += 10

    if is_table_of_contents_like(text):
        score -= 50

    sentence_markers = [
        "이다", "한다", "된다", "있다", "하였다", "되었다", "의미한다",
        "is", "are", "was", "were", "means", "represents", "because"
    ]

    if sum(str(text).lower().count(m) for m in sentence_markers) >= 2:
        score += 15

    return max(0, min(100, score))


def keyword_score(query: str, text: str, extra_keywords: Optional[List[str]] = None) -> int:
    q_tokens = tokenize_query(query)
    if extra_keywords:
        q_tokens.extend([kw for kw in extra_keywords if kw and kw not in q_tokens])

    compact_text = normalize_for_match(text)
    score = 0

    for token in q_tokens:
        nt = normalize_for_match(token)

        if not nt:
            continue

        if nt in compact_text:
            score += 3 if len(nt) >= 5 else 1

    return score


# =========================
# 4. ChromaDB 저장 / 검색
# =========================

def create_or_reset_collection(client, collection_name: str):
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    return client.create_collection(name=collection_name)


def store_chunks_in_chroma(collection, chunks: List[Dict[str, Any]], embedding_model) -> int:
    ids = []
    docs = []
    metas = []

    for chunk in chunks:
        text = chunk["text"].strip()

        if not text:
            continue

        ids.append(str(chunk["chunk_id"]))
        docs.append(text)
        metas.append({
            "page": int(chunk["page"]),
            "file_name": str(chunk.get("file_name", "")),
            "material_id": str(chunk.get("material_id", "")),
            "char_count": int(chunk.get("char_count", len(text))),
            "quality_score": int(chunk.get("quality_score", estimate_chunk_quality(text)))
        })

    if not docs:
        return 0

    embeddings = embedding_model.encode(docs).tolist()

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeddings
    )

    return len(ids)


def search_chunks(
    collection,
    query: str,
    embedding_model,
    top_k: int = 2,
    extra_keywords: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    total = collection.count()

    if total <= 0:
        return []

    query_for_embedding = query

    if extra_keywords:
        query_for_embedding += " " + " ".join(extra_keywords)

    q_emb = embedding_model.encode([query_for_embedding]).tolist()[0]
    n_results = min(max(top_k * 8, 12), total)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results
    )

    candidates = []

    for chunk_id, doc, meta, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        quality = int(meta.get("quality_score", estimate_chunk_quality(doc)))
        k_score = keyword_score(query, doc, extra_keywords)
        toc_like = is_table_of_contents_like(doc)

        try:
            distance = float(distance)
        except Exception:
            distance = 999.0

        # Chroma distance는 낮을수록 좋다.
        semantic_score = max(0.0, 1.5 - distance)
        rerank = semantic_score * 10 + k_score * 6 + quality / 10

        if toc_like:
            rerank -= 25

        candidates.append({
            "chunk_id": chunk_id,
            "file_name": meta.get("file_name", ""),
            "material_id": meta.get("material_id", ""),
            "page": meta.get("page"),
            "distance": distance,
            "text": doc,
            "quality_score": quality,
            "keyword_score": k_score,
            "is_noise": toc_like,
            "rerank_score": round(rerank, 4)
        })

    filtered = [
        c for c in candidates
        if not c["is_noise"] and c["quality_score"] >= 25
    ]

    if not filtered:
        filtered = candidates

    filtered = sorted(
        filtered,
        key=lambda x: (-x["rerank_score"], -x["keyword_score"], x["distance"])
    )

    return filtered[:top_k]


# =========================
# 5. 질문 의도 / 프롬프트
# =========================

def get_level_number(student_level: str) -> int:
    match = re.search(r"Level\s*(\d)", str(student_level))

    if match:
        return int(match.group(1))

    return 2


def detect_question_intent(user_query: str) -> str:
    q = str(user_query)

    if any(word in q for word in ["무엇인지", "무엇", "이름", "명칭", "누구", "언제", "날짜"]):
        return "term_lookup"

    if any(word in q for word in ["비교", "차이", "구분"]):
        return "comparison"

    if any(word in q for word in ["역할", "임무", "기능", "목적", "작용", "담당", "맡은", "하는 일"]):
        return "role_lookup"

    if any(word in q for word in ["의미", "영향", "평가", "해석"]):
        return "interpretation"

    if any(word in q for word in ["배경", "성격", "이유", "원인", "결과"]):
        return "cause_effect"

    return "general"


def get_question_direction_rule(student_level: str, question_direction: str) -> str:
    level_num = get_level_number(student_level)

    if question_direction.startswith("긍정형"):
        return """
- 반드시 positive 문제를 만든다.
- question_polarity는 "positive"로 작성한다.
"""

    if question_direction.startswith("부정형"):
        return """
- 반드시 negative 문제를 만든다.
- question_polarity는 "negative"로 작성한다.
- negative 문제에서는 정답이 교안 내용과 일치하지 않는 보기여야 한다.
- 나머지 3개 보기는 교안 내용과 일치해야 한다.
"""

    if level_num <= 1:
        return """
- 초급 문제이므로 positive 문제를 만든다.
- question_polarity는 "positive"로 작성한다.
"""

    return """
- 자동 모드에서는 positive 문제를 우선 생성한다.
- question_polarity는 "positive"로 작성한다.
"""


def build_quiz_prompt(
    user_query: str,
    retrieved_chunks: List[Dict[str, Any]],
    student_level: str,
    weak_concept: str,
    question_type: str,
    question_direction: str,
    question_intent: str
) -> str:
    allowed_pages = sorted({
        int(chunk["page"])
        for chunk in retrieved_chunks
        if chunk.get("page") is not None
    })

    context_blocks = []

    per_chunk_limit = 750 if question_intent in ["comparison", "role_lookup", "general"] else 600

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[근거 {idx}]\n"
            f"페이지: {chunk['page']}\n"
            f"내용:\n{chunk['text'][:per_chunk_limit]}\n"
        )

    context = "\n\n".join(context_blocks)
    level_num = get_level_number(student_level)
    direction_rule = get_question_direction_rule(student_level, question_direction)

    intent_rule = {
        "term_lookup": """
- 사용자가 묻는 이름/용어/날짜/인물 자체를 정답으로 하는 문제를 만든다.
- 질문을 특징/의미/배경 문제로 바꾸지 마라.
- 정답은 25자 이내의 짧은 명사구 또는 날짜여야 한다.
- 해설과 요약은 짧게 작성한다.
""",
        "role_lookup": """
- 특정 개념/요소/기관/방법의 역할, 임무, 기능을 묻는 문제를 만든다.
- 근거에 없는 역할, 목적, 효과를 추가하지 마라.
- 근거에 '등'처럼 넓은 표현만 있으면 그 범위를 넘어 새로운 역할을 만들지 마라.
""",
        "comparison": """
- 두 개 이상의 개념, 단계, 조건, 역할, 사건, 수식, 실험 결과를 구분하는 문제를 만든다.
- 한 보기에는 하나의 비교 관계만 담는다.
""",
        "cause_effect": """
- 배경, 원인, 결과를 연결하는 문제를 만든다.
- 근거에 없는 인과관계를 추측하지 마라.
""",
        "interpretation": """
- 교안 근거에 명시된 의미나 해석만 사용한다.
- 근거 밖 평가를 추가하지 마라.
""",
        "general": """
- 학생 요청의 중심 주제와 직접 관련된 문제를 만든다.
"""
    }.get(question_intent, "")

    is_ox = "OX" in question_type or "O/X" in question_type or "ox" in question_type.lower()
    is_fill_blank = (not is_ox) and ("빈칸" in question_type or "주관식" in question_type)

    if is_ox:
        type_rule = """
[문제 유형별 규칙]
- OX 문제를 만든다.
- question은 반드시 "다음 설명이 옳으면 O, 틀리면 X를 고르시오:" 형식으로 작성한다.
- choices는 반드시 ["O", "X"]로 작성한다.
- answer는 반드시 "O" 또는 "X" 중 하나로만 작성한다.
- 자동 또는 긍정형 문제에서는 교안 내용과 일치하는 설명을 만들고 answer를 "O"로 작성한다.
- 부정형 문제에서는 교안 내용과 명확히 어긋나는 설명을 만들고 answer를 "X"로 작성한다.
- 설명문은 교안 근거에 나온 핵심 사실을 바탕으로 만든다.
- 교안 근거에 없는 내용을 섞어서 판단하기 어려운 문장을 만들지 마라.
- choice_explanations는 반드시 빈 리스트 []로 둔다.
"""

        output_format = f"""
[출력 형식]
{{
  "question_type": "ox",
  "question_polarity": "positive",
  "question": "다음 설명이 옳으면 O, 틀리면 X를 고르시오: 교안 근거 기반 설명문",
  "choices": ["O", "X"],
  "answer": "O",
  "part_summary": "출제 파트 요약",
  "evidence_text": "정답 근거",
  "explanation": "O 또는 X가 정답인 이유",
  "choice_explanations": [],
  "source_pages": [{allowed_pages[0] if allowed_pages else 1}],
  "concept": "핵심 개념",
  "difficulty": {level_num},
  "hint": "정답을 직접 말하지 않는 짧은 힌트 한 문장",
  "grading_criteria": []
}}
"""

    elif is_fill_blank:
        type_rule = """
[문제 유형별 규칙]
- 빈칸 주관식 문제를 만든다.
- question에는 반드시 빈칸 표시 "_____"를 정확히 한 번 포함한다.
- answer에는 빈칸에 들어갈 정답 단어 또는 짧은 구절만 작성한다.
- answer는 30자 이내로 작성한다.
- choices는 반드시 빈 리스트 []로 둔다.
- choice_explanations도 반드시 빈 리스트 []로 둔다.
- grading_criteria에는 채점 기준 3개를 작성한다.
- 빈칸에는 교안 근거에 실제로 등장하거나, 교안 근거에서 직접 확인 가능한 핵심 용어가 들어가야 한다.
- 정답을 문장 전체로 쓰지 말고, 빈칸에 들어갈 단어 또는 짧은 구절만 작성한다.
"""

        output_format = f"""
[출력 형식]
{{
  "question_type": "fill_blank",
  "question_polarity": "positive",
  "question": "빈칸 _____ 이 포함된 질문 한 문장",
  "choices": [],
  "answer": "빈칸에 들어갈 정답 단어 또는 짧은 구절",
  "part_summary": "출제 파트 요약",
  "evidence_text": "정답 근거",
  "explanation": "정답이 빈칸에 들어가야 하는 이유",
  "choice_explanations": [],
  "source_pages": [{allowed_pages[0] if allowed_pages else 1}],
  "concept": "핵심 개념",
  "difficulty": {level_num},
  "hint": "정답을 직접 말하지 않는 짧은 힌트 한 문장",
  "grading_criteria": [
    "빈칸에 핵심 용어를 정확히 작성했는가",
    "교안 근거의 의미와 맞는 답을 작성했는가",
    "유사 표현을 쓰더라도 핵심 개념이 유지되는가"
  ]
}}
"""

    else:
        type_rule = """
[문제 유형별 규칙]
- 4지선다 객관식 문제를 만든다.
- choices에는 보기 4개를 반드시 작성한다.
- 보기 4개는 서로 다른 내용이어야 한다.
- answer는 choices 중 하나와 글자까지 완전히 같아야 한다.
- choice_explanations에는 보기 4개 각각의 해설을 작성한다.
"""

        output_format = f"""
[출력 형식]
{{
  "question_type": "multiple_choice",
  "question_polarity": "positive",
  "question": "질문 한 문장",
  "choices": ["보기1", "보기2", "보기3", "보기4"],
  "answer": "choices 중 정답 문장 하나",
  "part_summary": "출제 파트 요약",
  "evidence_text": "정답 근거",
  "explanation": "정답인 이유",
  "choice_explanations": [
    {{"choice": "보기1", "is_answer": true, "is_factually_correct": true, "explanation": "해설"}},
    {{"choice": "보기2", "is_answer": false, "is_factually_correct": false, "explanation": "해설"}},
    {{"choice": "보기3", "is_answer": false, "is_factually_correct": false, "explanation": "해설"}},
    {{"choice": "보기4", "is_answer": false, "is_factually_correct": false, "explanation": "해설"}}
  ],
  "source_pages": [{allowed_pages[0] if allowed_pages else 1}],
  "concept": "핵심 개념",
  "difficulty": {level_num},
  "hint": "짧은 힌트 한 문장",
  "grading_criteria": []
}}
"""

    return f"""
너는 대학 강의자료 기반 학습 문제를 만드는 AI 튜터이다.

반드시 [교안 근거]에 있는 내용만 사용해서 학습 문제 1개를 만들어라.
교안 근거에 없는 개념, 정의, 수식, 수치, 용어, 사례, 역할, 효과를 상상해서 추가하지 마라.
전문용어, 고유명사, 수식, 숫자, 날짜는 교안 근거에 나온 표현을 최대한 그대로 사용하라.

[학생 요청]
{user_query}

[학생 수준]
{student_level}

[문제 유형]
{question_type}

[문항 방향 선택]
{question_direction}

[질문 의도]
{question_intent}

[학생 약점 개념]
{weak_concept if weak_concept else "없음"}

[사용 가능한 근거 페이지]
{allowed_pages}

[교안 근거]
{context}

[문항 방향 규칙]
{direction_rule}

[질문 의도별 규칙]
{intent_rule}

[공통 생성 규칙]
- 문제 문장은 반드시 질문형으로 작성한다.
- 교안 근거에 없는 내용을 만들지 마라.
- part_summary는 1~2문장으로 작성한다.
- evidence_text는 정답 판단에 필요한 직접 근거만 1~2문장으로 작성한다.
- explanation은 1~2문장으로 작성한다.
- source_pages에는 [사용 가능한 근거 페이지] 안에 있는 숫자만 넣는다.
- difficulty는 반드시 {level_num}으로 작성한다.
- 출력은 반드시 JSON 객체 하나만 반환한다.
- JSON 앞뒤에 설명 문장이나 마크다운 코드블록을 붙이지 마라.
- 빈칸 주관식이면 question에 반드시 "_____"를 정확히 한 번 포함한다.
- 빈칸 주관식이면 answer는 빈칸에 들어갈 단어 또는 짧은 구절만 작성한다.
- 빈칸 주관식이면 choices는 []로 둔다.
- OX 문제이면 choices는 ["O", "X"]로 둔다.
- OX 문제이면 answer는 반드시 "O" 또는 "X"로 작성한다.

{type_rule}

{output_format}
""".strip()

# =========================
# 6. Ollama 호출
# =========================

def normalize_ollama_generate_url(ollama_url: str) -> str:
    """
    사용자가 /api/chat 또는 루트 주소를 넣어도 /api/generate로 보정한다.
    이 앱은 prompt 기반 /api/generate를 사용한다.
    """
    url = str(ollama_url).strip().rstrip("/")

    if url.endswith("/api/chat"):
        return url[:-len("/api/chat")] + "/api/generate"

    if url.endswith("/api/generate"):
        return url

    if url.endswith("/api"):
        return url + "/generate"

    return url + "/api/generate"


def check_ollama_server(ollama_url: str) -> bool:
    try:
        generate_url = normalize_ollama_generate_url(ollama_url)
        tags_url = generate_url.replace("/api/generate", "/api/tags")
        response = requests.get(tags_url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def call_ollama_streaming(
    ollama_url: str,
    payload: Dict[str, Any],
    connect_timeout: int = 10,
    stream_read_timeout: int = 120
) -> str:
    """
    Ollama /api/generate를 streaming으로 호출한다.

    qwen3 계열은 stream 응답에서 실제 토큰이 response가 아니라 thinking 필드로
    먼저 오는 경우가 있다. 이 앱은 최종 JSON만 필요하므로 response를 우선 수집하고,
    response가 완전히 비어 있는데 thinking 안에 JSON이 들어온 경우에만 thinking을 fallback으로 사용한다.
    """
    generate_url = normalize_ollama_generate_url(ollama_url)

    payload = dict(payload)
    payload["stream"] = True

    response = requests.post(
        generate_url,
        json=payload,
        stream=True,
        timeout=(connect_timeout, stream_read_timeout)
    )
    response.raise_for_status()

    response_pieces = []
    thinking_pieces = []
    raw_lines_for_debug = []

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue

        raw_lines_for_debug.append(str(line)[:300])

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            response_pieces.append(str(line))
            continue

        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data.get('error')}")

        if "response" in data and data.get("response"):
            response_pieces.append(data.get("response") or "")

        # qwen3 thinking model 대응
        if "thinking" in data and data.get("thinking"):
            thinking_pieces.append(data.get("thinking") or "")

        # 일부 chat 형태 응답 방어
        if isinstance(data.get("message"), dict):
            content = data["message"].get("content") or ""
            thinking = data["message"].get("thinking") or ""

            if content:
                response_pieces.append(content)
            if thinking:
                thinking_pieces.append(thinking)

        if data.get("done") is True:
            break

    response_output = "".join(response_pieces).strip()
    thinking_output = "".join(thinking_pieces).strip()

    if response_output:
        return response_output

    # Qwen3가 JSON을 thinking 필드에만 넣는 경우가 있어서,
    # thinking 안에 JSON 객체가 보일 때만 fallback으로 사용한다.
    if thinking_output and "{" in thinking_output and "}" in thinking_output:
        return thinking_output

    debug_preview = " | ".join(raw_lines_for_debug[:5])
    raise ValueError(
        "Ollama가 최종 response를 반환하지 않았습니다. "
        "qwen3 thinking 모드가 켜져 있거나 모델이 빈 응답을 반환했을 수 있습니다. "
        f"URL={generate_url}, stream lines preview={debug_preview}"
    )


def parse_ollama_json(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text.strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    start = raw_text.find("{")
    end = raw_text.rfind("}") + 1

    if start != -1 and end > start:
        raw_text = raw_text[start:end]

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        repaired = repair_json(raw_text)
        return json.loads(repaired)


# =========================
# 6-1. 파일 전체 핵심 요약
# =========================

def build_summary_context_from_chunks(
    chunks: List[Dict[str, Any]],
    max_chars: int = 12000
) -> str:
    """
    전체 PDF를 한 번에 LLM에 넣으면 너무 길어지므로,
    목차/표지성 청크를 제외하고 페이지 순서대로 핵심 청크를 모아
    요약용 context를 만든다.
    """
    good_chunks = []

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()

        if not text:
            continue

        if is_table_of_contents_like(text):
            continue

        if int(chunk.get("quality_score", 0)) < 25:
            continue

        good_chunks.append(chunk)

    good_chunks = sorted(
        good_chunks,
        key=lambda x: (
            int(x.get("page", 0)),
            str(x.get("chunk_id", ""))
        )
    )

    context_parts = []
    current_len = 0

    for chunk in good_chunks:
        page = chunk.get("page", "")
        file_name = chunk.get("file_name", "")
        text = str(chunk.get("text", "")).strip()[:900]
        block = f"[파일 {file_name} / 페이지 {page}]\n{text}\n"

        if current_len + len(block) > max_chars:
            break

        context_parts.append(block)
        current_len += len(block)

    return "\n".join(context_parts)




def clean_summary_output(text: str) -> str:
    """
    Qwen 계열 모델이 영어 thinking/reasoning을 출력하는 경우,
    최종 한국어 요약 부분만 남기기 위한 후처리 함수.
    """
    if not text:
        return ""

    text = text.replace("```markdown", "").replace("```", "").strip()

    possible_starts = [
        "## 1. 한 줄 요약",
        "## 한 줄 요약",
        "# 한 줄 요약",
        "1. 한 줄 요약",
        "한 줄 요약",
        "## 2. 전체 핵심 요약",
        "전체 핵심 요약",
    ]

    for marker in possible_starts:
        idx = text.find(marker)
        if idx != -1:
            return text[idx:].strip()

    # 모델의 영어 사고 과정으로 보이는 줄 제거
    lines = text.splitlines()
    cleaned_lines = []
    skip_prefixes = (
        "Okay", "First,", "Wait", "The key", "The user",
        "I need", "Let's", "Now,", "The document",
        "The text", "This seems", "It seems", "Hmm",
    )

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if cleaned_lines:
                cleaned_lines.append(line)
            continue

        if stripped.startswith(skip_prefixes):
            continue

        # 영어 알파벳 비율이 매우 높고 한글이 거의 없으면 추론 문장으로 보고 제거
        korean_count = len(re.findall(r"[가-힣]", stripped))
        alpha_count = len(re.findall(r"[A-Za-z]", stripped))

        if korean_count == 0 and alpha_count >= 20:
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned if cleaned else text.strip()


def summary_json_to_markdown(summary_json: Dict[str, Any]) -> str:
    one_line = str(summary_json.get("one_line_summary", "")).strip()
    full_summary = summary_json.get("full_summary", [])
    key_concepts = summary_json.get("key_concepts", [])
    exam_points = summary_json.get("exam_points", [])

    if not isinstance(full_summary, list):
        full_summary = [str(full_summary)]

    if not isinstance(key_concepts, list):
        key_concepts = [str(key_concepts)]

    if not isinstance(exam_points, list):
        exam_points = [str(exam_points)]

    md = ""

    md += "## 1. 한 줄 요약\n"
    md += f"- {one_line}\n\n"

    md += "## 2. 전체 핵심 요약\n"
    for item in full_summary:
        item = str(item).strip()
        if item:
            md += f"- {item}\n"

    md += "\n## 3. 핵심 개념\n"
    for item in key_concepts:
        item = str(item).strip()
        if item:
            md += f"- {item}\n"

    md += "\n## 4. 시험 대비 포인트\n"
    for item in exam_points:
        item = str(item).strip()
        if item:
            md += f"- {item}\n"

    return md.strip()

def generate_document_summary_with_ollama(
    ollama_url: str,
    model_name: str,
    chunks: List[Dict[str, Any]],
    num_predict: int,
    stream_read_timeout: int
) -> str:
    """
    업로드된 PDF 전체 내용을 핵심 요약한다.
    Ollama에게 JSON으로 요약을 받은 뒤, 코드에서 Markdown으로 변환한다.
    """
    context = build_summary_context_from_chunks(chunks)

    if not context.strip():
        return "요약할 수 있는 본문 텍스트가 충분하지 않습니다."

    prompt = f"""
너는 대학 강의자료를 요약하는 AI 학습 도우미이다.

반드시 한국어로만 답하라.
영어를 절대 사용하지 마라.
생각 과정, 분석 과정, 추론 과정을 출력하지 마라.
"Okay", "First", "Wait", "The user", "start with" 같은 문장을 절대 출력하지 마라.
아래 [PDF 본문]에 있는 내용만 사용하라.
본문에 없는 사건, 인물, 연도, 용어를 추가하지 마라.

중요:
- 출력 형식 설명을 그대로 복사하지 마라.
- "파일 전체 내용을 한 문장으로 요약" 같은 예시 문구를 그대로 쓰지 마라.
- 반드시 [PDF 본문]의 실제 내용을 바탕으로 값을 채워라.
- JSON 객체 하나만 출력하라.

[PDF 본문]
{context}

[출력 JSON 형식]
{{
  "one_line_summary": "PDF 본문을 바탕으로 실제 한 줄 요약을 작성",
  "full_summary": [
    "PDF 본문을 바탕으로 실제 핵심 요약 문장 1",
    "PDF 본문을 바탕으로 실제 핵심 요약 문장 2",
    "PDF 본문을 바탕으로 실제 핵심 요약 문장 3",
    "PDF 본문을 바탕으로 실제 핵심 요약 문장 4",
    "PDF 본문을 바탕으로 실제 핵심 요약 문장 5"
  ],
  "key_concepts": [
    "PDF 본문에 등장하는 실제 핵심 개념 1",
    "PDF 본문에 등장하는 실제 핵심 개념 2",
    "PDF 본문에 등장하는 실제 핵심 개념 3",
    "PDF 본문에 등장하는 실제 핵심 개념 4",
    "PDF 본문에 등장하는 실제 핵심 개념 5"
  ],
  "exam_points": [
    "PDF 본문을 바탕으로 시험에 나올 수 있는 포인트 1",
    "PDF 본문을 바탕으로 시험에 나올 수 있는 포인트 2",
    "PDF 본문을 바탕으로 시험에 나올 수 있는 포인트 3"
  ]
}}
""".strip()

    payload = {
        "model": model_name,
        "prompt": "/no_think\n반드시 최종 답변만 한국어 JSON으로 출력하세요. 영어 사고 과정은 출력하지 마세요.\n" + prompt,
        "stream": True,
        "think": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": int(max(num_predict, 1200))
        }
    }

    try:
        summary_raw = call_ollama_streaming(
            ollama_url=ollama_url,
            payload=payload,
            stream_read_timeout=stream_read_timeout
        )

        summary_raw = clean_summary_output(summary_raw)

        try:
            summary_json = parse_ollama_json(summary_raw)
            return summary_json_to_markdown(summary_json)

        except Exception:
            # JSON 파싱이 실패해도 화면에는 정리된 텍스트를 보여준다.
            return clean_summary_output(summary_raw).strip()

    except Exception as e:
        return f"전체 요약 생성 실패: {e}"


def group_chunks_by_material(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    여러 PDF 청크를 파일별로 묶는다.
    파일별 핵심 요약과 이후 학습 로드맵 생성의 기본 단위가 된다.
    """
    grouped: Dict[str, Dict[str, Any]] = {}

    for chunk in chunks:
        file_name = str(chunk.get("file_name", "uploaded.pdf"))
        material_id = str(chunk.get("material_id", make_safe_id(file_name)))

        if material_id not in grouped:
            grouped[material_id] = {
                "material_id": material_id,
                "file_name": file_name,
                "chunks": []
            }

        grouped[material_id]["chunks"].append(chunk)

    result = list(grouped.values())
    result.sort(key=lambda x: x["file_name"])
    return result


def generate_document_summaries_by_file_with_ollama(
    ollama_url: str,
    model_name: str,
    chunks: List[Dict[str, Any]],
    num_predict: int,
    stream_read_timeout: int
) -> List[Dict[str, str]]:
    """
    여러 PDF를 파일별로 나누어 각각 핵심 요약을 생성한다.
    """
    summaries = []

    for item in group_chunks_by_material(chunks):
        file_name = item["file_name"]
        material_id = item["material_id"]
        file_chunks = item["chunks"]

        summary = generate_document_summary_with_ollama(
            ollama_url=ollama_url,
            model_name=model_name,
            chunks=file_chunks,
            num_predict=num_predict,
            stream_read_timeout=stream_read_timeout
        )

        summaries.append({
            "material_id": material_id,
            "file_name": file_name,
            "summary": summary
        })

    return summaries


def summaries_by_file_to_markdown(summaries: List[Dict[str, str]]) -> str:
    """
    파일별 요약 결과를 다운로드 가능한 하나의 Markdown 텍스트로 합친다.
    """
    parts = []

    for item in summaries:
        file_name = item.get("file_name", "uploaded.pdf")
        summary = item.get("summary", "")
        parts.append(f"# {file_name}\n\n{summary}")

    return "\n\n---\n\n".join(parts).strip()


# =========================
# 6-2. 학습 로드맵 생성
# =========================

def normalize_file_summaries_for_roadmap(file_summaries: Any) -> Dict[str, str]:
    """
    파일별 요약 결과를 로드맵에서 쓰기 쉽게 dict 형태로 정리한다.
    예상 형태:
    - {"파일명.pdf": "요약 내용"}
    - [{"file_name": "...", "summary": "..."}]
    """
    if not file_summaries:
        return {}

    if isinstance(file_summaries, dict):
        return {str(k): str(v) for k, v in file_summaries.items()}

    if isinstance(file_summaries, list):
        result = {}

        for item in file_summaries:
            if isinstance(item, dict):
                file_name = str(item.get("file_name", "업로드 자료"))
                summary = str(item.get("summary", ""))
                result[file_name] = summary

        return result

    return {}


def compact_page_ranges(pages: List[int]) -> str:
    pages = sorted(set(int(p) for p in pages if p is not None))

    if not pages:
        return "-"

    ranges = []
    start = pages[0]
    prev = pages[0]

    for page in pages[1:]:
        if page == prev + 1:
            prev = page
        else:
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
            start = page
            prev = page

    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")

    return ", ".join(ranges)


def get_chunk_file_name(chunk: Dict[str, Any]) -> str:
    return str(
        chunk.get("file_name")
        or chunk.get("material_name")
        or chunk.get("source_file")
        or "업로드 자료"
    )


def get_summary_hint(file_name: str, file_summaries: Dict[str, str]) -> str:
    summary = file_summaries.get(file_name, "")

    if not summary:
        return "해당 자료의 핵심 개념을 정리하고, 중요한 용어를 확인한다."

    lines = []

    for line in summary.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        line = line.lstrip("- ").strip()

        if line:
            lines.append(line)

        if len(lines) >= 2:
            break

    if not lines:
        return "해당 자료의 핵심 개념을 정리하고, 중요한 용어를 확인한다."

    return " / ".join(lines)[:160]


def build_learning_roadmap_df(
    chunks: List[Dict[str, Any]],
    duration_days: int,
    file_summaries: Optional[Any] = None
) -> pd.DataFrame:
    """
    선택한 기간에 맞춰 PDF 학습 로드맵을 생성한다.
    - 전체 기간의 약 70%는 진도 학습
    - 나머지는 누적 복습 / 문제풀이 / 최종 점검
    """
    duration_days = int(duration_days)
    summaries = normalize_file_summaries_for_roadmap(file_summaries)

    valid_chunks = []

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()

        if not text:
            continue

        if is_table_of_contents_like(text):
            continue

        if int(chunk.get("quality_score", 0)) < 25:
            continue

        valid_chunks.append(chunk)

    if not valid_chunks:
        valid_chunks = chunks

    valid_chunks = sorted(
        valid_chunks,
        key=lambda x: (
            get_chunk_file_name(x),
            int(x.get("page", 0)),
            str(x.get("chunk_id", ""))
        )
    )

    if not valid_chunks:
        return pd.DataFrame([
            {
                "Day": 1,
                "구분": "오류",
                "학습 범위": "학습할 청크가 없습니다.",
                "학습 목표": "PDF 처리를 먼저 진행하세요.",
                "할 일": "PDF 업로드 후 청킹을 다시 실행하세요.",
                "점검": "-"
            }
        ])

    learning_days = max(1, int(duration_days * 0.7))
    learning_days = min(learning_days, duration_days)
    learning_days = min(learning_days, len(valid_chunks))

    review_interval = 3 if duration_days <= 7 else 5 if duration_days <= 28 else 7
    per_day = max(1, (len(valid_chunks) + learning_days - 1) // learning_days)

    rows = []

    for day in range(1, duration_days + 1):
        if day <= learning_days:
            start = (day - 1) * per_day
            end = start + per_day
            day_chunks = valid_chunks[start:end]

            grouped: Dict[str, List[int]] = {}

            for chunk in day_chunks:
                file_name = get_chunk_file_name(chunk)
                grouped.setdefault(file_name, [])
                grouped[file_name].append(int(chunk.get("page", 0)))

            scope_parts = []

            for file_name, pages in grouped.items():
                scope_parts.append(f"{file_name} p.{compact_page_ranges(pages)}")

            scope = " / ".join(scope_parts) if scope_parts else "전체 누적 범위"

            main_file = get_chunk_file_name(day_chunks[0]) if day_chunks else "업로드 자료"
            goal = get_summary_hint(main_file, summaries)

            task = (
                "1) 해당 범위 핵심 요약 읽기\n"
                "2) 중요한 용어 3개 정리\n"
                "3) 이해 안 되는 부분을 질문으로 바꾸기\n"
                "4) 객관식/OX/빈칸 문제 중 2개 생성해서 확인"
            )

            check = "오늘 범위에서 핵심 개념 3개를 설명할 수 있으면 통과"

            if day % review_interval == 0:
                check += " + 이전 학습 내용 10분 누적 복습"

            rows.append({
                "Day": day,
                "구분": "진도 학습",
                "학습 범위": scope,
                "학습 목표": goal,
                "할 일": task,
                "점검": check
            })

        else:
            review_type = "누적 복습"

            if day == duration_days:
                review_type = "최종 점검"

            task = (
                "1) 지금까지의 파일별 핵심 요약 다시 읽기\n"
                "2) 헷갈리는 개념을 약점 개념에 입력해서 문제 생성\n"
                "3) 틀린 문제의 근거 페이지 다시 확인\n"
                "4) 빈칸/OX 문제로 빠르게 재점검"
            )

            if review_type == "최종 점검":
                task = (
                    "1) 전체 PDF 핵심 개념 목록 훑기\n"
                    "2) 파일별 핵심 요약을 기준으로 최종 복습\n"
                    "3) 약점 개념 중심으로 문제 5개 생성\n"
                    "4) 틀린 문제만 다시 정리"
                )

            rows.append({
                "Day": day,
                "구분": review_type,
                "학습 범위": "전체 누적 범위",
                "학습 목표": "학습한 자료를 잊지 않도록 누적 복습하고 약점 개념을 확인한다.",
                "할 일": task,
                "점검": "틀린 문제와 헷갈린 개념을 오답노트에 정리"
            })

    return pd.DataFrame(rows)


def roadmap_df_to_markdown(roadmap_df: pd.DataFrame, duration_days: int) -> str:
    md = f"# {duration_days}일 학습 로드맵\n\n"

    for _, row in roadmap_df.iterrows():
        md += f"## Day {row['Day']} - {row['구분']}\n\n"
        md += f"**학습 범위:** {row['학습 범위']}\n\n"
        md += f"**학습 목표:** {row['학습 목표']}\n\n"
        md += f"**할 일:**\n{row['할 일']}\n\n"
        md += f"**점검:** {row['점검']}\n\n"
        md += "---\n\n"

    return md.strip()


# =========================
# 7. 검증과 완화
# =========================

NEGATIVE_WORDS = ["옳지 않은", "틀린", "아닌", "잘못된", "해당하지 않는"]


def infer_polarity(question: str, model_polarity: str) -> str:
    if any(word in str(question) for word in NEGATIVE_WORDS):
        return "negative"

    if model_polarity in ["positive", "negative"]:
        return model_polarity

    return "positive"


def filter_source_pages(source_pages: Any, allowed_pages: List[int]) -> List[int]:
    allowed = sorted({int(p) for p in allowed_pages if p is not None})
    allowed_set = set(allowed)

    candidates = source_pages if isinstance(source_pages, list) else [source_pages]
    cleaned = []

    for page in candidates:
        text = str(page).strip()

        if ":" in text:
            continue

        match = re.search(r"\d+", text)

        if not match:
            continue

        p = int(match.group())

        if p in allowed_set:
            cleaned.append(p)

    return sorted(set(cleaned)) if cleaned else allowed


def clean_quiz_schema(
    quiz: Dict[str, Any],
    allowed_pages: List[int],
    expected_difficulty: int
) -> Dict[str, Any]:
    choices = quiz.get("choices", [])

    if not isinstance(choices, list):
        choices = []

    choices = [str(c).strip() for c in choices if str(c).strip()]
    answer = str(quiz.get("answer", "")).strip()

    clean = {
        "question_type": str(quiz.get("question_type", "multiple_choice")).strip(),
        "question_polarity": infer_polarity(
            str(quiz.get("question", "")),
            str(quiz.get("question_polarity", "positive"))
        ),
        "question": str(quiz.get("question", "")).strip(),
        "choices": choices,
        "answer": answer,
        "part_summary": str(quiz.get("part_summary", "")).strip(),
        "evidence_text": str(quiz.get("evidence_text", "")).strip(),
        "explanation": str(quiz.get("explanation", "")).strip(),
        "choice_explanations": [],
        "source_pages": filter_source_pages(quiz.get("source_pages", []), allowed_pages),
        "concept": str(quiz.get("concept", "")).strip(),
        "difficulty": int(expected_difficulty),
        "hint": str(quiz.get("hint", "")).strip(),
        "grading_criteria": quiz.get("grading_criteria", [])
    }

    # 빈칸 주관식 보정
    # 모델이 ____, ___처럼 다른 길이의 밑줄을 출력해도 표준 _____로 통일한다.
    if clean["question_type"] == "fill_blank":
        clean["question"] = normalize_blank_marker(clean["question"])
        clean["choices"] = []
        clean["choice_explanations"] = []
        return clean

    # OX 문제 보정
    if clean["question_type"] == "ox":
        clean["choices"] = ["O", "X"]

        answer_upper = str(clean["answer"]).strip().upper()

        if answer_upper in ["O", "○", "TRUE", "참", "맞다", "옳다"]:
            clean["answer"] = "O"
        elif answer_upper in ["X", "×", "FALSE", "거짓", "틀리다", "아니다"]:
            clean["answer"] = "X"

        # OX 문제는 O/X 자체가 보기이므로 별도의 보기별 해설을 보여주지 않는다.
        clean["choice_explanations"] = []
        return clean

    raw_explanations = quiz.get("choice_explanations", [])

    if not isinstance(raw_explanations, list):
        raw_explanations = []

    explanation_map = {}

    for item in raw_explanations:
        if isinstance(item, dict) and item.get("choice"):
            explanation_map[normalize_for_match(item.get("choice"))] = item

    aligned = []

    for choice in choices:
        item = explanation_map.get(normalize_for_match(choice), {})
        aligned.append({
            "choice": choice,
            "is_answer": choice == answer,
            "is_factually_correct": bool(item.get("is_factually_correct", choice == answer)),
            "explanation": str(item.get("explanation", "")).strip()
        })

    clean["choice_explanations"] = aligned

    return clean


def basic_validate_quiz(quiz: Dict[str, Any]) -> Tuple[bool, str]:
    if not quiz.get("question"):
        return False, "문제가 비어 있습니다."

    answer = quiz.get("answer", "")

    if not answer:
        return False, "정답 또는 모범 답안이 비어 있습니다."

    question_type = quiz.get("question_type", "multiple_choice")

    # OX 문제 검증
    if question_type == "ox":
        choices = quiz.get("choices", [])

        if choices != ["O", "X"]:
            return False, "OX 문제의 보기는 ['O', 'X']여야 합니다."

        if answer not in ["O", "X"]:
            return False, "OX 문제의 정답은 O 또는 X여야 합니다."

        if len(str(quiz.get("question", ""))) > 220:
            return False, "문제 문장이 너무 깁니다."

        return True, "OX 문제 검증 통과"

    # 빈칸 주관식 검증
    if question_type == "fill_blank":
        question = str(quiz.get("question", ""))

        # 모델이 ___, ____, _____ 중 무엇을 출력해도 3개 이상의 밑줄이면 빈칸으로 인정한다.
        if not has_blank_marker(question):
            return False, "빈칸 주관식 문제에는 ___ 또는 _____ 같은 빈칸 표시가 필요합니다."

        if len(str(answer)) > 50:
            return False, "빈칸 정답이 너무 깁니다."

        return True, "빈칸 주관식 검증 통과"

    # 객관식 검증
    choices = quiz.get("choices", [])

    if not isinstance(choices, list) or len(choices) != 4:
        return False, "보기 4개가 필요합니다."

    if answer not in choices:
        return False, "정답이 보기 중 하나와 정확히 일치하지 않습니다."

    if len(set(normalize_for_match(c) for c in choices)) != 4:
        return False, "보기가 중복됩니다."

    if len(str(quiz.get("question", ""))) > 180:
        return False, "문제 문장이 너무 깁니다."

    if any(len(str(c)) > 130 for c in choices):
        return False, "보기 중 하나가 너무 깁니다."

    return True, "객관식 검증 통과"


def grounded_warning(quiz: Dict[str, Any], context_text: str, banned_terms: List[str]) -> List[str]:
    # 실패시키기보다는 경고로 보여준다. 이게 너무 엄격하면 계속 generation_error가 난다.
    warnings = []
    generated = normalize_for_match(
        " ".join([
            quiz.get("question", ""),
            quiz.get("answer", ""),
            quiz.get("part_summary", ""),
            quiz.get("evidence_text", ""),
            quiz.get("explanation", "")
        ])
    )
    context = normalize_for_match(context_text)

    for term in banned_terms:
        nt = normalize_for_match(term)

        if nt and nt in generated and nt not in context:
            warnings.append(f"자료별 금지 표현이 생성 결과에 포함됨: {term}")

    # 정답이 아주 짧은 경우에는 근거 안에 직접 등장하는지 확인
    answer = str(quiz.get("answer", ""))

    if len(answer) <= 25 and normalize_for_match(answer) not in context:
        warnings.append("정답이 검색된 근거 문맥에 직접 등장하지 않을 수 있음")

    return warnings


def make_error_quiz(message: str, raw_output: str, allowed_pages: List[int]) -> Dict[str, Any]:
    return {
        "question_polarity": "positive",
        "question": "문제 생성 실패",
        "choices": [],
        "answer": "",
        "part_summary": "",
        "evidence_text": raw_output[:1500] if raw_output else "",
        "explanation": message,
        "choice_explanations": [],
        "source_pages": allowed_pages,
        "concept": "generation_error",
        "difficulty": 1,
        "hint": "Top-k를 1~2로 조정하거나, 모델/출력 길이 설정을 바꿔 다시 시도하세요.",
        "warnings": []
    }


def generate_quiz_with_ollama(
    prompt: str,
    model_name: str,
    ollama_url: str,
    allowed_pages: List[int],
    expected_difficulty: int,
    context_text: str,
    num_predict: int,
    stream_read_timeout: int,
    banned_terms: List[str]
) -> Dict[str, Any]:
    # qwen3 계열은 thinking 모드 때문에 response가 비어 보일 수 있다.
    # /no_think와 think=False를 함께 넣어 가능한 경우 reasoning 출력을 끈다.
    prompt_for_ollama = "/no_think\n" + prompt

    payload = {
        "model": model_name,
        "prompt": prompt_for_ollama,
        "stream": True,
        "format": "json",
        "think": False,
        "options": {
            "temperature": 0.1,
            "num_predict": int(num_predict)
        }
    }

    try:
        raw = call_ollama_streaming(
            ollama_url=ollama_url,
            payload=payload,
            stream_read_timeout=stream_read_timeout
        )

    except requests.exceptions.ReadTimeout:
        return make_error_quiz(
            "Ollama가 일정 시간 동안 새 토큰을 보내지 못했습니다. 모델이 느리거나 프롬프트가 긴 상태일 수 있습니다.",
            "",
            allowed_pages
        )

    except Exception as e:
        return make_error_quiz(
            f"Ollama 호출 실패: {e}",
            "",
            allowed_pages
        )

    try:
        raw_quiz = parse_ollama_json(raw)

    except Exception as e:
        return make_error_quiz(f"JSON 파싱 실패: {e}", raw, allowed_pages)

    quiz = clean_quiz_schema(
        raw_quiz,
        allowed_pages=allowed_pages,
        expected_difficulty=expected_difficulty
    )

    ok, msg = basic_validate_quiz(quiz)

    if not ok:
        return make_error_quiz(f"기본 형식 검증 실패: {msg}", raw, allowed_pages)

    quiz["warnings"] = grounded_warning(quiz, context_text, banned_terms)

    return quiz






# =========================
# 7-0. 학생 풀이 채점 / 학습 상태 분석 / 보충 문제 생성
# =========================

def init_learning_state() -> None:
    """
    학생 풀이 기록과 채점 결과를 session_state에 준비한다.
    실제 서비스에서는 이 부분을 SQLite/PostgreSQL로 옮기면 된다.
    """
    if "attempts" not in st.session_state:
        st.session_state["attempts"] = []

    if "graded_results" not in st.session_state:
        st.session_state["graded_results"] = {}

    if "remedial_quizzes" not in st.session_state:
        st.session_state["remedial_quizzes"] = {}


def normalize_answer_for_grading(text: Any) -> str:
    """
    채점 비교용 정규화.
    띄어쓰기, 대소문자, 일부 기호 차이를 완화한다.
    """
    text = str(text).strip()
    text = text.replace("○", "O").replace("×", "X")
    text = text.replace("정답:", "").replace("답:", "")
    return normalize_for_match(text)


def normalize_ox_answer(text: Any) -> str:
    value = str(text).strip().upper()

    if value in ["O", "○", "TRUE", "참", "맞다", "옳다", "1"]:
        return "O"

    if value in ["X", "×", "FALSE", "거짓", "틀리다", "아니다", "2"]:
        return "X"

    return value


def grade_student_answer(quiz: Dict[str, Any], student_answer: Any) -> bool:
    question_type = quiz.get("question_type", "multiple_choice")
    correct_answer = str(quiz.get("answer", "")).strip()

    if question_type == "ox":
        return normalize_ox_answer(student_answer) == normalize_ox_answer(correct_answer)

    if question_type == "multiple_choice":
        return normalize_answer_for_grading(student_answer) == normalize_answer_for_grading(correct_answer)

    if question_type == "fill_blank":
        student_norm = normalize_answer_for_grading(student_answer)
        answer_norm = normalize_answer_for_grading(correct_answer)

        if not student_norm or not answer_norm:
            return False

        # 빈칸 답은 보통 단어/짧은 구절이므로 기본은 정규화 후 완전일치.
        if student_norm == answer_norm:
            return True

        # 학생이 짧은 설명문을 같이 쓴 경우를 약간 허용한다.
        if len(answer_norm) >= 2 and answer_norm in student_norm:
            return True

        return False

    return normalize_answer_for_grading(student_answer) == normalize_answer_for_grading(correct_answer)


def build_attempt_record(
    quiz: Dict[str, Any],
    student_answer: Any,
    is_correct: bool,
    confidence: int,
    used_hint: bool,
    origin: str,
    origin_key: str
) -> Dict[str, Any]:
    concept = str(quiz.get("concept", "")).strip() or "기타 개념"

    return {
        "attempt_id": f"attempt_{len(st.session_state.get('attempts', [])) + 1}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "origin": origin,
        "origin_key": origin_key,
        "file_name": quiz.get("file_name", ""),
        "material_id": quiz.get("material_id", ""),
        "roadmap_day": quiz.get("roadmap_day", ""),
        "question_type": quiz.get("question_type", "multiple_choice"),
        "question": quiz.get("question", ""),
        "answer": quiz.get("answer", ""),
        "student_answer": str(student_answer).strip(),
        "is_correct": bool(is_correct),
        "concept": concept,
        "difficulty": quiz.get("difficulty", ""),
        "confidence": int(confidence),
        "used_hint": bool(used_hint),
        "source_pages": quiz.get("source_pages", []),
        "explanation": quiz.get("explanation", ""),
        "evidence_text": quiz.get("evidence_text", "")
    }


def save_attempt_record(attempt: Dict[str, Any]) -> None:
    init_learning_state()
    st.session_state["attempts"].append(attempt)


def render_quiz_feedback(quiz: Dict[str, Any], is_correct: bool) -> None:
    if is_correct:
        st.success("정답입니다.")
    else:
        st.error("오답입니다.")

    st.markdown("### 정답")
    st.success(quiz.get("answer", ""))

    if quiz.get("explanation"):
        st.markdown("### 해설")
        st.write(quiz.get("explanation", ""))

    if quiz.get("evidence_text"):
        st.markdown("### 정답 근거")
        st.info(quiz.get("evidence_text", ""))

    if quiz.get("source_pages"):
        st.markdown("### 근거 페이지")
        st.write(", ".join(str(page) for page in quiz.get("source_pages", [])))

    if quiz.get("concept"):
        st.markdown("### 저장된 개념")
        st.write({
            "concept": quiz.get("concept", ""),
            "difficulty": quiz.get("difficulty", "")
        })


def render_interactive_quiz(
    quiz: Dict[str, Any],
    idx: int,
    origin: str,
    origin_key: str,
    title: str = "문제"
) -> None:
    """
    문제를 먼저 보여주고, 학생이 답을 입력/선택한 뒤 채점한다.
    채점 후에만 정답, 해설, 근거를 공개하고 풀이 기록을 저장한다.
    """
    init_learning_state()

    st.markdown(f"#### {title} {idx}")

    if quiz.get("concept") == "generation_error":
        st.error("문제 생성에 실패했습니다.")
        st.write(quiz.get("explanation", ""))
        if quiz.get("evidence_text"):
            with st.expander("원본 출력 보기", expanded=False):
                st.write(quiz.get("evidence_text", ""))
        return

    question_type_value = quiz.get("question_type", "multiple_choice")
    choices = quiz.get("choices", [])
    key_base = make_safe_id(f"{origin}_{origin_key}_{idx}_{str(quiz.get('question', ''))[:40]}")
    result_key = f"graded_result_{key_base}"

    st.markdown(f"**Q. {quiz.get('question', '')}**")

    if question_type_value == "multiple_choice" and choices:
        for c_idx, choice in enumerate(choices, start=1):
            st.write(f"{c_idx}. {choice}")
    elif question_type_value == "ox":
        st.info("OX 문항입니다. 설명이 옳으면 O, 틀리면 X를 선택하면 됩니다.")
        st.write("1. O")
        st.write("2. X")
    elif question_type_value == "fill_blank":
        st.info("빈칸 주관식 문항입니다. 빈칸에 들어갈 단어 또는 짧은 구절을 입력하세요.")

    if quiz.get("hint"):
        with st.expander("힌트 보기", expanded=False):
            st.info(quiz.get("hint", ""))

    with st.form(key=f"answer_form_{key_base}"):
        if question_type_value == "multiple_choice" and choices:
            student_answer = st.radio(
                "답을 선택하세요.",
                choices,
                index=None,
                key=f"student_answer_{key_base}"
            )
        elif question_type_value == "ox":
            student_answer = st.radio(
                "답을 선택하세요.",
                ["O", "X"],
                index=None,
                key=f"student_answer_{key_base}"
            )
        else:
            student_answer = st.text_input(
                "답을 입력하세요.",
                key=f"student_answer_{key_base}",
                placeholder="예: 데이터, queue, 과전법"
            )

        confidence = st.slider(
            "정답 확신도",
            min_value=1,
            max_value=5,
            value=3,
            help="1은 거의 모르겠음, 5는 확실히 알고 있음입니다.",
            key=f"confidence_{key_base}"
        )

        used_hint = st.checkbox(
            "힌트를 보고 풀었어요.",
            key=f"used_hint_{key_base}"
        )

        submitted = st.form_submit_button("채점하기")

    if submitted:
        if student_answer is None or not str(student_answer).strip():
            st.warning("답을 입력하거나 선택하세요.")
        else:
            is_correct = grade_student_answer(quiz, student_answer)
            attempt = build_attempt_record(
                quiz=quiz,
                student_answer=student_answer,
                is_correct=is_correct,
                confidence=confidence,
                used_hint=used_hint,
                origin=origin,
                origin_key=origin_key
            )
            save_attempt_record(attempt)
            st.session_state["graded_results"][result_key] = attempt

    if st.session_state["graded_results"].get(result_key):
        attempt = st.session_state["graded_results"][result_key]
        render_quiz_feedback(quiz, bool(attempt.get("is_correct")))


def analyze_concept_mastery(attempts: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    풀이 기록을 바탕으로 개념별 학습 상태를 계산한다.
    - 정답률
    - 확신도
    - 힌트 사용 여부
    - 오답 횟수
    를 종합하여 취약/틀림/흔들림/학습됨으로 분류한다.
    """
    if not attempts:
        return pd.DataFrame()

    stats: Dict[str, Dict[str, Any]] = {}

    for attempt in attempts:
        concept = str(attempt.get("concept", "기타 개념")).strip() or "기타 개념"

        if concept not in stats:
            stats[concept] = {
                "concept": concept,
                "total": 0,
                "correct": 0,
                "wrong": 0,
                "confidence_sum": 0,
                "hint_used": 0,
                "last_attempt": ""
            }

        stat = stats[concept]
        stat["total"] += 1

        if attempt.get("is_correct"):
            stat["correct"] += 1
        else:
            stat["wrong"] += 1

        stat["confidence_sum"] += int(attempt.get("confidence", 3))

        if attempt.get("used_hint"):
            stat["hint_used"] += 1

        stat["last_attempt"] = attempt.get("timestamp", "")

    rows = []

    for concept, stat in stats.items():
        total = max(1, stat["total"])
        accuracy = stat["correct"] / total
        avg_confidence = stat["confidence_sum"] / total
        hint_rate = stat["hint_used"] / total
        no_hint_rate = 1 - hint_rate

        mastery_score = (
            accuracy * 60
            + (avg_confidence / 5) * 25
            + no_hint_rate * 15
        )

        if stat["wrong"] > 0 and accuracy <= 0.5:
            status = "틀린 개념"
            recommendation = "오답 원인을 확인하고 같은 개념의 보충 문제를 다시 풀기"
        elif mastery_score < 60:
            status = "취약 개념"
            recommendation = "핵심 요약을 다시 읽고 쉬운 문제부터 다시 풀기"
        elif avg_confidence < 3.5 or stat["hint_used"] > 0:
            status = "흔들리는 개념"
            recommendation = "힌트 없이 비슷한 문제를 1~3개 더 풀기"
        else:
            status = "학습됨"
            recommendation = "짧은 복습만 유지"

        rows.append({
            "concept": concept,
            "status": status,
            "mastery_score": round(mastery_score, 1),
            "total": stat["total"],
            "correct": stat["correct"],
            "wrong": stat["wrong"],
            "accuracy": round(accuracy, 2),
            "avg_confidence": round(avg_confidence, 2),
            "hint_used": stat["hint_used"],
            "recommendation": recommendation,
            "last_attempt": stat["last_attempt"]
        })

    return pd.DataFrame(rows).sort_values(
        by=["mastery_score", "wrong"],
        ascending=[True, False]
    )


def get_target_concepts_for_remedial(mastery_df: pd.DataFrame) -> List[str]:
    if mastery_df is None or mastery_df.empty:
        return []

    target_df = mastery_df[
        mastery_df["status"].isin(["틀린 개념", "취약 개념", "흔들리는 개념"])
    ]

    if target_df.empty:
        target_df = mastery_df

    return target_df["concept"].dropna().astype(str).tolist()


def build_remedial_user_query(
    target_concept: str,
    quiz_index: int,
    quiz_count: int,
    question_type: str
) -> str:
    return (
        f"학생이 '{target_concept}' 개념에서 부족함을 보였어. "
        f"이 개념을 다시 학습할 수 있는 보충 문제를 만들어줘. "
        f"현재 문제는 {quiz_count}개 중 {quiz_index + 1}번째 문제이고, "
        f"문제 유형은 {question_type}이야. "
        "검색된 교안 근거 안에서만 만들고, 기존 문제와 최대한 중복되지 않게 만들어줘."
    )


def generate_remedial_quizzes_with_ollama(
    ollama_url: str,
    model_name: str,
    collection,
    embedding_model,
    target_concept: str,
    quiz_count: int,
    student_level: str,
    question_type_mode: str,
    top_k: int,
    extra_keywords: List[str],
    num_predict: int,
    stream_read_timeout: int,
    banned_terms: List[str]
) -> List[Dict[str, Any]]:
    """
    풀이 기록에서 발견된 취약/틀린/흔들리는 개념을 대상으로 보충 문제를 생성한다.
    학생은 1개, 3개, 5개 중 원하는 개수를 선택할 수 있다.
    """
    quiz_count = int(quiz_count)
    search_query = f"{target_concept} 개념 보충 문제"
    boosted_keywords = list(extra_keywords or [])

    if target_concept and target_concept not in boosted_keywords:
        boosted_keywords.append(target_concept)

    retrieved = search_chunks(
        collection=collection,
        query=search_query,
        embedding_model=embedding_model,
        top_k=max(int(top_k), quiz_count),
        extra_keywords=boosted_keywords
    )

    if not retrieved:
        return [make_error_quiz("보충 문제를 만들 관련 청크를 찾지 못했습니다.", "", [])]

    selected_chunks = select_representative_chunks_for_quizzes(retrieved, quiz_count)
    quizzes = []

    for idx, chunk in enumerate(selected_chunks):
        q_type = get_review_question_type(idx, question_type_mode)
        user_query = build_remedial_user_query(
            target_concept=target_concept,
            quiz_index=idx,
            quiz_count=quiz_count,
            question_type=q_type
        )

        retrieved_chunks = [chunk]
        allowed_pages = [
            int(item.get("page"))
            for item in retrieved_chunks
            if item.get("page") is not None
        ]
        context_text = "\n\n".join(str(item.get("text", "")) for item in retrieved_chunks)

        prompt = build_quiz_prompt(
            user_query=user_query,
            retrieved_chunks=retrieved_chunks,
            student_level=student_level,
            weak_concept=target_concept,
            question_type=q_type,
            question_direction="자동(긍정형 우선)",
            question_intent="general"
        )

        quiz = generate_quiz_with_ollama(
            prompt=prompt,
            model_name=model_name,
            ollama_url=ollama_url,
            allowed_pages=allowed_pages,
            expected_difficulty=get_level_number(student_level),
            context_text=context_text,
            num_predict=num_predict,
            stream_read_timeout=stream_read_timeout,
            banned_terms=banned_terms
        )

        quiz["target_concept"] = target_concept
        quiz["remedial_quiz_index"] = idx + 1
        quiz["remedial_quiz_total"] = quiz_count
        quiz["review_question_type_mode"] = question_type_mode
        quizzes.append(quiz)

    return quizzes


# =========================
# 7-1. 파일별 학습 확인 문제 생성
# =========================

def select_representative_chunks_for_quizzes(
    file_chunks: List[Dict[str, Any]],
    quiz_count: int
) -> List[Dict[str, Any]]:
    """
    파일별 학습 확인 문제를 만들 때 사용할 대표 청크를 고른다.
    - 목차/표지성 청크 제외
    - quality_score가 낮은 청크 제외
    - 파일 앞/중간/뒤에서 골고루 선택
    """
    quiz_count = int(quiz_count)
    good_chunks = []

    for chunk in file_chunks:
        text = str(chunk.get("text", "")).strip()

        if not text:
            continue

        if is_table_of_contents_like(text):
            continue

        if int(chunk.get("quality_score", 0)) < 25:
            continue

        good_chunks.append(chunk)

    if not good_chunks:
        good_chunks = [chunk for chunk in file_chunks if str(chunk.get("text", "")).strip()]

    good_chunks = sorted(
        good_chunks,
        key=lambda x: (
            int(x.get("page", 0)),
            str(x.get("chunk_id", ""))
        )
    )

    if not good_chunks:
        return []

    selected = []

    if len(good_chunks) >= quiz_count:
        step = len(good_chunks) / quiz_count

        for i in range(quiz_count):
            idx = min(int(i * step), len(good_chunks) - 1)
            selected.append(good_chunks[idx])
    else:
        # 청크 수가 문제 수보다 적으면 순환 사용한다.
        for i in range(quiz_count):
            selected.append(good_chunks[i % len(good_chunks)])

    return selected


def get_review_question_type(index: int, mode: str) -> str:
    """
    파일별 학습 확인 문제의 유형을 결정한다.
    학생 요청 기반 문제 생성은 그대로 1개만 유지하고,
    이 함수는 파일별 확인 문제 세트에서만 사용한다.
    """
    if mode == "4지선다 객관식만":
        return "4지선다 객관식"

    if mode == "OX 문제만":
        return "OX 문제"

    if mode == "빈칸 주관식만":
        return "빈칸 주관식"

    mixed = ["4지선다 객관식", "OX 문제", "빈칸 주관식", "4지선다 객관식", "OX 문제"]
    return mixed[index % len(mixed)]


def find_material_group(
    chunks: List[Dict[str, Any]],
    material_id: str
) -> Optional[Dict[str, Any]]:
    for item in group_chunks_by_material(chunks):
        if str(item.get("material_id", "")) == str(material_id):
            return item

    return None


def build_file_review_user_query(
    file_name: str,
    quiz_index: int,
    quiz_count: int,
    question_type: str
) -> str:
    return (
        f"'{file_name}' 자료의 핵심 내용을 확인하는 학습 확인 문제를 만들어줘. "
        f"현재 문제는 {quiz_count}개 중 {quiz_index + 1}번째 문제이고, "
        f"문제 유형은 {question_type}이야. "
        "선택된 교안 근거의 핵심 개념을 바탕으로 중복되지 않는 문제를 만들어줘."
    )


def generate_file_review_quizzes_with_ollama(
    ollama_url: str,
    model_name: str,
    chunks: List[Dict[str, Any]],
    material_id: str,
    quiz_count: int,
    student_level: str,
    question_type_mode: str,
    num_predict: int,
    stream_read_timeout: int,
    banned_terms: List[str]
) -> List[Dict[str, Any]]:
    """
    특정 PDF 파일 하나에 대해 학생이 선택한 개수만큼 학습 확인 문제를 생성한다.
    - 학생 요청 기반 문제 생성과 별개로 동작한다.
    - 학생 요청 기반 문제 생성은 기존처럼 1개만 생성된다.
    - 이 함수는 파일별 핵심 요약 이후, 학생이 원할 때 1/3/5개 문제를 추가 생성한다.
    """
    material_group = find_material_group(chunks, material_id)

    if not material_group:
        return [make_error_quiz("해당 파일의 청크를 찾지 못했습니다.", "", [])]

    file_name = str(material_group.get("file_name", "uploaded.pdf"))
    file_chunks = material_group.get("chunks", [])
    selected_chunks = select_representative_chunks_for_quizzes(file_chunks, quiz_count)

    if not selected_chunks:
        return [make_error_quiz("학습 확인 문제를 만들 수 있는 본문 청크가 없습니다.", "", [])]

    quizzes = []

    for idx, chunk in enumerate(selected_chunks):
        q_type = get_review_question_type(idx, question_type_mode)
        user_query = build_file_review_user_query(
            file_name=file_name,
            quiz_index=idx,
            quiz_count=quiz_count,
            question_type=q_type
        )

        retrieved_chunks = [chunk]
        allowed_pages = [
            int(item.get("page"))
            for item in retrieved_chunks
            if item.get("page") is not None
        ]
        context_text = "\n\n".join(str(item.get("text", "")) for item in retrieved_chunks)
        question_intent = "general"

        prompt = build_quiz_prompt(
            user_query=user_query,
            retrieved_chunks=retrieved_chunks,
            student_level=student_level,
            weak_concept="",
            question_type=q_type,
            question_direction="자동(긍정형 우선)",
            question_intent=question_intent
        )

        quiz = generate_quiz_with_ollama(
            prompt=prompt,
            model_name=model_name,
            ollama_url=ollama_url,
            allowed_pages=allowed_pages,
            expected_difficulty=get_level_number(student_level),
            context_text=context_text,
            num_predict=num_predict,
            stream_read_timeout=stream_read_timeout,
            banned_terms=banned_terms
        )

        quiz["review_quiz_index"] = idx + 1
        quiz["review_quiz_total"] = quiz_count
        quiz["file_name"] = file_name
        quiz["material_id"] = material_id
        quiz["review_question_type_mode"] = question_type_mode

        quizzes.append(quiz)

    return quizzes


def quiz_to_markdown(quiz: Dict[str, Any], idx: int) -> str:
    question_type = quiz.get("question_type", "multiple_choice")
    md = f"## 문제 {idx}\n\n"
    md += f"**유형:** {question_type}\n\n"
    md += f"**Q. {quiz.get('question', '')}**\n\n"

    choices = quiz.get("choices", [])

    if question_type == "multiple_choice" and choices:
        for c_idx, choice in enumerate(choices, start=1):
            md += f"{c_idx}. {choice}\n"
        md += "\n"
    elif question_type == "ox":
        md += "1. O\n2. X\n\n"
    elif question_type == "fill_blank":
        md += "빈칸에 들어갈 단어 또는 짧은 구절을 작성하세요.\n\n"

    md += f"**정답:** {quiz.get('answer', '')}\n\n"

    if quiz.get("explanation"):
        md += f"**해설:** {quiz.get('explanation', '')}\n\n"

    if quiz.get("evidence_text"):
        md += f"**정답 근거:** {quiz.get('evidence_text', '')}\n\n"

    if quiz.get("source_pages"):
        md += "**근거 페이지:** " + ", ".join(str(p) for p in quiz.get("source_pages", [])) + "\n\n"

    if quiz.get("concept"):
        md += f"**핵심 개념:** {quiz.get('concept', '')}\n\n"

    return md.strip()


def review_quizzes_to_markdown(file_name: str, quizzes: List[Dict[str, Any]]) -> str:
    md = f"# {file_name} 학습 확인 문제\n\n"

    for idx, quiz in enumerate(quizzes, start=1):
        md += quiz_to_markdown(quiz, idx)
        md += "\n\n---\n\n"

    return md.strip()


# =========================
# 7-2. 로드맵 Day별 학습 확인 문제 생성
# =========================

def get_valid_learning_chunks_for_roadmap(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    로드맵과 로드맵 기반 문제 생성에서 공통으로 사용할 학습용 청크를 고른다.
    """
    valid_chunks = []

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()

        if not text:
            continue

        if is_table_of_contents_like(text):
            continue

        if int(chunk.get("quality_score", 0)) < 25:
            continue

        valid_chunks.append(chunk)

    if not valid_chunks:
        valid_chunks = [chunk for chunk in chunks if str(chunk.get("text", "")).strip()]

    valid_chunks = sorted(
        valid_chunks,
        key=lambda x: (
            get_chunk_file_name(x),
            int(x.get("page", 0)),
            str(x.get("chunk_id", ""))
        )
    )

    return valid_chunks


def get_roadmap_learning_days(total_chunks: int, duration_days: int) -> int:
    """
    build_learning_roadmap_df와 동일한 방식으로 진도 학습일 수를 계산한다.
    """
    duration_days = int(duration_days)

    if total_chunks <= 0:
        return 1

    learning_days = max(1, int(duration_days * 0.7))
    learning_days = min(learning_days, duration_days)
    learning_days = min(learning_days, total_chunks)

    return max(1, learning_days)


def get_chunks_for_roadmap_day(
    chunks: List[Dict[str, Any]],
    duration_days: int,
    day: int
) -> List[Dict[str, Any]]:
    """
    선택한 로드맵 Day의 학습 범위에 해당하는 청크를 반환한다.
    - 진도 학습일: 해당 날짜에 배정된 청크만 사용
    - 누적 복습/최종 점검일: 전체 누적 범위 청크를 사용
    """
    valid_chunks = get_valid_learning_chunks_for_roadmap(chunks)

    if not valid_chunks:
        return []

    duration_days = int(duration_days)
    day = int(day)
    learning_days = get_roadmap_learning_days(len(valid_chunks), duration_days)
    per_day = max(1, (len(valid_chunks) + learning_days - 1) // learning_days)

    if day <= learning_days:
        start = (day - 1) * per_day
        end = start + per_day
        return valid_chunks[start:end]

    # 복습/최종 점검일은 지금까지의 누적 범위를 대상으로 한다.
    return valid_chunks


def describe_chunks_scope(chunks: List[Dict[str, Any]]) -> str:
    """
    청크 목록을 '파일명 p.1-3 / 파일명2 p.4' 형태의 학습 범위 텍스트로 바꾼다.
    """
    grouped: Dict[str, List[int]] = {}

    for chunk in chunks:
        file_name = get_chunk_file_name(chunk)
        grouped.setdefault(file_name, [])
        grouped[file_name].append(int(chunk.get("page", 0)))

    scope_parts = []

    for file_name, pages in grouped.items():
        scope_parts.append(f"{file_name} p.{compact_page_ranges(pages)}")

    return " / ".join(scope_parts) if scope_parts else "전체 누적 범위"


def build_roadmap_day_user_query(
    day: int,
    duration_days: int,
    scope_text: str,
    quiz_index: int,
    quiz_count: int,
    question_type: str
) -> str:
    return (
        f"{duration_days}일 학습 로드맵의 Day {day} 학습 범위에 해당하는 학습 확인 문제를 만들어줘. "
        f"학습 범위는 {scope_text}이고, 현재 문제는 {quiz_count}개 중 {quiz_index + 1}번째 문제야. "
        f"문제 유형은 {question_type}이야. "
        "선택된 교안 근거 안에서만 핵심 개념을 확인하는 문제를 만들어줘. "
        "같은 날짜 안에서 앞 문제와 최대한 중복되지 않게 만들어줘."
    )


def generate_roadmap_day_quizzes_with_ollama(
    ollama_url: str,
    model_name: str,
    chunks: List[Dict[str, Any]],
    duration_days: int,
    day: int,
    quiz_count: int,
    student_level: str,
    question_type_mode: str,
    num_predict: int,
    stream_read_timeout: int,
    banned_terms: List[str]
) -> List[Dict[str, Any]]:
    """
    선택한 로드맵 Day의 학습 범위에 대해 학생이 선택한 개수만큼 문제를 생성한다.
    학생 요청 기반 문제 생성은 기존처럼 1개만 생성되고, 이 함수와 분리되어 동작한다.
    """
    day_chunks = get_chunks_for_roadmap_day(
        chunks=chunks,
        duration_days=duration_days,
        day=day
    )

    if not day_chunks:
        return [make_error_quiz("해당 Day의 학습 범위 청크를 찾지 못했습니다.", "", [])]

    selected_chunks = select_representative_chunks_for_quizzes(day_chunks, quiz_count)

    if not selected_chunks:
        return [make_error_quiz("로드맵 학습 확인 문제를 만들 수 있는 본문 청크가 없습니다.", "", [])]

    scope_text = describe_chunks_scope(day_chunks)
    quizzes = []

    for idx, chunk in enumerate(selected_chunks):
        q_type = get_review_question_type(idx, question_type_mode)

        user_query = build_roadmap_day_user_query(
            day=day,
            duration_days=duration_days,
            scope_text=scope_text,
            quiz_index=idx,
            quiz_count=quiz_count,
            question_type=q_type
        )

        retrieved_chunks = [chunk]
        allowed_pages = [
            int(item.get("page"))
            for item in retrieved_chunks
            if item.get("page") is not None
        ]
        context_text = "\n\n".join(str(item.get("text", "")) for item in retrieved_chunks)
        question_intent = "general"

        prompt = build_quiz_prompt(
            user_query=user_query,
            retrieved_chunks=retrieved_chunks,
            student_level=student_level,
            weak_concept="",
            question_type=q_type,
            question_direction="자동(긍정형 우선)",
            question_intent=question_intent
        )

        quiz = generate_quiz_with_ollama(
            prompt=prompt,
            model_name=model_name,
            ollama_url=ollama_url,
            allowed_pages=allowed_pages,
            expected_difficulty=get_level_number(student_level),
            context_text=context_text,
            num_predict=num_predict,
            stream_read_timeout=stream_read_timeout,
            banned_terms=banned_terms
        )

        quiz["roadmap_day"] = int(day)
        quiz["roadmap_duration_days"] = int(duration_days)
        quiz["roadmap_scope"] = scope_text
        quiz["roadmap_quiz_index"] = idx + 1
        quiz["roadmap_quiz_total"] = quiz_count
        quiz["review_question_type_mode"] = question_type_mode

        quizzes.append(quiz)

    return quizzes


def roadmap_day_quizzes_to_markdown(
    duration_days: int,
    day: int,
    scope_text: str,
    quizzes: List[Dict[str, Any]]
) -> str:
    md = f"# {duration_days}일 로드맵 Day {day} 학습 확인 문제\n\n"
    md += f"**학습 범위:** {scope_text}\n\n"

    for idx, quiz in enumerate(quizzes, start=1):
        md += quiz_to_markdown(quiz, idx)
        md += "\n\n---\n\n"

    return md.strip()


def render_quiz_for_review(quiz: Dict[str, Any], idx: int) -> None:
    """
    파일별/로드맵/보충 학습 확인 문제를 화면에 출력하고,
    학생 답안을 받아 채점한 뒤 풀이 기록을 저장한다.
    """
    origin_parts = [
        str(quiz.get("material_id", "")),
        str(quiz.get("roadmap_duration_days", "")),
        str(quiz.get("roadmap_day", "")),
        str(quiz.get("target_concept", "")),
        str(quiz.get("review_quiz_index", "")),
        str(quiz.get("roadmap_quiz_index", "")),
        str(quiz.get("remedial_quiz_index", ""))
    ]
    origin_key = "_".join([part for part in origin_parts if part]) or f"review_{idx}"

    if quiz.get("target_concept"):
        origin = "보충 문제"
    elif quiz.get("roadmap_day"):
        origin = "로드맵 문제"
    elif quiz.get("material_id"):
        origin = "파일별 확인 문제"
    else:
        origin = "학습 확인 문제"

    render_interactive_quiz(
        quiz=quiz,
        idx=idx,
        origin=origin,
        origin_key=origin_key,
        title="문제"
    )

# =========================
# 8. 화면
# =========================

st.title("🧠 범용 PDF 기반 RAG 퀴즈 생성기")

st.write(
    "대학 강의자료 PDF를 업로드하면 텍스트를 청킹하고, 관련 근거를 검색한 뒤 Ollama 로컬 LLM으로 "
    "근거 기반 객관식 문제를 생성합니다."
)

with st.expander("전체 구현 흐름 보기", expanded=False):
    st.code(
        """
PDF 업로드
→ 페이지별 텍스트 추출
→ 청킹
→ 임베딩
→ ChromaDB 저장
→ 학생 요청 입력
→ 범용 RAG 검색
→ Ollama stream=True 문제 생성
→ 기본 형식 검증
→ 근거/해설/보기별 해설 출력
        """,
        language="text"
    )

init_learning_state()

st.sidebar.header("설정")

ollama_url = st.sidebar.text_input(
    "Ollama API URL",
    value="http://localhost:11434/api/generate"
)

ollama_model = st.sidebar.text_input(
    "Ollama 모델 이름",
    value="qwen3:4b",
    help="예: qwen3:4b, gemma3:4b, llama3.2:3b"
)

collection_name = st.sidebar.text_input(
    "ChromaDB 컬렉션 이름",
    value="pdf_lecture_chunks"
)

top_k = st.sidebar.slider(
    "검색할 청크 개수 Top-k",
    min_value=1,
    max_value=5,
    value=2,
    help="서로 다른 개념/단계가 섞이면 1로 줄이는 것이 안정적입니다."
)

chunk_size = st.sidebar.slider(
    "청크 크기",
    min_value=300,
    max_value=1500,
    value=800,
    step=100
)

chunk_overlap = st.sidebar.slider(
    "청크 오버랩",
    min_value=0,
    max_value=300,
    value=120,
    step=50
)

with st.sidebar.expander("고급 생성 설정", expanded=False):
    num_predict_setting = st.slider(
        "최대 생성 토큰 수",
        min_value=400,
        max_value=1600,
        value=900,
        step=100
    )

    stream_timeout_setting = st.slider(
        "Ollama 스트리밍 대기 시간(초)",
        min_value=30,
        max_value=240,
        value=120,
        step=30
    )

    extra_keywords_text = st.text_area(
        "검색 보강 키워드(선택)",
        value="",
        help="과목별 핵심 용어를 쉼표로 입력하면 검색에만 보강됩니다."
    )

    banned_terms_text = st.text_area(
        "자료별 금지 표현(선택)",
        value="",
        help="자주 발생하는 오타/환각 표현을 쉼표로 입력하세요. 기본은 비워두세요."
    )

extra_keywords = [
    item.strip()
    for item in re.split(r"[,;\n]+", extra_keywords_text)
    if item.strip()
]

banned_terms = [
    item.strip()
    for item in re.split(r"[,;\n]+", banned_terms_text)
    if item.strip()
]

if st.sidebar.button("Ollama 연결 확인"):
    if check_ollama_server(ollama_url):
        st.sidebar.success("Ollama 서버가 실행 중입니다.")
    else:
        st.sidebar.error("Ollama 서버에 연결할 수 없습니다.")


uploaded_pdfs = st.file_uploader(
    "PDF 강의자료를 업로드하세요. 여러 개를 한 번에 선택할 수 있습니다.",
    type=["pdf"],
    accept_multiple_files=True
)

if not uploaded_pdfs:
    st.info("먼저 PDF 강의자료를 1개 이상 업로드하세요.")
    st.stop()

try:
    st.subheader("1단계: PDF 텍스트 추출 및 청킹")

    if st.button("PDF 처리하기"):
        all_pages = []
        all_chunks = []
        material_rows = []

        with st.spinner("여러 PDF에서 텍스트를 추출하고 청킹하는 중입니다..."):
            for file_idx, uploaded_file in enumerate(uploaded_pdfs, start=1):
                file_bytes = uploaded_file.read()
                file_name = uploaded_file.name
                material_id = make_safe_id(f"m{file_idx}_{file_name}")

                pages = extract_text_from_pdf(
                    file_bytes=file_bytes,
                    file_name=file_name,
                    material_id=material_id
                )

                chunks = make_chunks_from_pages(
                    pages,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                all_pages.extend(pages)
                all_chunks.extend(chunks)

                material_rows.append({
                    "file_name": file_name,
                    "material_id": material_id,
                    "pages": len(pages),
                    "chars": sum(p["char_count"] for p in pages),
                    "chunks": len(chunks)
                })

        st.session_state["pages"] = all_pages
        st.session_state["chunks"] = all_chunks
        st.session_state["materials"] = material_rows

        st.success(f"PDF {len(material_rows)}개 처리 완료")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("업로드 파일 수", len(material_rows))

        with col2:
            st.metric("전체 페이지 수", len(all_pages))

        with col3:
            st.metric("전체 글자 수", sum(p["char_count"] for p in all_pages))

        with col4:
            st.metric("생성된 청크 수", len(all_chunks))

        st.markdown("#### 처리된 파일 목록")
        st.dataframe(pd.DataFrame(material_rows), use_container_width=True)

    if "chunks" not in st.session_state:
        st.stop()

    pages = st.session_state["pages"]
    chunks = st.session_state["chunks"]
    materials = st.session_state.get("materials", [])

    if materials:
        with st.expander("업로드된 PDF 목록 보기", expanded=False):
            st.dataframe(pd.DataFrame(materials), use_container_width=True)

    with st.expander("페이지별 텍스트 요약 보기", expanded=False):
        page_df = pd.DataFrame([
            {
                "file_name": p.get("file_name", ""),
                "material_id": p.get("material_id", ""),
                "page": p["page"],
                "char_count": p["char_count"],
                "preview": p["text"][:150].replace("\n", " ")
            }
            for p in pages
        ])
        st.dataframe(page_df, use_container_width=True)

    with st.expander("청크 미리보기", expanded=False):
        chunk_df = pd.DataFrame([
            {
                "chunk_id": c["chunk_id"],
                "file_name": c.get("file_name", ""),
                "material_id": c.get("material_id", ""),
                "page": c["page"],
                "char_count": c["char_count"],
                "quality_score": c["quality_score"],
                "preview": c["text"][:180].replace("\n", " ")
            }
            for c in chunks[:120]
        ])
        st.dataframe(chunk_df, use_container_width=True)

    st.subheader("파일별 핵심 요약")

    if st.button("파일별 핵심 요약 생성"):
        if not check_ollama_server(ollama_url):
            st.error("Ollama 서버에 연결할 수 없습니다.")
            st.stop()

        with st.spinner("PDF별 핵심 요약을 생성하는 중입니다..."):
            file_summaries = generate_document_summaries_by_file_with_ollama(
                ollama_url=ollama_url,
                model_name=ollama_model,
                chunks=chunks,
                num_predict=num_predict_setting,
                stream_read_timeout=stream_timeout_setting
            )

        st.session_state["file_summaries"] = file_summaries

    if st.session_state.get("file_summaries"):
        st.markdown("### PDF별 핵심 요약")

        if "file_review_quizzes" not in st.session_state:
            st.session_state["file_review_quizzes"] = {}

        for item in st.session_state["file_summaries"]:
            file_name = item.get("file_name", "uploaded.pdf")
            material_id = item.get("material_id", make_safe_id(file_name))
            summary = item.get("summary", "")

            with st.expander(f"📄 {file_name}", expanded=True):
                st.markdown(summary)

                st.markdown("#### 이 파일 학습 확인 문제 생성")
                st.caption("학생 요청 기반 문제 생성은 아래 3단계에서 계속 1개만 생성됩니다. 이 영역은 파일별 요약을 본 뒤 추가로 확인 문제를 만드는 기능입니다.")

                review_col1, review_col2, review_col3 = st.columns(3)

                with review_col1:
                    review_count = st.selectbox(
                        "생성할 문제 수",
                        [1, 3, 5],
                        index=0,
                        key=f"review_count_{material_id}"
                    )

                with review_col2:
                    review_type_mode = st.selectbox(
                        "문제 유형 구성",
                        ["혼합", "4지선다 객관식만", "OX 문제만", "빈칸 주관식만"],
                        index=0,
                        key=f"review_type_{material_id}"
                    )

                with review_col3:
                    review_level = st.selectbox(
                        "난이도",
                        ["Level 1 - 초급", "Level 2 - 중급", "Level 3 - 상급", "Level 4 - 심화"],
                        index=1,
                        key=f"review_level_{material_id}"
                    )

                if st.button("이 파일 학습 확인 문제 생성", key=f"generate_review_{material_id}"):
                    if not check_ollama_server(ollama_url):
                        st.error("Ollama 서버에 연결할 수 없습니다.")
                        st.stop()

                    with st.spinner(f"{file_name} 학습 확인 문제 {review_count}개를 생성하는 중입니다..."):
                        review_quizzes = generate_file_review_quizzes_with_ollama(
                            ollama_url=ollama_url,
                            model_name=ollama_model,
                            chunks=chunks,
                            material_id=material_id,
                            quiz_count=review_count,
                            student_level=review_level,
                            question_type_mode=review_type_mode,
                            num_predict=num_predict_setting,
                            stream_read_timeout=stream_timeout_setting,
                            banned_terms=banned_terms
                        )

                    st.session_state["file_review_quizzes"][material_id] = review_quizzes
                    st.success(f"{file_name} 학습 확인 문제 {len(review_quizzes)}개 생성 완료")

                if st.session_state["file_review_quizzes"].get(material_id):
                    review_quizzes = st.session_state["file_review_quizzes"][material_id]
                    st.markdown("#### 생성된 학습 확인 문제")

                    for quiz_idx, quiz in enumerate(review_quizzes, start=1):
                        render_quiz_for_review(quiz, quiz_idx)
                        st.divider()

                    review_md = review_quizzes_to_markdown(file_name, review_quizzes)

                    st.download_button(
                        label="이 파일 학습 확인 문제 다운로드",
                        data=review_md,
                        file_name=f"review_quizzes_{make_safe_id(file_name)}.md",
                        mime="text/markdown",
                        key=f"download_review_{material_id}"
                    )

        all_summaries_md = summaries_by_file_to_markdown(st.session_state["file_summaries"])

        st.download_button(
            label="파일별 전체 요약 다운로드",
            data=all_summaries_md,
            file_name="file_summaries_by_pdf.md",
            mime="text/markdown"
        )


    st.subheader("학습 로드맵 생성")

    roadmap_days = st.selectbox(
        "학습 기간을 선택하세요.",
        [3, 5, 7, 14, 28, 50, 100],
        index=2,
        help="학생이 직접 학습 기간을 선택하면, PDF 자료량에 맞춰 날짜별 학습 계획을 생성합니다."
    )

    if st.button("학습 로드맵 생성"):
        file_summaries_for_roadmap = st.session_state.get("file_summaries", [])

        roadmap_df = build_learning_roadmap_df(
            chunks=chunks,
            duration_days=roadmap_days,
            file_summaries=file_summaries_for_roadmap
        )

        st.session_state["learning_roadmap_df"] = roadmap_df
        st.session_state["learning_roadmap_days"] = roadmap_days

    if st.session_state.get("learning_roadmap_df") is not None:
        roadmap_df = st.session_state["learning_roadmap_df"]
        roadmap_days = st.session_state["learning_roadmap_days"]

        st.markdown(f"### {roadmap_days}일 학습 로드맵")
        st.dataframe(roadmap_df, use_container_width=True)

        roadmap_md = roadmap_df_to_markdown(roadmap_df, roadmap_days)

        with st.expander("로드맵 Markdown 보기", expanded=False):
            st.markdown(roadmap_md)

        st.download_button(
            label="학습 로드맵 다운로드",
            data=roadmap_md,
            file_name=f"learning_roadmap_{roadmap_days}days.md",
            mime="text/markdown"
        )


        st.markdown("### 로드맵 Day별 학습 확인 문제 생성")
        st.caption(
            "선택한 날짜의 학습 범위에 해당하는 문제를 생성합니다. "
            "진도 학습일은 해당 날짜 범위에서, 누적 복습/최종 점검일은 전체 누적 범위에서 생성합니다."
        )

        if "roadmap_day_quizzes" not in st.session_state:
            st.session_state["roadmap_day_quizzes"] = {}

        roadmap_quiz_col1, roadmap_quiz_col2, roadmap_quiz_col3, roadmap_quiz_col4 = st.columns(4)

        with roadmap_quiz_col1:
            selected_roadmap_day = st.selectbox(
                "문제를 만들 Day",
                list(range(1, int(roadmap_days) + 1)),
                index=0,
                key=f"roadmap_quiz_day_{roadmap_days}"
            )

        with roadmap_quiz_col2:
            roadmap_quiz_count = st.selectbox(
                "문제 수",
                [1, 3, 5],
                index=1,
                key=f"roadmap_quiz_count_{roadmap_days}"
            )

        with roadmap_quiz_col3:
            roadmap_question_type_mode = st.selectbox(
                "문제 유형 구성",
                ["혼합", "4지선다 객관식만", "OX 문제만", "빈칸 주관식만"],
                index=0,
                key=f"roadmap_quiz_type_{roadmap_days}"
            )

        with roadmap_quiz_col4:
            roadmap_student_level = st.selectbox(
                "난이도",
                ["Level 1 - 초급", "Level 2 - 중급", "Level 3 - 상급", "Level 4 - 심화"],
                index=1,
                key=f"roadmap_quiz_level_{roadmap_days}"
            )

        selected_day_chunks = get_chunks_for_roadmap_day(
            chunks=chunks,
            duration_days=roadmap_days,
            day=selected_roadmap_day
        )
        selected_day_scope = describe_chunks_scope(selected_day_chunks)

        st.info(f"선택한 Day {selected_roadmap_day} 학습 범위: {selected_day_scope}")

        if st.button("선택한 Day 학습 확인 문제 생성", key=f"generate_roadmap_quiz_{roadmap_days}"):
            if not check_ollama_server(ollama_url):
                st.error("Ollama 서버에 연결할 수 없습니다.")
                st.stop()

            with st.spinner(
                f"로드맵 Day {selected_roadmap_day} 범위에서 학습 확인 문제 {roadmap_quiz_count}개를 생성하는 중입니다..."
            ):
                roadmap_quizzes = generate_roadmap_day_quizzes_with_ollama(
                    ollama_url=ollama_url,
                    model_name=ollama_model,
                    chunks=chunks,
                    duration_days=roadmap_days,
                    day=selected_roadmap_day,
                    quiz_count=roadmap_quiz_count,
                    student_level=roadmap_student_level,
                    question_type_mode=roadmap_question_type_mode,
                    num_predict=num_predict_setting,
                    stream_read_timeout=stream_timeout_setting,
                    banned_terms=banned_terms
                )

            roadmap_quiz_key = f"{roadmap_days}_days_day_{selected_roadmap_day}"
            st.session_state["roadmap_day_quizzes"][roadmap_quiz_key] = {
                "duration_days": roadmap_days,
                "day": selected_roadmap_day,
                "scope": selected_day_scope,
                "quizzes": roadmap_quizzes
            }
            st.success(f"Day {selected_roadmap_day} 학습 확인 문제 {len(roadmap_quizzes)}개 생성 완료")

        roadmap_quiz_key = f"{roadmap_days}_days_day_{selected_roadmap_day}"

        if st.session_state["roadmap_day_quizzes"].get(roadmap_quiz_key):
            saved_item = st.session_state["roadmap_day_quizzes"][roadmap_quiz_key]
            roadmap_quizzes = saved_item.get("quizzes", [])
            saved_scope = saved_item.get("scope", selected_day_scope)

            st.markdown(f"#### Day {selected_roadmap_day} 생성된 학습 확인 문제")
            st.caption(f"학습 범위: {saved_scope}")

            for quiz_idx, quiz in enumerate(roadmap_quizzes, start=1):
                render_quiz_for_review(quiz, quiz_idx)
                st.divider()

            roadmap_quiz_md = roadmap_day_quizzes_to_markdown(
                duration_days=roadmap_days,
                day=selected_roadmap_day,
                scope_text=saved_scope,
                quizzes=roadmap_quizzes
            )

            st.download_button(
                label="로드맵 Day 학습 확인 문제 다운로드",
                data=roadmap_quiz_md,
                file_name=f"roadmap_day_{selected_roadmap_day}_quizzes.md",
                mime="text/markdown",
                key=f"download_roadmap_quiz_{roadmap_days}_{selected_roadmap_day}"
            )

    st.subheader("2단계: ChromaDB 저장")

    if st.button("ChromaDB에 저장하기"):
        embedding_model = load_embedding_model()
        client = get_chroma_client()
        collection = create_or_reset_collection(client, collection_name)

        with st.spinner("청크를 임베딩하고 ChromaDB에 저장하는 중입니다..."):
            saved = store_chunks_in_chroma(collection, chunks, embedding_model)

        st.session_state["collection_ready"] = True
        st.session_state["collection_name"] = collection_name

        st.success(f"{len(st.session_state.get('materials', []))}개 PDF의 {saved}개 청크를 저장했습니다.")

    if not st.session_state.get("collection_ready"):
        st.stop()

    st.divider()
    st.subheader("3단계: 학생 요청 기반 문제 생성")

    user_query = st.text_input(
        "학생 요청",
        placeholder="예: CNN과 RNN의 차이를 구분하는 문제를 만들어줘"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        student_level = st.selectbox(
            "학생 수준",
            ["Level 1 - 초급", "Level 2 - 중급", "Level 3 - 상급", "Level 4 - 심화"],
            index=1
        )

    with col2:
        question_type = st.selectbox(
            "문제 유형",
            ["4지선다 객관식", "빈칸 주관식", "OX 문제"],
            index=0
        )

    with col3:
        question_direction = st.selectbox(
            "문항 방향",
            ["자동(긍정형 우선)", "긍정형(옳은 것은?)", "부정형(옳지 않은 것은?)"],
            index=0
        )

    with col4:
        weak_concept = st.text_input(
            "약점 개념",
            placeholder="예: attention, overfitting, 신탁통치"
        )

    if st.button("RAG 검색 후 문제 생성"):
        if not user_query.strip():
            st.warning("학생 요청을 입력하세요.")
            st.stop()

        if not check_ollama_server(ollama_url):
            st.error("Ollama 서버에 연결할 수 없습니다.")
            st.stop()

        embedding_model = load_embedding_model()
        client = get_chroma_client()
        collection = client.get_collection(name=st.session_state["collection_name"])

        search_query_parts = [user_query]

        if weak_concept.strip():
            search_query_parts.append(weak_concept)

        search_query = " ".join(search_query_parts)

        retrieved = search_chunks(
            collection=collection,
            query=search_query,
            embedding_model=embedding_model,
            top_k=top_k,
            extra_keywords=extra_keywords
        )

        if not retrieved:
            st.error("관련 청크를 찾지 못했습니다. 검색 보강 키워드를 추가하거나 Top-k를 조정해보세요.")
            st.stop()

        st.subheader("검색된 교안 청크")

        result_df = pd.DataFrame([
            {
                "rank": idx + 1,
                "chunk_id": item["chunk_id"],
                "file_name": item.get("file_name", ""),
                "page": item["page"],
                "distance": round(item["distance"], 4),
                "quality_score": item["quality_score"],
                "keyword_score": item["keyword_score"],
                "rerank_score": item["rerank_score"],
                "preview": item["text"][:180].replace("\n", " ")
            }
            for idx, item in enumerate(retrieved)
        ])
        st.dataframe(result_df, use_container_width=True)

        with st.expander("검색된 청크 원문 보기"):
            for idx, item in enumerate(retrieved, start=1):
                st.markdown(f"### Rank {idx} | {item.get('file_name', '')} | Page {item['page']} | {item['chunk_id']}")
                st.write(item["text"])

        allowed_pages = [
            int(item["page"])
            for item in retrieved
            if item.get("page") is not None
        ]

        context_text = "\n\n".join(item["text"] for item in retrieved)
        question_intent = detect_question_intent(user_query)

        prompt = build_quiz_prompt(
            user_query=user_query,
            retrieved_chunks=retrieved,
            student_level=student_level,
            weak_concept=weak_concept,
            question_type=question_type,
            question_direction=question_direction,
            question_intent=question_intent
        )

        with st.expander("Ollama에 전달한 프롬프트 보기"):
            st.text_area("Prompt", value=prompt, height=480)

        with st.spinner("Ollama 로컬 모델이 문제를 생성하는 중입니다..."):
            quiz = generate_quiz_with_ollama(
                prompt=prompt,
                model_name=ollama_model,
                ollama_url=ollama_url,
                allowed_pages=allowed_pages,
                expected_difficulty=get_level_number(student_level),
                context_text=context_text,
                num_predict=num_predict_setting,
                stream_read_timeout=stream_timeout_setting,
                banned_terms=banned_terms
            )

        st.session_state["latest_user_quiz"] = quiz
        st.session_state["latest_user_question_intent"] = question_intent
        st.success("학생 요청 기반 문제 1개 생성 완료")


    if st.session_state.get("latest_user_quiz"):
        st.divider()
        st.subheader("생성된 문제")
        st.caption(
            "학생 요청 기반 문제 생성은 기존처럼 1개만 생성됩니다. "
            "답을 제출하면 채점 후 정답, 해설, 정답 근거가 공개되고 풀이 기록에 저장됩니다."
        )
        render_interactive_quiz(
            quiz=st.session_state["latest_user_quiz"],
            idx=1,
            origin="학생 요청 문제",
            origin_key="latest_user_quiz",
            title="문제"
        )

        st.download_button(
            label="생성된 문제 JSON 다운로드",
            data=json.dumps(st.session_state["latest_user_quiz"], ensure_ascii=False, indent=2),
            file_name="generated_quiz.json",
            mime="application/json"
        )

    st.divider()
    st.subheader("4단계: 학습 상태 분석 및 보충 문제 생성")

    attempts = st.session_state.get("attempts", [])

    if not attempts:
        st.info("문제를 풀고 채점하면 이곳에 취약 개념, 틀린 개념, 흔들리는 개념이 자동으로 정리됩니다.")
    else:
        mastery_df = analyze_concept_mastery(attempts)

        st.markdown("### 개념별 학습 상태")
        st.dataframe(mastery_df, use_container_width=True)

        with st.expander("전체 풀이 기록 보기", expanded=False):
            attempts_df = pd.DataFrame(attempts)
            st.dataframe(attempts_df, use_container_width=True)

        st.download_button(
            label="풀이 기록 JSON 다운로드",
            data=json.dumps(attempts, ensure_ascii=False, indent=2),
            file_name="learning_attempts.json",
            mime="application/json"
        )

        if st.button("풀이 기록 초기화"):
            st.session_state["attempts"] = []
            st.session_state["graded_results"] = {}
            st.session_state["remedial_quizzes"] = {}
            st.rerun()

        target_concepts = get_target_concepts_for_remedial(mastery_df)

        if target_concepts:
            st.markdown("### 부족한 개념 보충 문제 생성")
            st.caption("틀린 개념, 취약 개념, 흔들리는 개념을 바탕으로 관련 문제를 추가 생성합니다.")

            remedial_col1, remedial_col2, remedial_col3, remedial_col4 = st.columns(4)

            with remedial_col1:
                selected_concept = st.selectbox(
                    "보충 학습할 개념",
                    target_concepts,
                    index=0,
                    key="selected_remedial_concept"
                )

            with remedial_col2:
                remedial_count = st.selectbox(
                    "문제 수",
                    [1, 3, 5],
                    index=1,
                    key="remedial_count"
                )

            with remedial_col3:
                remedial_type_mode = st.selectbox(
                    "문제 유형 구성",
                    ["혼합", "4지선다 객관식만", "OX 문제만", "빈칸 주관식만"],
                    index=0,
                    key="remedial_type_mode"
                )

            with remedial_col4:
                remedial_level = st.selectbox(
                    "난이도",
                    ["Level 1 - 초급", "Level 2 - 중급", "Level 3 - 상급", "Level 4 - 심화"],
                    index=1,
                    key="remedial_level"
                )

            if st.button("부족한 개념 보충 문제 생성"):
                if not check_ollama_server(ollama_url):
                    st.error("Ollama 서버에 연결할 수 없습니다.")
                    st.stop()

                embedding_model = load_embedding_model()
                client = get_chroma_client()
                collection = client.get_collection(name=st.session_state["collection_name"])

                with st.spinner(f"'{selected_concept}' 개념 보충 문제 {remedial_count}개를 생성하는 중입니다..."):
                    remedial_quizzes = generate_remedial_quizzes_with_ollama(
                        ollama_url=ollama_url,
                        model_name=ollama_model,
                        collection=collection,
                        embedding_model=embedding_model,
                        target_concept=selected_concept,
                        quiz_count=remedial_count,
                        student_level=remedial_level,
                        question_type_mode=remedial_type_mode,
                        top_k=top_k,
                        extra_keywords=extra_keywords,
                        num_predict=num_predict_setting,
                        stream_read_timeout=stream_timeout_setting,
                        banned_terms=banned_terms
                    )

                remedial_key = make_safe_id(selected_concept)
                st.session_state["remedial_quizzes"][remedial_key] = {
                    "concept": selected_concept,
                    "quizzes": remedial_quizzes
                }
                st.success(f"'{selected_concept}' 보충 문제 {len(remedial_quizzes)}개 생성 완료")

            selected_remedial_key = make_safe_id(st.session_state.get("selected_remedial_concept", target_concepts[0]))

            if st.session_state["remedial_quizzes"].get(selected_remedial_key):
                saved_remedial = st.session_state["remedial_quizzes"][selected_remedial_key]
                remedial_quizzes = saved_remedial.get("quizzes", [])
                concept_name = saved_remedial.get("concept", "")

                st.markdown(f"#### '{concept_name}' 보충 문제")

                for quiz_idx, quiz in enumerate(remedial_quizzes, start=1):
                    render_quiz_for_review(quiz, quiz_idx)
                    st.divider()

                remedial_md = review_quizzes_to_markdown(f"{concept_name}_보충", remedial_quizzes)

                st.download_button(
                    label="보충 문제 다운로드",
                    data=remedial_md,
                    file_name=f"remedial_quizzes_{make_safe_id(concept_name)}.md",
                    mime="text/markdown",
                    key=f"download_remedial_{selected_remedial_key}"
                )

except Exception as e:
    st.error("처리 중 오류가 발생했습니다.")
    st.exception(e)

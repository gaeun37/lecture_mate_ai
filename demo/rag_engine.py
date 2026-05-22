
import json
import re
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple

import chromadb
import pandas as pd
import pymupdf
import requests
from sentence_transformers import SentenceTransformer
from json_repair import repair_json


# =========================
# 기본 설정
# =========================


# =========================
# 1. 모델 / DB
# =========================

@lru_cache(maxsize=1)
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


@lru_cache(maxsize=1)
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
# FastAPI용 보조 함수
# =========================

def generate_material_summary_with_ollama(
    chunks: List[Dict[str, Any]],
    ollama_url: str,
    model_name: str,
    user_id: int = 1,
    num_predict: int = 900,
    stream_read_timeout: int = 120
) -> Dict[str, Any]:
    """
    FastAPI 자료 분석 화면용 요약.
    기존 Streamlit의 generate_document_summary_with_ollama() 로직을 그대로 사용하되,
    HTML 화면의 summary/keywords/concepts 구조로 변환한다.
    """
    summary_md = generate_document_summary_with_ollama(
        ollama_url=ollama_url,
        model_name=model_name,
        chunks=chunks,
        num_predict=num_predict,
        stream_read_timeout=stream_read_timeout
    )

    return material_summary_markdown_to_cards(summary_md, chunks, user_id=user_id)


def material_summary_markdown_to_cards(
    summary_md: str,
    chunks: List[Dict[str, Any]],
    user_id: int = 1
) -> Dict[str, Any]:
    """
    Streamlit 요약 Markdown을 HTML UI 카드 구조로 변환한다.
    """
    text = str(summary_md or "").strip()

    if not text or text.startswith("전체 요약 생성 실패"):
        return fallback_summary_from_chunks(chunks, user_id=user_id)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    one_line = ""
    full_items = []
    concept_items = []
    current = None

    for line in lines:
        if "한 줄 요약" in line:
            current = "one"
            continue
        if "전체 핵심 요약" in line:
            current = "summary"
            continue
        if "핵심 개념" in line:
            current = "concepts"
            continue
        if "시험 대비" in line:
            current = "exam"
            continue

        cleaned = line.lstrip("- ").strip()
        if not cleaned:
            continue

        if current == "one" and not one_line:
            one_line = cleaned
        elif current == "summary":
            full_items.append(cleaned)
        elif current == "concepts":
            concept_items.append(cleaned)

    if not one_line and full_items:
        one_line = full_items[0]

    summary = " ".join(full_items[:5]).strip() or one_line or text[:500]

    keywords = []
    for item in concept_items:
        name = re.split(r"[:：\-–]", item, maxsplit=1)[0].strip()
        if name and len(name) <= 30 and name not in keywords:
            keywords.append(name)
    if not keywords:
        preview_text = " ".join(str(c.get("text", ""))[:200] for c in chunks[:6])
        keywords = tokenize_query(preview_text)[:5]

    concepts = []
    pages = [int(c.get("page", 1)) for c in chunks if c.get("page") is not None]
    default_page = min(pages) if pages else 1

    for idx, item in enumerate(concept_items[:5], start=1):
        if ":" in item:
            name, desc = item.split(":", 1)
        elif "：" in item:
            name, desc = item.split("：", 1)
        elif "-" in item:
            name, desc = item.split("-", 1)
        else:
            name, desc = item[:25], item

        concepts.append({
            "name": name.strip() or f"핵심 개념 {idx}",
            "summary": desc.strip() or item.strip(),
            "source_page": default_page
        })

    if not concepts:
        fallback = fallback_summary_from_chunks(chunks, user_id=user_id)
        concepts = fallback.get("concepts", [])

    return {
        "user_id": user_id,
        "summary": summary,
        "keywords": keywords[:8],
        "concepts": concepts[:6],
        "raw_markdown": text
    }


def fallback_summary_from_chunks(
    chunks: List[Dict[str, Any]],
    user_id: int = 1
) -> Dict[str, Any]:
    usable = [
        c for c in chunks
        if str(c.get("text", "")).strip()
        and not is_table_of_contents_like(str(c.get("text", "")))
    ]

    if not usable:
        return {
            "user_id": user_id,
            "summary": "요약할 수 있는 본문 텍스트가 충분하지 않습니다.",
            "keywords": [],
            "concepts": []
        }

    preview_text = " ".join(str(c.get("text", ""))[:250] for c in usable[:5])
    keywords = tokenize_query(preview_text)[:8]

    concepts = []
    for c in usable[:4]:
        text = str(c.get("text", "")).replace("\n", " ").strip()
        concepts.append({
            "name": c.get("file_name", "업로드 자료"),
            "summary": text[:140] + ("..." if len(text) > 140 else ""),
            "source_page": c.get("page", 1)
        })

    return {
        "user_id": user_id,
        "summary": "업로드된 자료의 주요 본문을 바탕으로 핵심 내용을 확인할 수 있습니다. 아래 키워드와 개념 카드는 추출된 본문을 기준으로 정리한 내용입니다.",
        "keywords": keywords,
        "concepts": concepts
    }


def build_tutor_answer_prompt(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    recent_history: Optional[List[Dict[str, str]]] = None
) -> str:
    context = "\n\n".join(
        f"[근거 {idx + 1}]\n"
        f"파일: {c.get('file_name', '')}\n"
        f"페이지: {c.get('page')}\n"
        f"내용: {c.get('text', '')[:700]}"
        for idx, c in enumerate(retrieved_chunks[:6])
    )

    history_text = "\n".join(
        f"{h.get('role')}: {h.get('content')}"
        for h in (recent_history or [])[-4:]
    )

    return f"""
너는 한국사 강의자료 기반 AI 튜터이다.

아래 규칙을 반드시 지켜라.
- 반드시 한국어로만 답하라.
- JSON, 표, 코드블록을 출력하지 마라.
- step, action, input, output 같은 분석 형식을 출력하지 마라.
- 생각 과정, 추론 과정, 자기 점검을 출력하지 마라.
- "Okay", "Wait", "Hmm", "Let's" 같은 영어 표현을 쓰지 마라.
- 학생에게 보여줄 최종 답변만 작성하라.
- 자료에 없는 내용은 확정적으로 말하지 말고 "자료에서 확인되는 범위에서는"이라고 표현하라.
- 시대 흐름 질문은 시대 순서대로 4~7문장으로 설명하라.
- 마지막에 후속 질문을 유도하는 한 문장을 붙여라.

[이전 대화]
{history_text if history_text else "없음"}

[학생 질문]
{question}

[강의자료 근거]
{context}

최종 답변:
""".strip()


def build_tutor_extra_keywords(question: str) -> List[str]:
    q = str(question)
    keywords = []

    if any(w in q for w in ["흐름", "시대", "순서", "정리", "전체"]) or ("부터" in q and "까지" in q):
        keywords.extend([
            "고조선", "부여", "고구려", "백제", "신라", "가야",
            "삼국", "통일신라", "발해", "후삼국", "후백제",
            "후고구려", "태봉", "궁예", "견훤", "왕건", "고려"
        ])

    for term in ["고조선", "삼국", "통일신라", "발해", "후삼국", "고려", "왕건", "견훤", "궁예", "태봉"]:
        if term in q and term not in keywords:
            keywords.append(term)

    return keywords


def clean_tutor_answer_text(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    # JSON 분석 출력 방어
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(text[start:end])
            if isinstance(data, dict):
                for key in ["answer", "final_answer", "response"]:
                    if data.get(key):
                        return str(data[key]).strip()
                if {"step", "action", "input", "output"}.intersection(data.keys()):
                    return str(data.get("output", "")).strip() or "자료에서 확인되는 범위에서는 답변을 다시 생성해야 합니다."
    except Exception:
        pass

    markers = ["최종 답변:", "최종답변:", "답변:", "정리하면,"]
    for marker in markers:
        if marker in text:
            text = text.split(marker, 1)[-1].strip()
            break

    banned_prefixes = ("Okay", "Wait", "Hmm", "Let's", '"step"', '"action"', "{", "}")
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(banned_prefixes):
            continue
        lines.append(stripped)

    return "\n".join(lines).strip() or "자료에서 확인되는 범위에서는 답변을 생성하지 못했습니다."

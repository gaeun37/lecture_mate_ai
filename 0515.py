
import json
import re
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

def extract_text_from_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    pages = []

    with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            pages.append({
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

        if not text:
            continue

        for idx, chunk_text in enumerate(split_text(text, chunk_size, chunk_overlap), start=1):
            chunks.append({
                "chunk_id": f"p{page_num}_c{idx}",
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

    # 핵심: 너무 길게 넣지 않는다. 과한 프롬프트가 timeout과 환각을 만든다.
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

    return f"""
너는 대학 강의자료 기반 학습 문제를 만드는 AI 튜터이다.

반드시 [교안 근거]에 있는 내용만 사용해서 4지선다 객관식 문제 1개를 만들어라.
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
- 보기 4개는 서로 다른 내용이어야 한다.
- 정답은 보기 4개 중 하나와 글자까지 완전히 같아야 한다.
- 정답 보기는 하나의 핵심 사실만 담는다.
- 한 보기 안에 서로 다른 개념, 조건, 역할, 수식, 단계, 사건을 무리하게 합치지 마라.
- part_summary는 1~2문장으로 작성한다.
- evidence_text는 정답 판단에 필요한 직접 근거만 1~2문장으로 작성한다.
- explanation은 1~2문장으로 작성한다.
- choice_explanations에는 보기 4개 각각의 해설을 1문장씩 작성한다.
- source_pages에는 [사용 가능한 근거 페이지] 안에 있는 숫자만 넣는다.
- difficulty는 반드시 {level_num}으로 작성한다.
- 출력은 반드시 JSON 객체 하나만 반환한다.

[출력 형식]
{{
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
  "hint": "짧은 힌트 한 문장"
}}
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
        "hint": str(quiz.get("hint", "")).strip()
    }

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
    # 여기서 너무 빡세게 막지 않는다. 핵심 구조만 검사한다.
    if not quiz.get("question"):
        return False, "문제가 비어 있습니다."

    choices = quiz.get("choices", [])
    answer = quiz.get("answer", "")

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

    return True, "검증 통과"


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


uploaded_pdf = st.file_uploader(
    "PDF 강의자료를 업로드하세요.",
    type=["pdf"]
)

if uploaded_pdf is None:
    st.info("먼저 PDF 강의자료를 업로드하세요.")
    st.stop()

try:
    file_bytes = uploaded_pdf.read()

    st.subheader("1단계: PDF 텍스트 추출 및 청킹")

    if st.button("PDF 처리하기"):
        with st.spinner("PDF에서 텍스트를 추출하는 중입니다..."):
            pages = extract_text_from_pdf(file_bytes)

        with st.spinner("텍스트를 청킹하는 중입니다..."):
            chunks = make_chunks_from_pages(
                pages,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        st.session_state["pages"] = pages
        st.session_state["chunks"] = chunks

        st.success("PDF 처리 완료")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("전체 페이지 수", len(pages))

        with col2:
            st.metric("전체 글자 수", sum(p["char_count"] for p in pages))

        with col3:
            st.metric("생성된 청크 수", len(chunks))

    if "chunks" not in st.session_state:
        st.stop()

    pages = st.session_state["pages"]
    chunks = st.session_state["chunks"]

    with st.expander("페이지별 텍스트 요약 보기", expanded=False):
        page_df = pd.DataFrame([
            {
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
                "page": c["page"],
                "char_count": c["char_count"],
                "quality_score": c["quality_score"],
                "preview": c["text"][:180].replace("\n", " ")
            }
            for c in chunks[:80]
        ])
        st.dataframe(chunk_df, use_container_width=True)

    st.subheader("2단계: ChromaDB 저장")

    if st.button("ChromaDB에 저장하기"):
        embedding_model = load_embedding_model()
        client = get_chroma_client()
        collection = create_or_reset_collection(client, collection_name)

        with st.spinner("청크를 임베딩하고 ChromaDB에 저장하는 중입니다..."):
            saved = store_chunks_in_chroma(collection, chunks, embedding_model)

        st.session_state["collection_ready"] = True
        st.session_state["collection_name"] = collection_name

        st.success(f"{saved}개 청크를 저장했습니다.")

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
            ["4지선다 객관식"],
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
                st.markdown(f"### Rank {idx} | Page {item['page']} | {item['chunk_id']}")
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

        st.divider()
        st.subheader("생성된 문제")

        if quiz.get("concept") == "generation_error":
            st.error("문제 생성에 실패했습니다. 아래 내용을 확인하세요.")

        polarity_label = "부정형 문제" if quiz.get("question_polarity") == "negative" else "긍정형 문제"
        st.caption(f"문항 방향: {polarity_label} / 질문 의도: {question_intent}")

        if quiz.get("warnings"):
            for warning in quiz["warnings"]:
                st.warning(warning)

        st.markdown(f"### Q. {quiz.get('question', '')}")

        choices = quiz.get("choices", [])

        if choices:
            for idx, choice in enumerate(choices, start=1):
                st.write(f"{idx}. {choice}")

        st.markdown("### 정답")
        st.success(quiz.get("answer", ""))

        if quiz.get("part_summary"):
            st.markdown("### 출제 파트 요약")
            st.write(quiz.get("part_summary", ""))

        if quiz.get("evidence_text"):
            st.markdown("### 정답 근거")
            st.info(quiz.get("evidence_text", ""))

        st.markdown("### 해설")
        st.write(quiz.get("explanation", ""))

        if quiz.get("choice_explanations"):
            st.markdown("### 보기별 해설")

            for idx, item in enumerate(quiz["choice_explanations"], start=1):
                answer_tag = "정답" if item.get("is_answer") else "정답 아님"
                factual_tag = "교안 내용과 일치" if item.get("is_factually_correct") else "교안 내용과 불일치"
                st.markdown(
                    f"**{idx}. {answer_tag} / {factual_tag}**  \n"
                    f"{item.get('explanation', '')}"
                )

        st.markdown("### 힌트")
        st.info(quiz.get("hint", ""))

        st.markdown("### 근거 페이지")
        source_pages = quiz.get("source_pages", [])
        if source_pages:
            st.write(", ".join(str(page) for page in source_pages))
        else:
            st.write([])

        st.markdown("### 핵심 개념 / 난이도")
        st.write({
            "concept": quiz.get("concept", ""),
            "difficulty": quiz.get("difficulty", "")
        })

        st.download_button(
            label="생성된 문제 JSON 다운로드",
            data=json.dumps(quiz, ensure_ascii=False, indent=2),
            file_name="generated_quiz.json",
            mime="application/json"
        )

except Exception as e:
    st.error("처리 중 오류가 발생했습니다.")
    st.exception(e)

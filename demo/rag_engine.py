
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
# 출력 언어 정리
# =========================

HANJA_PHRASE_MAP = {
    "崗上墓": "강상묘",
    "樓上墓": "누상묘",
    "鐵鐸": "철탁",
    "鐵刀子": "철도자",
    "浿水": "패수",
    "上下障": "상하장",
    "準王": "준왕",
    "外臣": "외신",
    "兵威財物": "병위재물",
    "東夷": "동이",
    "遼陽": "요양",
    "遼陽": "요양",
    "二道河子": "이도하자",
    "瀋陽": "심양",
    "鄭家窪子": "정가와자",
    "요령식동검": "요령식 동검",
}

# 외부 패키지 없이 최소한의 한자 독음 변환을 수행한다.
# 모르는 한자는 최종적으로 제거해서 학생 화면에 한자/중국어 문자가 노출되지 않게 한다.
HANJA_CHAR_MAP = {
    "一":"일","二":"이","三":"삼","四":"사","五":"오","六":"육","七":"칠","八":"팔","九":"구","十":"십","百":"백","千":"천","萬":"만",
    "上":"상","下":"하","中":"중","東":"동","西":"서","南":"남","北":"북","前":"전","後":"후","內":"내","外":"외","大":"대","小":"소",
    "王":"왕","臣":"신","民":"민","國":"국","君":"군","皇":"황","帝":"제","官":"관","兵":"병","軍":"군","城":"성","京":"경","都":"도",
    "威":"위","財":"재","物":"물","水":"수","江":"강","河":"하","山":"산","海":"해","島":"도","里":"리","洞":"동","郡":"군","縣":"현","州":"주",
    "鐵":"철","銅":"동","金":"금","銀":"은","石":"석","骨":"골","木":"목","土":"토","器":"기","劍":"검","刀":"도","鏃":"촉","鐸":"탁",
    "墓":"묘","墳":"분","棺":"관","葬":"장","坑":"갱","穴":"혈","窟":"굴","窪":"와","家":"가","子":"자","陽":"양","瀋":"심","鄭":"정","崗":"강","岡":"강","樓":"누","樓":"누",
    "時":"시","代":"대","年":"년","月":"월","日":"일","世":"세","紀":"기","初":"초","末":"말","期":"기","古":"고","新":"신","舊":"구",
    "文":"문","化":"화","史":"사","學":"학","敎":"교","佛":"불","寺":"사","塔":"탑","遺":"유","蹟":"적","跡":"적","址":"지","人":"인","類":"류","社":"사","會":"회","經":"경","濟":"제",
    "高":"고","句":"구","麗":"려","百":"백","濟":"제","新":"신","羅":"라","伽":"가","倭":"왜","漢":"한","唐":"당","宋":"송","元":"원","明":"명","淸":"청","秦":"진","燕":"연","魏":"위","晉":"진","隋":"수","遼":"요","遼":"요",
    "準":"준","浿":"패","夷":"이","靑":"청","龍":"용","白":"백","虎":"호","朱":"주","雀":"작","玄":"현","武":"무",
}

HANJA_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
KANA_RE = re.compile(r"[\u3040-\u30FF]")


def contains_hanja(text: Any) -> bool:
    return HANJA_RE.search(str(text or "")) is not None


def sanitize_user_text(text: Any) -> str:
    """
    학생 화면에는 한국어/영어 중심으로만 보이도록 정리한다.
    원문 PDF에 한자가 있어도 가능한 경우 한글 독음으로 바꾸고, 모르는 한자는 제거한다.
    """
    value = str(text or "")

    for src, dst in HANJA_PHRASE_MAP.items():
        value = value.replace(src, dst)

    value = "".join(HANJA_CHAR_MAP.get(ch, ch) for ch in value)
    value = HANJA_RE.sub("", value)
    value = KANA_RE.sub("", value)

    # 한자 제거 후 남는 어색한 괄호와 공백 정리
    value = re.sub(r"\(\s*\)", "", value)
    value = re.sub(r"\[\s*\]", "", value)
    value = re.sub(r"\s+", " ", value)
    value = value.replace(" .", ".").replace(" ,", ",").replace(" :", ":")
    return value.strip()


def sanitize_student_output(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_user_text(value)
    if isinstance(value, list):
        return [sanitize_student_output(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_student_output(item) for key, item in value.items()}
    return value


def sanitize_quiz_language(quiz: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = sanitize_student_output(quiz)
    # source_pages, difficulty처럼 숫자여야 하는 값은 그대로 보존한다.
    if isinstance(cleaned, dict):
        if "source_pages" in quiz:
            cleaned["source_pages"] = quiz.get("source_pages", [])
        if "difficulty" in quiz:
            cleaned["difficulty"] = quiz.get("difficulty", cleaned.get("difficulty"))
    return cleaned


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
            f"내용:\n{sanitize_user_text(chunk['text'][:per_chunk_limit])}\n"
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
단, 한자/중국어/일본어 문자는 절대 출력하지 마라.
원문 근거에 한자가 있으면 한글 독음이나 한국어 풀이로 바꾸어 작성하라. 예: 崗上墓는 강상묘, 樓上墓는 누상묘처럼 쓴다.
문제, 보기, 정답, 힌트, 해설, 정답 근거는 한국어를 기본으로 작성하고, 영어는 원문 용어 보충이 필요할 때 괄호 안에서만 사용하라.

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
- 학생 요청의 핵심 주제와 직접 관련된 근거가 부족하면, 넓게 추측하지 말고 검색된 근거에서 직접 확인되는 사실만 묻는다.
- 특정 지역/시기/유적/인물의 '기원'을 묻는 문제는 근거에 기원 또는 가장 오래된 지역이 직접 드러날 때만 만든다.
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
    one_line = sanitize_user_text(summary_json.get("one_line_summary", ""))
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
        item = sanitize_user_text(item)
        if item:
            md += f"- {item}\n"

    md += "\n## 3. 핵심 개념\n"
    for item in key_concepts:
        item = sanitize_user_text(item)
        if item:
            md += f"- {item}\n"

    md += "\n## 4. 시험 대비 포인트\n"
    for item in exam_points:
        item = sanitize_user_text(item)
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

    choices = [sanitize_user_text(c) for c in choices if str(c).strip()]
    answer = sanitize_user_text(quiz.get("answer", ""))

    clean = {
        "question_type": sanitize_user_text(quiz.get("question_type", "multiple_choice")),
        "question_polarity": infer_polarity(
            sanitize_user_text(quiz.get("question", "")),
            str(quiz.get("question_polarity", "positive"))
        ),
        "question": sanitize_user_text(quiz.get("question", "")),
        "choices": choices,
        "answer": answer,
        "part_summary": sanitize_user_text(quiz.get("part_summary", "")),
        "evidence_text": sanitize_user_text(quiz.get("evidence_text", "")),
        "explanation": sanitize_user_text(quiz.get("explanation", "")),
        "choice_explanations": [],
        "source_pages": filter_source_pages(quiz.get("source_pages", []), allowed_pages),
        "concept": sanitize_user_text(quiz.get("concept", "")),
        "difficulty": int(expected_difficulty),
        "hint": sanitize_user_text(quiz.get("hint", "")),
        "grading_criteria": sanitize_student_output(quiz.get("grading_criteria", []))
    }

    # 빈칸 주관식 보정
    # 모델이 ____, ___처럼 다른 길이의 밑줄을 출력해도 표준 _____로 통일한다.
    if clean["question_type"] == "fill_blank":
        clean["question"] = normalize_blank_marker(clean["question"])
        clean["choices"] = []
        clean["choice_explanations"] = []
        return sanitize_quiz_language(clean)

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
        return sanitize_quiz_language(clean)

    raw_explanations = quiz.get("choice_explanations", [])

    if not isinstance(raw_explanations, list):
        raw_explanations = []

    explanation_map = {}

    for item in raw_explanations:
        if isinstance(item, dict) and item.get("choice"):
            explanation_map[normalize_for_match(sanitize_user_text(item.get("choice")))] = sanitize_student_output(item)

    aligned = []

    for choice in choices:
        item = explanation_map.get(normalize_for_match(choice), {})
        aligned.append({
            "choice": choice,
            "is_answer": choice == answer,
            "is_factually_correct": bool(item.get("is_factually_correct", choice == answer)),
            "explanation": sanitize_user_text(item.get("explanation", ""))
        })

    clean["choice_explanations"] = aligned

    return sanitize_quiz_language(clean)


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
    context = normalize_for_match(sanitize_user_text(context_text))

    for term in banned_terms:
        nt = normalize_for_match(term)

        if nt and nt in generated and nt not in context:
            warnings.append(f"자료별 금지 표현이 생성 결과에 포함됨: {sanitize_user_text(term)}")

    # 정답이 아주 짧은 경우에는 근거 안에 직접 등장하는지 확인
    answer = sanitize_user_text(quiz.get("answer", ""))

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
        "evidence_text": sanitize_user_text(raw_output[:1500]) if raw_output else "",
        "explanation": sanitize_user_text(message),
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
        f"내용: {sanitize_user_text(c.get('text', '')[:700])}"
        for idx, c in enumerate(retrieved_chunks[:6])
    )

    history_text = "\n".join(
        f"{h.get('role')}: {h.get('content')}"
        for h in (recent_history or [])[-4:]
    )

    return f"""
너는 한국사 강의자료 기반 AI 튜터이다.

아래 규칙을 반드시 지켜라.
- 출력은 반드시 JSON 객체 하나만 작성한다.
- JSON의 key는 answer 하나만 사용한다.
- answer 값에는 학생에게 보여줄 최종 한국어 답변만 넣는다.
- 생각 과정, 추론 과정, 자기 점검, 근거별 분석 과정을 절대 쓰지 마라.
- "First", "I need", "Let me", "Starting with", "Looking at", "Okay", "Wait", "Hmm", "Let's" 같은 영어 사고 표현을 절대 쓰지 마라.
- 한자/중국어/일본어 문자는 절대 출력하지 마라.
- 원문 근거에 한자가 있으면 한글 독음이나 한국어 풀이로 바꾸어 작성하라. 예: 鐵鐸는 철탁, 崗上墓는 강상묘, 樓上墓는 누상묘처럼 쓴다.
- 한국어를 기본으로 답하고, 영어는 원어 병기가 꼭 필요한 용어에만 괄호 안에서 짧게 사용한다.
- 자료에 없는 내용은 확정적으로 말하지 말고 "자료에서 확인되는 범위에서는"이라고 표현하라.
- 시대 흐름 질문은 시대 순서대로 4~7문장으로 설명하라.
- 마지막에 후속 질문을 유도하는 한 문장을 붙여라.

[이전 대화]
{history_text if history_text else "없음"}

[학생 질문]
{question}

[강의자료 근거]
{context}

[출력 형식]
{{"answer":"학생에게 보여줄 최종 한국어 답변"}}
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

    # JSON 출력이면 answer 계열 필드만 꺼낸다.
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(text[start:end])
            if isinstance(data, dict):
                for key in ["answer", "final_answer", "response"]:
                    if data.get(key):
                        return sanitize_user_text(data[key])
                if {"step", "action", "input", "output"}.intersection(data.keys()):
                    return sanitize_user_text(data.get("output", "")) or "자료에서 확인되는 범위에서는 답변을 다시 생성해야 합니다."
    except Exception:
        pass

    # 모델이 사고 과정을 앞에 붙인 경우, 최종 답변 표식 뒤만 남긴다.
    markers = ["최종 답변:", "최종답변:", "답변:", "정리하면,", "answer:"]
    for marker in markers:
        if marker in text:
            text = text.split(marker, 1)[-1].strip()
            break

    banned_prefixes = (
        "Okay", "Wait", "Hmm", "Let's", "First", "I need", "Let me", "Starting with",
        "Looking at", "The user", "The question", "The text", "This talks", "This discusses",
        "From 근거", "근거 1", "근거 2", "근거 3", '"step"', '"action"', "{", "}"
    )
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(banned_prefixes):
            continue

        korean_count = len(re.findall(r"[가-힣]", stripped))
        alpha_count = len(re.findall(r"[A-Za-z]", stripped))
        # 영어 사고 과정으로 보이는 줄은 제거한다. 단, 한국어가 충분히 섞인 학습 답변은 유지한다.
        if alpha_count > max(20, korean_count * 2) and korean_count < 8:
            continue
        lines.append(stripped)

    cleaned = sanitize_user_text("\n".join(lines))

    # 그래도 사고 과정 흔적이 남으면 안전한 안내문으로 바꾼다.
    thought_markers = ["First, I need", "Starting with", "Looking at", "Let me", "I need to"]
    if any(marker in cleaned for marker in thought_markers):
        return "자료에서 확인되는 범위에서는 답변을 다시 생성해야 합니다. 질문을 조금 더 구체적으로 입력해 주세요."

    return cleaned or "자료에서 확인되는 범위에서는 답변을 생성하지 못했습니다."


# =========================
# 7-0. 학생 풀이 채점 / 학습 상태 분석 / 보충 문제 생성
# Streamlit 원본의 session_state 의존 부분을 FastAPI에서도 쓸 수 있도록 순수 함수로 분리
# =========================

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
    question_type = quiz.get("question_type") or quiz.get("type") or "multiple_choice"
    correct_answer = str(quiz.get("answer") or quiz.get("correct_answer") or "").strip()

    if question_type == "ox":
        return normalize_ox_answer(student_answer) == normalize_ox_answer(correct_answer)

    if question_type == "multiple_choice":
        return normalize_answer_for_grading(student_answer) == normalize_answer_for_grading(correct_answer)

    if question_type == "fill_blank":
        student_norm = normalize_answer_for_grading(student_answer)
        answer_norm = normalize_answer_for_grading(correct_answer)

        if not student_norm or not answer_norm:
            return False

        if student_norm == answer_norm:
            return True

        if len(answer_norm) >= 2 and answer_norm in student_norm:
            return True

        return False

    return normalize_answer_for_grading(student_answer) == normalize_answer_for_grading(correct_answer)


def build_attempt_record(
    quiz: Dict[str, Any],
    student_answer: Any,
    is_correct: bool,
    confidence: int = 3,
    used_hint: bool = False,
    origin: str = "학습 문제",
    origin_key: str = ""
) -> Dict[str, Any]:
    concept = str(quiz.get("concept", "")).strip() or "기타 개념"

    return {
        "attempt_id": f"attempt_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "origin": origin,
        "origin_key": origin_key,
        "file_name": quiz.get("file_name", ""),
        "material_id": quiz.get("material_id", ""),
        "roadmap_day": quiz.get("roadmap_day", ""),
        "question_type": quiz.get("question_type") or quiz.get("type", "multiple_choice"),
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
        for i in range(quiz_count):
            selected.append(good_chunks[i % len(good_chunks)])

    return selected


def get_review_question_type(index: int, mode: str) -> str:
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
    학생 요청 기반 문제 생성과 별개로 동작한다.
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

        prompt = build_quiz_prompt(
            user_query=user_query,
            retrieved_chunks=retrieved_chunks,
            student_level=student_level,
            weak_concept="",
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
        source_file = str(quiz.get("file_name") or quiz.get("source_file") or quiz.get("material_name") or "").strip()
        page_text = ", ".join(str(p) for p in quiz.get("source_pages", []))
        if source_file:
            md += f"**근거 문서:** {source_file} p.{page_text}\n\n"
        else:
            md += "**근거 페이지:** " + page_text + "\n\n"

    if quiz.get("concept"):
        md += f"**핵심 개념:** {quiz.get('concept', '')}\n\n"

    return md.strip()


def review_quizzes_to_markdown(file_name: str, quizzes: List[Dict[str, Any]]) -> str:
    md = f"# {file_name} 학습 확인 문제\n\n"

    for idx, quiz in enumerate(quizzes, start=1):
        md += quiz_to_markdown(quiz, idx)
        md += "\n\n---\n\n"

    return md.strip()


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
    학생 요청 기반 문제 생성은 1개만 생성되고, 이 함수와 분리되어 동작한다.
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

        prompt = build_quiz_prompt(
            user_query=user_query,
            retrieved_chunks=retrieved_chunks,
            student_level=student_level,
            weak_concept="",
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

        quiz["roadmap_day"] = int(day)
        quiz["roadmap_duration_days"] = int(duration_days)
        quiz["roadmap_scope"] = scope_text
        quiz["roadmap_quiz_index"] = idx + 1
        quiz["roadmap_quiz_total"] = quiz_count
        quiz["review_question_type_mode"] = question_type_mode
        quiz["file_name"] = get_chunk_file_name(chunk)
        quiz["material_id"] = str(chunk.get("material_id", ""))

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




def select_relevant_chunks_for_concept(
    retrieved_chunks: List[Dict[str, Any]],
    target_concept: str,
    quiz_count: int
) -> List[Dict[str, Any]]:
    """
    보충 문제는 '파일 전체 대표 청크'보다 '틀린 개념과 가장 직접 관련된 청크'를 우선 사용한다.
    관련성이 낮은 청크가 섞이면 철기 기원 문제에서 전혀 다른 유적/시대 문제가 생성될 수 있기 때문이다.
    """
    if not retrieved_chunks:
        return []

    quiz_count = int(quiz_count)
    concept_tokens = tokenize_query(target_concept)
    scored = []

    for chunk in retrieved_chunks:
        text = str(chunk.get("text", ""))
        score = keyword_score(target_concept, text, concept_tokens)
        score += int(chunk.get("keyword_score", 0))

        # 정확한 개념 문구가 붙어서 등장하는 경우 가산점
        if normalize_for_match(target_concept) and normalize_for_match(target_concept) in normalize_for_match(text):
            score += 8

        scored.append((score, float(chunk.get("rerank_score", 0) or 0), chunk))

    scored.sort(key=lambda x: (-x[0], -x[1]))
    max_score = scored[0][0]

    # 관련 점수가 너무 낮은 후보는 제외한다. 후보가 하나뿐이면 그 청크를 반복 사용한다.
    if max_score > 0:
        keep = [chunk for score, _, chunk in scored if score >= max(1, int(max_score * 0.6))]
    else:
        keep = [scored[0][2]]

    if not keep:
        keep = [scored[0][2]]

    selected = []
    for i in range(quiz_count):
        selected.append(keep[i % len(keep)])

    return selected

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
        top_k=max(int(top_k), quiz_count * 3, 8),
        extra_keywords=boosted_keywords
    )

    if not retrieved:
        return [make_error_quiz("보충 문제를 만들 관련 청크를 찾지 못했습니다.", "", [])]

    selected_chunks = select_relevant_chunks_for_concept(retrieved, target_concept, quiz_count)
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

        quiz["target_concept"] = sanitize_user_text(target_concept)
        quiz["remedial_quiz_index"] = idx + 1
        quiz["remedial_quiz_total"] = quiz_count
        quiz["review_question_type_mode"] = question_type_mode
        quiz["file_name"] = get_chunk_file_name(chunk)
        quiz["material_id"] = str(chunk.get("material_id", ""))
        quizzes.append(quiz)

    return quizzes


# =========================
# 7-3. Streamlit 화면 함수의 FastAPI 대응 래퍼
# 원본 함수명을 보존하되, Streamlit st 호출 대신 API/프론트엔드에서 쓰기 쉬운 구조화 데이터 반환
# =========================

def init_learning_state(state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    원본 Streamlit의 st.session_state 초기화 함수와 같은 역할.
    FastAPI demo에서는 dict 상태 객체를 받아 필요한 키를 채워 반환한다.
    """
    state = state if state is not None else {}
    state.setdefault("attempts", [])
    state.setdefault("graded_results", {})
    state.setdefault("remedial_quizzes", {})
    return state


def save_attempt_record(attempt: Dict[str, Any], attempts: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    원본 Streamlit 함수명 보존.
    FastAPI demo에서는 app_state['attempts']에 직접 append하지만,
    테스트와 재사용을 위해 리스트를 받아 저장한 뒤 반환한다.
    """
    attempts = attempts if attempts is not None else []
    attempts.append(attempt)
    return attempts


def render_quiz_feedback(quiz: Dict[str, Any], is_correct: bool) -> Dict[str, Any]:
    """
    원본 Streamlit 화면 출력 함수의 데이터 버전.
    프론트엔드는 이 반환값과 submit_answer 응답을 이용해 정답/해설/근거를 렌더링한다.
    """
    return {
        "is_correct": bool(is_correct),
        "answer": quiz.get("answer", ""),
        "explanation": quiz.get("explanation", ""),
        "evidence_text": quiz.get("evidence_text", ""),
        "source_pages": quiz.get("source_pages", []),
        "concept": quiz.get("concept", ""),
        "difficulty": quiz.get("difficulty", ""),
        "choice_explanations": quiz.get("choice_explanations", []),
        "grading_criteria": quiz.get("grading_criteria", [])
    }


def render_interactive_quiz(
    quiz: Dict[str, Any],
    idx: int,
    origin: str,
    origin_key: str,
    title: str = "문제"
) -> Dict[str, Any]:
    """
    원본 Streamlit 상호작용 함수의 데이터 버전.
    실제 답 입력/채점 UI는 frontend/index.html과 /api/answers/submit에서 처리한다.
    """
    return {
        "idx": idx,
        "origin": origin,
        "origin_key": origin_key,
        "title": title,
        "question_type": quiz.get("question_type", "multiple_choice"),
        "question": quiz.get("question", ""),
        "choices": quiz.get("choices", []),
        "hint": quiz.get("hint", ""),
        "concept": quiz.get("concept", ""),
        "source_pages": quiz.get("source_pages", [])
    }


def render_quiz_for_review(quiz: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    파일별/로드맵/보충 문제 렌더링 함수의 데이터 버전.
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

    return render_interactive_quiz(
        quiz=quiz,
        idx=idx,
        origin=origin,
        origin_key=origin_key,
        title="문제"
    )

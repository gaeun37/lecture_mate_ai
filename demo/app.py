from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_engine import (
    load_embedding_model,
    get_chroma_client,
    make_safe_id,
    extract_text_from_pdf,
    make_chunks_from_pages,
    create_or_reset_collection,
    store_chunks_in_chroma,
    search_chunks,
    detect_question_intent,
    build_quiz_prompt,
    generate_quiz_with_ollama,
    get_level_number,
    call_ollama_streaming,
    check_ollama_server,
    group_chunks_by_material,
    generate_document_summaries_by_file_with_ollama,
    summaries_by_file_to_markdown,
    generate_material_summary_with_ollama,
    fallback_summary_from_chunks,
    build_learning_roadmap_df,
    roadmap_df_to_markdown,
    get_chunks_for_roadmap_day,
    describe_chunks_scope,
    generate_file_review_quizzes_with_ollama,
    generate_roadmap_day_quizzes_with_ollama,
    roadmap_day_quizzes_to_markdown,
    review_quizzes_to_markdown,
    grade_student_answer,
    build_attempt_record,
    analyze_concept_mastery,
    get_target_concepts_for_remedial,
    generate_remedial_quizzes_with_ollama,
    build_tutor_answer_prompt,
    build_tutor_extra_keywords,
    clean_tutor_answer_text,
    sanitize_user_text,
    sanitize_student_output,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "mockdata_korean_history_until_goryeo_v3"
FRONTEND_DIR = BASE_DIR / "frontend"


def load_json(name: str, default: Any) -> Any:
    path = DATA_DIR / name
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


mock_chunks = load_json("mock_chunks.json", {"materials": [], "material_set_name": "데모 자료"})
mock_ai = load_json("mock_ai_outputs.json", {"summary": {}, "tutor_example": {}, "flashcards": []})
mock_questions = load_json("mock_questions.json", {"user_id": 1, "questions": []})
mock_analytics = load_json("mock_analytics.json", {
    "wrong_notes": [],
    "weak_concepts": [],
    "study_plan": [],
    "dashboard_summary": {
        "user_id": 1,
        "total_solved": 0,
        "correct_count": 0,
        "wrong_count": 0,
        "accuracy": 0,
        "recommended_next": "먼저 자료를 업로드하고 문제를 풀어보세요."
    }
})

app = FastAPI(title="LectureMate AI Complete Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_SETTINGS = {
    "collection_name": "pdf_lecture_chunks",
    "ollama_url": "http://localhost:11434/api/generate",
    "ollama_model": "qwen3:4b",
    "top_k": 2,
    "chunk_size": 800,
    "chunk_overlap": 120,
    "num_predict": 900,
    "stream_timeout": 120,
    "extra_keywords": [],
    "banned_terms": [],
}

app_state: Dict[str, Any] = {
    "user_id": 1,
    "settings": DEFAULT_SETTINGS.copy(),
    "materials": [],
    "pages": [],
    "chunks": [],
    "collection_ready": False,
    "questions": [],
    "attempts": [],
    "summary_cache": {},
    "file_summaries": [],
    "roadmap": None,
    "roadmap_markdown": "",
    "generated_sets": {},
    "latest_question_set": {"set_key": "", "title": "", "origin": "", "created_at": "", "questions": []},
    "question_history_sets": [],
    "tutor_history": [],
}


class SettingsRequest(BaseModel):
    ollama_url: Optional[str] = None
    ollama_model: Optional[str] = None
    collection_name: Optional[str] = None
    top_k: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    num_predict: Optional[int] = None
    stream_timeout: Optional[int] = None
    extra_keywords: Optional[List[str]] = None
    banned_terms: Optional[List[str]] = None


class TutorRequest(BaseModel):
    user_id: int = 1
    question: str
    top_k: int = 6


class QuestionRequest(BaseModel):
    user_id: int = 1
    query: str
    student_level: str = "Level 2 - 중급"
    question_type: str = "4지선다 객관식"
    question_direction: str = "자동(긍정형 우선)"
    weak_concept: str = ""
    top_k: int = 2


class AnswerSubmitRequest(BaseModel):
    user_id: int = 1
    question_id: int
    selected_answer: str
    elapsed_time: int = 30
    confidence: int = Field(default=3, ge=1, le=5)
    used_hint: bool = False


class FileReviewQuizRequest(BaseModel):
    user_id: int = 1
    material_id: str
    quiz_count: int = Field(default=1, ge=1, le=5)
    student_level: str = "Level 2 - 중급"
    question_type_mode: str = "혼합"


class RoadmapRequest(BaseModel):
    user_id: int = 1
    duration_days: int = Field(default=7, ge=1, le=365)


class RoadmapDayQuizRequest(BaseModel):
    user_id: int = 1
    duration_days: int = Field(default=7, ge=1, le=365)
    day: int = Field(default=1, ge=1)
    quiz_count: int = Field(default=3, ge=1, le=5)
    student_level: str = "Level 2 - 중급"
    question_type_mode: str = "혼합"


class RemedialQuizRequest(BaseModel):
    user_id: int = 1
    target_concept: str
    quiz_count: int = Field(default=3, ge=1, le=5)
    student_level: str = "Level 2 - 중급"
    question_type_mode: str = "혼합"
    top_k: int = 3


def settings() -> Dict[str, Any]:
    return app_state["settings"]


def normalize_question_type(qtype: str) -> str:
    qtype = str(qtype)
    if "OX" in qtype or "O/X" in qtype or "ox" in qtype.lower():
        return "ox"
    if "빈칸" in qtype or "주관식" in qtype:
        return "fill_blank"
    return "multiple_choice"


def get_collection_name() -> str:
    return settings().get("collection_name", DEFAULT_SETTINGS["collection_name"])


def get_material_chunks(material_id: str | None = None, file_name: str | None = None) -> List[Dict[str, Any]]:
    chunks = app_state.get("chunks", [])
    if material_id:
        return [c for c in chunks if str(c.get("material_id", "")) == str(material_id)]
    if file_name:
        return [c for c in chunks if str(c.get("file_name", "")) == str(file_name)]
    return chunks


def get_current_questions() -> List[Dict[str, Any]]:
    if app_state.get("questions"):
        return app_state["questions"]
    return mock_questions.get("questions", [])


def to_records(df) -> List[Dict[str, Any]]:
    if df is None or getattr(df, "empty", True):
        return []
    return df.to_dict(orient="records")


def make_source_label(file_name: str, pages: Any) -> str:
    """화면에 보여줄 출처를 '문서명 p.페이지' 형식으로 만든다."""
    file_name = sanitize_user_text(file_name or "")
    if isinstance(pages, list):
        page_values = [str(p).strip() for p in pages if str(p).strip()]
    elif pages is None or str(pages).strip() == "":
        page_values = []
    else:
        page_values = [str(pages).strip()]

    page_text = ", ".join(page_values)
    if file_name and page_text:
        return f"{file_name} p.{page_text}"
    if file_name:
        return file_name
    if page_text:
        return f"p.{page_text}"
    return ""


def make_question_set_title(origin: str, questions: List[Dict[str, Any]], explicit_title: Optional[str] = None) -> str:
    """생성된 문제 묶음을 히스토리 목록에서 보기 좋은 제목으로 만든다."""
    if explicit_title:
        return sanitize_user_text(explicit_title)

    count = len(questions or [])
    first = questions[0] if questions else {}
    source = first.get("source", {}) if isinstance(first.get("source"), dict) else {}
    file_name = source.get("file_name", "") or first.get("file_name", "")
    concept = first.get("concept", "")

    if file_name:
        return sanitize_user_text(f"{origin} · {file_name} · {count}문제")
    if concept:
        return sanitize_user_text(f"{origin} · {concept} · {count}문제")
    return sanitize_user_text(f"{origin} · {count}문제")


def update_latest_question_set(
    questions: List[Dict[str, Any]],
    origin: str,
    set_key: str,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    직전 생성 결과와 전체 문제 히스토리를 분리해서 저장한다.
    - latest_question_set: 화면 중앙의 '직전 생성 문제' 영역에만 표시
    - question_history_sets: 옆 탭/목록의 '지금까지 생성된 전체 문제'에서 표시
    """
    safe_questions = questions or []
    record = {
        "set_key": set_key,
        "title": make_question_set_title(origin, safe_questions, title),
        "origin": origin,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question_count": len(safe_questions),
        "questions": safe_questions,
    }

    app_state["latest_question_set"] = record

    history = app_state.setdefault("question_history_sets", [])
    history = [item for item in history if item.get("set_key") != set_key]
    history.append(record)
    app_state["question_history_sets"] = history

    return record


def get_all_questions_for_user(user_id: int = 1) -> List[Dict[str, Any]]:
    return [
        q for q in get_current_questions()
        if int(q.get("user_id", 1)) == int(user_id)
    ]


def source_from_quiz_or_chunk(quiz: Dict[str, Any], source_chunk: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    source_chunk = source_chunk or {}
    raw_pages = quiz.get("source_pages") or []
    pages = raw_pages if isinstance(raw_pages, list) else [raw_pages]
    page = pages[0] if pages else source_chunk.get("page")
    if (not pages or not pages[0]) and page:
        pages = [page]

    file_name = sanitize_user_text(
        quiz.get("file_name")
        or quiz.get("source_file")
        or quiz.get("material_name")
        or source_chunk.get("file_name", "")
    )
    evidence = sanitize_user_text(quiz.get("evidence_text") or source_chunk.get("text", "")[:280])
    return {
        "file_name": file_name,
        "page": page,
        "pages": pages,
        "label": make_source_label(file_name, pages),
        "evidence": evidence,
    }


def add_quizzes_to_question_store(
    quizzes: List[Dict[str, Any]],
    user_id: int,
    origin: str,
    source_chunks: Optional[List[Dict[str, Any]]] = None,
    set_key: Optional[str] = None,
    set_title: Optional[str] = None,
    update_latest: bool = True,
) -> List[Dict[str, Any]]:
    created = []
    source_chunks = source_chunks or []

    for idx, quiz in enumerate(quizzes):
        qtype = quiz.get("question_type") or normalize_question_type(quiz.get("type", ""))
        source_chunk = source_chunks[idx] if idx < len(source_chunks) else None
        quiz = sanitize_student_output(quiz)
        question = {
            "question_id": len(app_state.get("questions", [])) + 1,
            "user_id": user_id,
            "origin": origin,
            "origin_key": set_key or origin,
            "type": qtype,
            "question": sanitize_user_text(quiz.get("question", "")),
            "choices": sanitize_student_output(quiz.get("choices", [])),
            "answer": sanitize_user_text(quiz.get("answer", "")),
            "explanation": sanitize_user_text(quiz.get("explanation", "")),
            "concept": sanitize_user_text(quiz.get("concept", "")),
            "source": source_from_quiz_or_chunk(quiz, source_chunk),
            "raw_quiz": quiz
        }
        app_state.setdefault("questions", []).append(question)
        created.append(question)

    final_set_key = set_key or f"{origin}_{len(app_state.get('generated_sets', {})) + 1}"
    app_state.setdefault("generated_sets", {})[final_set_key] = created

    if update_latest and created:
        update_latest_question_set(
            questions=created,
            origin=origin,
            set_key=final_set_key,
            title=set_title,
        )

    return created


def dashboard_from_attempts(user_id: int = 1) -> Dict[str, Any]:
    attempts = [a for a in app_state.get("attempts", []) if int(a.get("user_id", 1)) == int(user_id)]
    if not attempts:
        return mock_analytics.get("dashboard_summary", {})
    total = len(attempts)
    correct = sum(1 for a in attempts if a.get("is_correct"))
    wrong = total - correct
    accuracy = round(correct / max(1, total) * 100)
    mastery_df = analyze_concept_mastery(attempts)
    targets = get_target_concepts_for_remedial(mastery_df)
    recommended = f"{targets[0]} 보충 문제 풀기" if targets else "짧은 복습 유지"
    return {
        "user_id": user_id,
        "total_solved": total,
        "correct_count": correct,
        "wrong_count": wrong,
        "accuracy": accuracy,
        "recommended_next": recommended
    }


def wrong_notes_from_attempts(user_id: int = 1) -> List[Dict[str, Any]]:
    notes = []
    for a in app_state.get("attempts", []):
        if int(a.get("user_id", 1)) != int(user_id):
            continue
        if a.get("is_correct"):
            continue
        source = a.get("source") if isinstance(a.get("source"), dict) else {}
        notes.append({
            "user_id": user_id,
            "question_id": a.get("question_id"),
            "question": a.get("question", ""),
            "selected_answer": a.get("selected_answer") or a.get("student_answer", ""),
            "correct_answer": a.get("correct_answer") or a.get("answer", ""),
            "concept": a.get("concept", "기타 개념"),
            "explanation": a.get("explanation", ""),
            "evidence_text": a.get("evidence_text", ""),
            "source": source,
            "source_file": source.get("file_name", ""),
            "source_label": source.get("label") or make_source_label(source.get("file_name", ""), source.get("pages") or source.get("page")),
            "source_page": source.get("page", "-"),
            "source_pages": a.get("source_pages", []) or source.get("pages", []),
        })
    return notes


def weak_concepts_from_attempts(user_id: int = 1) -> List[Dict[str, Any]]:
    attempts = [a for a in app_state.get("attempts", []) if int(a.get("user_id", 1)) == int(user_id)]
    mastery_df = analyze_concept_mastery(attempts)
    rows = to_records(mastery_df)
    for row in rows:
        row["accuracy_percent"] = round(float(row.get("accuracy", 0)) * 100)
        row["reason"] = row.get("recommendation", "최근 풀이 기록을 기준으로 복습이 필요합니다.")
    return rows


def current_file_summaries_map() -> Dict[str, str]:
    result = {}
    for item in app_state.get("file_summaries", []) or []:
        if isinstance(item, dict):
            result[str(item.get("file_name", "업로드 자료"))] = str(item.get("summary", ""))
    return result


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "collection_ready": app_state["collection_ready"],
        "uploaded_material_count": len(app_state["materials"]),
        "chunk_count": len(app_state["chunks"]),
        "settings": settings(),
    }


@app.get("/api/settings")
def get_settings():
    return settings()


@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    updates = req.model_dump(exclude_none=True) if hasattr(req, "model_dump") else req.dict(exclude_none=True)
    for key, value in updates.items():
        app_state["settings"][key] = value
    return settings()


@app.get("/api/ollama/check")
def ollama_check():
    ok = check_ollama_server(settings()["ollama_url"])
    return {"ok": ok, "ollama_url": settings()["ollama_url"]}


@app.get("/api/materials")
def get_materials(user_id: int = 1):
    if app_state["materials"]:
        return {"user_id": user_id, "materials": app_state["materials"], "material_set_name": "업로드 자료 세트"}
    return {"user_id": user_id, "materials": mock_chunks.get("materials", []), "material_set_name": mock_chunks.get("material_set_name", "데모 자료")}


@app.get("/api/materials/list")
def list_materials(user_id: int = 1):
    return {"user_id": user_id, "materials": app_state.get("materials", [])}


@app.post("/api/materials/upload")
async def upload_materials(
    user_id: int = Form(1),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    files: List[UploadFile] = File(...)
):
    if chunk_size is not None:
        app_state["settings"]["chunk_size"] = int(chunk_size)
    if chunk_overlap is not None:
        app_state["settings"]["chunk_overlap"] = int(chunk_overlap)

    embedding_model = load_embedding_model()
    client = get_chroma_client()
    collection = create_or_reset_collection(client, get_collection_name())

    all_pages = []
    all_chunks = []
    materials = []

    for idx, file in enumerate(files, start=1):
        file_bytes = await file.read()
        file_name = file.filename or f"uploaded_{idx}.pdf"
        material_id = make_safe_id(f"u{user_id}_m{idx}_{file_name}")

        pages = extract_text_from_pdf(file_bytes=file_bytes, file_name=file_name, material_id=material_id)
        chunks = make_chunks_from_pages(
            pages,
            chunk_size=int(settings()["chunk_size"]),
            chunk_overlap=int(settings()["chunk_overlap"])
        )

        for page in pages:
            page["user_id"] = user_id
        for chunk in chunks:
            chunk["user_id"] = user_id

        all_pages.extend(pages)
        all_chunks.extend(chunks)

        materials.append({
            "user_id": user_id,
            "material_id": material_id,
            "title": file_name,
            "file_name": file_name,
            "pages": len(pages),
            "chunks": len(chunks),
            "chars": sum(int(p.get("char_count", 0)) for p in pages),
            "status": "분석 완료"
        })

    saved = store_chunks_in_chroma(collection, all_chunks, embedding_model)

    app_state["user_id"] = user_id
    app_state["materials"] = materials
    app_state["pages"] = all_pages
    app_state["chunks"] = all_chunks
    app_state["collection_ready"] = True
    app_state["questions"] = []
    app_state["attempts"] = []
    app_state["summary_cache"] = {}
    app_state["file_summaries"] = []
    app_state["roadmap"] = None
    app_state["roadmap_markdown"] = ""
    app_state["generated_sets"] = {}
    app_state["latest_question_set"] = {"set_key": "", "title": "", "origin": "", "created_at": "", "questions": []}
    app_state["question_history_sets"] = []
    app_state["tutor_history"] = []

    return {
        "user_id": user_id,
        "message": "PDF 처리 및 ChromaDB 저장 완료",
        "materials": materials,
        "saved_chunks": saved,
        "total_pages": len(all_pages),
        "total_chunks": len(all_chunks),
        "collection_name": get_collection_name(),
    }


@app.get("/api/chunks/preview")
def chunks_preview(limit: int = 120):
    rows = []
    for c in app_state.get("chunks", [])[: int(limit)]:
        rows.append({
            "chunk_id": c.get("chunk_id"),
            "file_name": c.get("file_name", ""),
            "material_id": c.get("material_id", ""),
            "page": c.get("page"),
            "char_count": c.get("char_count"),
            "quality_score": c.get("quality_score"),
            "preview": str(c.get("text", ""))[:180].replace("\n", " ")
        })
    return {"chunks": rows, "total": len(app_state.get("chunks", []))}


@app.get("/api/materials/summary")
def get_summary(user_id: int = 1, material_id: str | None = None, file_name: str | None = None):
    if not app_state.get("chunks"):
        return mock_ai.get("summary", {})

    target_chunks = get_material_chunks(material_id=material_id, file_name=file_name)
    if not target_chunks:
        return {"user_id": user_id, "summary": "선택한 자료에서 분석할 수 있는 텍스트를 찾지 못했습니다.", "keywords": [], "raw_markdown": ""}

    cache_key = material_id or file_name or "__all__"
    if cache_key in app_state["summary_cache"]:
        return app_state["summary_cache"][cache_key]

    summary_result = generate_material_summary_with_ollama(
        chunks=target_chunks,
        ollama_url=settings()["ollama_url"],
        model_name=settings()["ollama_model"],
        user_id=user_id,
        num_predict=int(settings()["num_predict"]),
        stream_read_timeout=int(settings()["stream_timeout"])
    )
    # 화면에서 개념 카드가 따로 뜨지 않도록 raw_markdown/summary 중심으로 반환한다.
    app_state["summary_cache"][cache_key] = summary_result
    return summary_result


@app.post("/api/materials/summaries-by-file")
def generate_summaries_by_file(user_id: int = 1):
    if not app_state.get("chunks"):
        return {"user_id": user_id, "summaries": [], "markdown": ""}
    summaries = generate_document_summaries_by_file_with_ollama(
        ollama_url=settings()["ollama_url"],
        model_name=settings()["ollama_model"],
        chunks=app_state["chunks"],
        num_predict=int(settings()["num_predict"]),
        stream_read_timeout=int(settings()["stream_timeout"])
    )
    app_state["file_summaries"] = summaries
    markdown = summaries_by_file_to_markdown(summaries)
    return {"user_id": user_id, "summaries": summaries, "markdown": markdown}


@app.get("/api/materials/summaries-by-file")
def get_summaries_by_file(user_id: int = 1):
    markdown = summaries_by_file_to_markdown(app_state.get("file_summaries", [])) if app_state.get("file_summaries") else ""
    return {"user_id": user_id, "summaries": app_state.get("file_summaries", []), "markdown": markdown}


@app.post("/api/materials/review-quizzes")
def generate_file_review(req: FileReviewQuizRequest):
    if not app_state.get("chunks"):
        return {"user_id": req.user_id, "questions": [], "message": "먼저 PDF를 업로드하세요."}
    if not check_ollama_server(settings()["ollama_url"]):
        return {"user_id": req.user_id, "questions": [], "message": "Ollama 서버에 연결할 수 없습니다."}

    quizzes = generate_file_review_quizzes_with_ollama(
        ollama_url=settings()["ollama_url"],
        model_name=settings()["ollama_model"],
        chunks=app_state["chunks"],
        material_id=req.material_id,
        quiz_count=req.quiz_count,
        student_level=req.student_level,
        question_type_mode=req.question_type_mode,
        num_predict=int(settings()["num_predict"]),
        stream_read_timeout=int(settings()["stream_timeout"]),
        banned_terms=settings().get("banned_terms", [])
    )
    set_key = f"file_review_{req.material_id}_{len(app_state.get('generated_sets', {})) + 1}"
    questions = add_quizzes_to_question_store(
        quizzes,
        req.user_id,
        "요약 기반 복습 문제",
        set_key=set_key,
        set_title=f"파일별 학습 확인 문제 · {req.material_id} · {req.quiz_count}문제",
    )
    file_name = questions[0].get("source", {}).get("file_name", req.material_id) if questions else req.material_id
    markdown = review_quizzes_to_markdown(file_name, quizzes)
    return {"user_id": req.user_id, "questions": questions, "quizzes": quizzes, "markdown": markdown, "set_key": set_key}


@app.post("/api/roadmap/generate")
def generate_roadmap(req: RoadmapRequest):
    if not app_state.get("chunks"):
        return {"user_id": req.user_id, "rows": [], "markdown": "", "message": "먼저 PDF를 업로드하세요."}
    roadmap_df = build_learning_roadmap_df(
        chunks=app_state["chunks"],
        duration_days=req.duration_days,
        file_summaries=app_state.get("file_summaries", [])
    )
    markdown = roadmap_df_to_markdown(roadmap_df, req.duration_days)
    rows = to_records(roadmap_df)
    app_state["roadmap"] = {"duration_days": req.duration_days, "rows": rows}
    app_state["roadmap_markdown"] = markdown
    return {"user_id": req.user_id, "duration_days": req.duration_days, "rows": rows, "markdown": markdown}


@app.get("/api/roadmap")
def get_roadmap(user_id: int = 1):
    roadmap = app_state.get("roadmap") or {"duration_days": 0, "rows": []}
    return {"user_id": user_id, **roadmap, "markdown": app_state.get("roadmap_markdown", "")}


@app.get("/api/roadmap/day-scope")
def get_roadmap_day_scope(duration_days: int = 7, day: int = 1):
    day_chunks = get_chunks_for_roadmap_day(app_state.get("chunks", []), duration_days, day)
    return {"duration_days": duration_days, "day": day, "scope": describe_chunks_scope(day_chunks)}


@app.post("/api/roadmap/day-quizzes")
def generate_roadmap_day_quizzes(req: RoadmapDayQuizRequest):
    if not app_state.get("chunks"):
        return {"user_id": req.user_id, "questions": [], "message": "먼저 PDF를 업로드하세요."}
    if not check_ollama_server(settings()["ollama_url"]):
        return {"user_id": req.user_id, "questions": [], "message": "Ollama 서버에 연결할 수 없습니다."}

    quizzes = generate_roadmap_day_quizzes_with_ollama(
        ollama_url=settings()["ollama_url"],
        model_name=settings()["ollama_model"],
        chunks=app_state["chunks"],
        duration_days=req.duration_days,
        day=req.day,
        quiz_count=req.quiz_count,
        student_level=req.student_level,
        question_type_mode=req.question_type_mode,
        num_predict=int(settings()["num_predict"]),
        stream_read_timeout=int(settings()["stream_timeout"]),
        banned_terms=settings().get("banned_terms", [])
    )
    day_chunks = get_chunks_for_roadmap_day(app_state.get("chunks", []), req.duration_days, req.day)
    scope = describe_chunks_scope(day_chunks)
    set_key = f"roadmap_{req.duration_days}_day_{req.day}_{len(app_state.get('generated_sets', {})) + 1}"
    questions = add_quizzes_to_question_store(
        quizzes,
        req.user_id,
        "로드맵 문제",
        set_key=set_key,
        set_title=f"로드맵 Day {req.day} 학습 확인 문제 · {req.quiz_count}문제",
    )
    markdown = roadmap_day_quizzes_to_markdown(req.duration_days, req.day, scope, quizzes)
    return {"user_id": req.user_id, "questions": questions, "quizzes": quizzes, "scope": scope, "markdown": markdown, "set_key": set_key}


@app.post("/api/tutor/ask")
def ask_tutor(req: TutorRequest):
    if not app_state["collection_ready"]:
        result = mock_ai.get("tutor_example", {}).copy()
        result["question"] = req.question
        result["notice"] = "아직 PDF가 업로드되지 않아 mock 답변을 반환했습니다."
        return result

    embedding_model = load_embedding_model()
    client = get_chroma_client()
    collection = client.get_collection(name=get_collection_name())

    retrieved = search_chunks(
        collection=collection,
        query=req.question,
        embedding_model=embedding_model,
        top_k=max(req.top_k, 6),
        extra_keywords=build_tutor_extra_keywords(req.question)
    )

    if not retrieved:
        return {"user_id": req.user_id, "question": req.question, "answer": "업로드된 자료에서 관련 내용을 찾지 못했습니다.", "sources": []}

    prompt = build_tutor_answer_prompt(
        question=req.question,
        retrieved_chunks=retrieved,
        recent_history=app_state.get("tutor_history", [])
    )

    payload = {
        "model": settings()["ollama_model"],
        "prompt": "/no_think\n" + prompt,
        "stream": True,
        "format": "json",
        "think": False,
        "options": {"temperature": 0.1, "num_predict": 500}
    }

    try:
        raw_answer = call_ollama_streaming(
            ollama_url=settings()["ollama_url"],
            payload=payload,
            stream_read_timeout=90
        )
        answer = sanitize_user_text(clean_tutor_answer_text(raw_answer))
    except Exception as e:
        answer = f"자료에서 관련 근거는 찾았지만 답변 생성 중 오류가 발생했습니다: {e}"

    app_state.setdefault("tutor_history", []).append({"role": "user", "content": req.question})
    app_state.setdefault("tutor_history", []).append({"role": "assistant", "content": answer})
    app_state["tutor_history"] = app_state["tutor_history"][-10:]

    return {
        "user_id": req.user_id,
        "question": req.question,
        "answer": answer,
        "sources": [
            {"file_name": sanitize_user_text(c.get("file_name", "")), "page": c.get("page"), "evidence": sanitize_user_text(c.get("text", "")[:280])}
            for c in retrieved[:4]
        ]
    }


@app.get("/api/review/questions")
def get_questions(user_id: int = 1):
    """지금까지 생성된 전체 문제 목록. 옆 탭의 '지금까지 생성된 전체 문제'에서 사용한다."""
    return {"user_id": user_id, "questions": get_all_questions_for_user(user_id)}


@app.get("/api/review/questions/latest")
def get_latest_question_set(user_id: int = 1):
    """직전 요청으로 생성된 문제 묶음만 반환한다."""
    latest = app_state.get("latest_question_set", {}) or {}
    questions = [
        q for q in latest.get("questions", [])
        if int(q.get("user_id", 1)) == int(user_id)
    ]
    return {
        "user_id": user_id,
        "set_key": latest.get("set_key", ""),
        "title": latest.get("title", ""),
        "origin": latest.get("origin", ""),
        "created_at": latest.get("created_at", ""),
        "question_count": len(questions),
        "questions": questions,
    }


@app.get("/api/review/questions/history")
def get_question_history(user_id: int = 1):
    """문제 생성 요청 단위로 묶은 전체 히스토리. 왼쪽/오른쪽 사이드 목록 탭에서 사용한다."""
    history = []
    for item in app_state.get("question_history_sets", []):
        questions = [
            q for q in item.get("questions", [])
            if int(q.get("user_id", 1)) == int(user_id)
        ]
        if not questions:
            continue
        history.append({
            "set_key": item.get("set_key", ""),
            "title": item.get("title", ""),
            "origin": item.get("origin", ""),
            "created_at": item.get("created_at", ""),
            "question_count": len(questions),
            "questions": questions,
        })
    return {
        "user_id": user_id,
        "sets": history,
        "questions": get_all_questions_for_user(user_id),
        "total_questions": len(get_all_questions_for_user(user_id)),
    }


@app.post("/api/review/questions")
def generate_question(req: QuestionRequest):
    if not app_state["collection_ready"]:
        return {"user_id": req.user_id, "questions": mock_questions.get("questions", []), "notice": "아직 PDF가 업로드되지 않아 mock 문제를 반환했습니다."}
    if not check_ollama_server(settings()["ollama_url"]):
        return {"user_id": req.user_id, "questions": [], "message": "Ollama 서버에 연결할 수 없습니다."}

    embedding_model = load_embedding_model()
    client = get_chroma_client()
    collection = client.get_collection(name=get_collection_name())

    search_query = f"{req.query} {req.weak_concept}".strip()
    retrieved = search_chunks(
        collection=collection,
        query=search_query,
        embedding_model=embedding_model,
        top_k=req.top_k,
        extra_keywords=settings().get("extra_keywords", [])
    )

    if not retrieved:
        return {"user_id": req.user_id, "questions": [], "message": "관련 청크를 찾지 못했습니다."}

    allowed_pages = [int(c["page"]) for c in retrieved if c.get("page") is not None]
    context_text = "\n\n".join(c["text"] for c in retrieved)
    question_intent = detect_question_intent(req.query)

    prompt = build_quiz_prompt(
        user_query=req.query,
        retrieved_chunks=retrieved,
        student_level=req.student_level,
        weak_concept=req.weak_concept,
        question_type=req.question_type,
        question_direction=req.question_direction,
        question_intent=question_intent
    )

    quiz = generate_quiz_with_ollama(
        prompt=prompt,
        model_name=settings()["ollama_model"],
        ollama_url=settings()["ollama_url"],
        allowed_pages=allowed_pages,
        expected_difficulty=get_level_number(req.student_level),
        context_text=context_text,
        num_predict=int(settings()["num_predict"]),
        stream_read_timeout=int(settings()["stream_timeout"]),
        banned_terms=settings().get("banned_terms", [])
    )

    set_key = f"latest_user_quiz_{len(app_state.get('generated_sets', {})) + 1}"
    created = add_quizzes_to_question_store(
        [quiz],
        req.user_id,
        "맞춤 문제",
        source_chunks=[retrieved[0]],
        set_key=set_key,
        set_title=f"학생 요청 기반 문제 · {req.query[:30]}",
    )
    return {
        "user_id": req.user_id,
        # 화면 중앙에는 직전 요청으로 생성된 문제만 보여주기 위해 created만 반환한다.
        "questions": created,
        "latest_question": created[0],
        "latest_question_set": app_state.get("latest_question_set", {}),
        # 필요하면 옆 탭의 히스토리에서 전체 문제를 불러올 수 있다.
        "all_questions_count": len(app_state.get("questions", [])),
        "retrieved_chunks": [
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
        ],
        "prompt": prompt,
        "question_intent": question_intent,
    }


@app.get("/api/review/flashcards")
def get_flashcards(user_id: int = 1):
    # 원본 Streamlit에는 별도 개념 카드 UI가 없어서 mock flashcard만 유지한다.
    return {"user_id": user_id, "flashcards": mock_ai.get("flashcards", [])}


@app.post("/api/answers/submit")
def submit_answer(req: AnswerSubmitRequest):
    target = None
    for q in get_current_questions():
        if int(q.get("question_id", -1)) == int(req.question_id):
            target = q
            break

    if target is None:
        return {"error": "question_not_found"}

    raw_quiz = target.get("raw_quiz", {}) or {}
    quiz_for_grade = dict(raw_quiz)
    quiz_for_grade.setdefault("question_type", target.get("type", "multiple_choice"))
    quiz_for_grade.setdefault("answer", target.get("answer", ""))
    quiz_for_grade.setdefault("question", target.get("question", ""))
    quiz_for_grade.setdefault("concept", target.get("concept", ""))
    quiz_for_grade.setdefault("source_pages", raw_quiz.get("source_pages", []))

    is_correct = grade_student_answer(quiz_for_grade, req.selected_answer)

    attempt = build_attempt_record(
        quiz=quiz_for_grade,
        student_answer=req.selected_answer,
        is_correct=is_correct,
        confidence=req.confidence,
        used_hint=req.used_hint,
        origin=target.get("origin", "학습 문제"),
        origin_key=target.get("origin_key", "")
    )
    source = target.get("source", {}) or {}
    attempt.update({
        "user_id": req.user_id,
        "question_id": req.question_id,
        "selected_answer": req.selected_answer,
        "correct_answer": target.get("answer", ""),
        "elapsed_time": req.elapsed_time,
        "source": source,
        "source_label": source.get("label") or make_source_label(source.get("file_name", ""), source.get("pages") or source.get("page")),
        "raw_quiz": raw_quiz,
        "part_summary": raw_quiz.get("part_summary", ""),
        "choice_explanations": raw_quiz.get("choice_explanations", []),
        "hint": raw_quiz.get("hint", ""),
        "grading_criteria": raw_quiz.get("grading_criteria", []),
    })

    app_state["attempts"].append(attempt)
    return attempt


@app.get("/api/attempts")
def get_attempts(user_id: int = 1):
    attempts = [a for a in app_state.get("attempts", []) if int(a.get("user_id", 1)) == int(user_id)]
    return {"user_id": user_id, "attempts": attempts}


@app.post("/api/attempts/reset")
def reset_attempts(user_id: int = 1):
    app_state["attempts"] = [a for a in app_state.get("attempts", []) if int(a.get("user_id", 1)) != int(user_id)]
    return {"user_id": user_id, "message": "풀이 기록을 초기화했습니다."}


@app.get("/api/wrong-notes")
def get_wrong_notes(user_id: int = 1):
    return {"user_id": user_id, "wrong_notes": wrong_notes_from_attempts(user_id)}


@app.get("/api/weak-concepts")
def get_weak_concepts(user_id: int = 1):
    return {"user_id": user_id, "weak_concepts": weak_concepts_from_attempts(user_id)}


@app.get("/api/mastery")
def get_mastery(user_id: int = 1):
    attempts = [a for a in app_state.get("attempts", []) if int(a.get("user_id", 1)) == int(user_id)]
    mastery_df = analyze_concept_mastery(attempts)
    target_concepts = get_target_concepts_for_remedial(mastery_df)
    return {"user_id": user_id, "mastery": to_records(mastery_df), "target_concepts": target_concepts}


@app.post("/api/remedial/quizzes")
def generate_remedial_quizzes(req: RemedialQuizRequest):
    if not app_state["collection_ready"]:
        return {"user_id": req.user_id, "questions": [], "message": "먼저 PDF를 업로드하고 문제를 푸세요."}
    if not check_ollama_server(settings()["ollama_url"]):
        return {"user_id": req.user_id, "questions": [], "message": "Ollama 서버에 연결할 수 없습니다."}

    embedding_model = load_embedding_model()
    client = get_chroma_client()
    collection = client.get_collection(name=get_collection_name())

    quizzes = generate_remedial_quizzes_with_ollama(
        ollama_url=settings()["ollama_url"],
        model_name=settings()["ollama_model"],
        collection=collection,
        embedding_model=embedding_model,
        target_concept=req.target_concept,
        quiz_count=req.quiz_count,
        student_level=req.student_level,
        question_type_mode=req.question_type_mode,
        top_k=req.top_k,
        extra_keywords=settings().get("extra_keywords", []),
        num_predict=int(settings()["num_predict"]),
        stream_read_timeout=int(settings()["stream_timeout"]),
        banned_terms=settings().get("banned_terms", [])
    )
    set_key = f"remedial_{make_safe_id(req.target_concept)}_{len(app_state.get('generated_sets', {})) + 1}"
    questions = add_quizzes_to_question_store(
        quizzes,
        req.user_id,
        "보충 문제",
        set_key=set_key,
        set_title=f"{req.target_concept} 보충 문제 · {req.quiz_count}문제",
    )
    markdown = review_quizzes_to_markdown(f"{req.target_concept}_보충", quizzes)
    return {"user_id": req.user_id, "questions": questions, "quizzes": quizzes, "markdown": markdown, "set_key": set_key}


@app.get("/api/study-plan")
def get_study_plan(user_id: int = 1):
    weak = weak_concepts_from_attempts(user_id)
    if weak:
        targets = [w for w in weak if w.get("wrong", 0) > 0 or w.get("status") != "학습됨"][:3] or weak[:2]
        plan = []
        for idx, w in enumerate(targets, start=1):
            plan.append({
                "user_id": user_id,
                "step": idx,
                "title": f"{w['concept']} 복습",
                "time": "10분",
                "reason": w.get("recommendation", "최근 풀이 기록을 기준으로 복습이 필요합니다.")
            })
        return {"user_id": user_id, "study_plan": plan}
    return {"user_id": user_id, "study_plan": []}


@app.get("/api/dashboard/summary")
def get_dashboard_summary(user_id: int = 1):
    return dashboard_from_attempts(user_id)


@app.get("/api/download/file-summaries.md")
def download_file_summaries_md():
    return {"file_name": "file_summaries_by_pdf.md", "markdown": summaries_by_file_to_markdown(app_state.get("file_summaries", []))}


@app.get("/api/download/roadmap.md")
def download_roadmap_md():
    duration = (app_state.get("roadmap") or {}).get("duration_days", 0)
    return {"file_name": f"learning_roadmap_{duration}days.md", "markdown": app_state.get("roadmap_markdown", "")}


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

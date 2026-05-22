from pathlib import Path
from typing import List
import json

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
    generate_material_summary_with_ollama,
    fallback_summary_from_chunks,
    build_tutor_answer_prompt,
    build_tutor_extra_keywords,
    clean_tutor_answer_text,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "mockdata_korean_history_until_goryeo_v3"
FRONTEND_DIR = BASE_DIR / "frontend"

def load_json(name: str, default):
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

app = FastAPI(title="LectureMate AI Connected Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COLLECTION_NAME = "pdf_lecture_chunks"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:4b"

app_state = {
    "user_id": 1,
    "materials": [],
    "pages": [],
    "chunks": [],
    "collection_ready": False,
    "questions": [],
    "attempts": [],
    "summary_cache": {},
    "tutor_history": [],
}

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

def normalize_question_type(qtype: str) -> str:
    qtype = str(qtype)
    if "OX" in qtype or "O/X" in qtype or "ox" in qtype.lower():
        return "ox"
    if "빈칸" in qtype or "주관식" in qtype:
        return "fill_blank"
    return "multiple_choice"

def get_material_chunks(material_id: str | None = None, file_name: str | None = None):
    chunks = app_state.get("chunks", [])
    if material_id:
        return [c for c in chunks if str(c.get("material_id", "")) == str(material_id)]
    if file_name:
        return [c for c in chunks if str(c.get("file_name", "")) == str(file_name)]
    return chunks

def get_current_questions():
    if app_state.get("questions"):
        return app_state["questions"]
    return mock_questions.get("questions", [])

def dashboard_from_attempts(user_id: int = 1):
    attempts = [a for a in app_state.get("attempts", []) if int(a.get("user_id", 1)) == int(user_id)]
    if not attempts:
        return mock_analytics.get("dashboard_summary", {})
    total = len(attempts)
    correct = sum(1 for a in attempts if a.get("is_correct"))
    wrong = total - correct
    accuracy = round(correct / max(1, total) * 100)
    wrong_concepts = {}
    for a in attempts:
        if not a.get("is_correct"):
            concept = a.get("concept") or "기타 개념"
            wrong_concepts[concept] = wrong_concepts.get(concept, 0) + 1
    recommended = "생성 문제 다시 풀기"
    if wrong_concepts:
        recommended = max(wrong_concepts.items(), key=lambda x: x[1])[0] + " 복습"
    return {
        "user_id": user_id,
        "total_solved": total,
        "correct_count": correct,
        "wrong_count": wrong,
        "accuracy": accuracy,
        "recommended_next": recommended
    }

def wrong_notes_from_attempts(user_id: int = 1):
    notes = []
    for a in app_state.get("attempts", []):
        if int(a.get("user_id", 1)) != int(user_id):
            continue
        if a.get("is_correct"):
            continue
        notes.append({
            "user_id": user_id,
            "question_id": a.get("question_id"),
            "question": a.get("question", ""),
            "selected_answer": a.get("selected_answer", ""),
            "correct_answer": a.get("correct_answer", ""),
            "concept": a.get("concept", "기타 개념"),
            "explanation": a.get("explanation", ""),
            "source_page": a.get("source", {}).get("page", "-")
        })
    return notes

def weak_concepts_from_attempts(user_id: int = 1):
    attempts = [a for a in app_state.get("attempts", []) if int(a.get("user_id", 1)) == int(user_id)]
    if not attempts:
        return []
    stats = {}
    for a in attempts:
        concept = a.get("concept") or "기타 개념"
        stats.setdefault(concept, {"total": 0, "correct": 0, "wrong": 0})
        stats[concept]["total"] += 1
        if a.get("is_correct"):
            stats[concept]["correct"] += 1
        else:
            stats[concept]["wrong"] += 1
    rows = []
    for concept, s in stats.items():
        accuracy = round(s["correct"] / max(1, s["total"]) * 100)
        rows.append({
            "user_id": user_id,
            "concept": concept,
            "accuracy": accuracy,
            "wrong_count": s["wrong"],
            "reason": f"{concept} 관련 문제에서 오답 {s['wrong']}회가 발생했습니다." if s["wrong"] else "현재까지 오답이 없습니다."
        })
    return sorted(rows, key=lambda x: (x["accuracy"], -x["wrong_count"]))

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "collection_ready": app_state["collection_ready"],
        "uploaded_material_count": len(app_state["materials"]),
        "chunk_count": len(app_state["chunks"]),
    }

@app.get("/api/materials")
def get_materials(user_id: int = 1):
    if app_state["materials"]:
        return {"user_id": user_id, "materials": app_state["materials"], "material_set_name": "업로드 자료 세트"}
    return {"user_id": user_id, "materials": mock_chunks.get("materials", []), "material_set_name": mock_chunks.get("material_set_name", "데모 자료")}

@app.get("/api/materials/list")
def list_materials(user_id: int = 1):
    return {"user_id": user_id, "materials": app_state.get("materials", [])}

@app.post("/api/materials/upload")
async def upload_materials(user_id: int = Form(1), files: List[UploadFile] = File(...)):
    embedding_model = load_embedding_model()
    client = get_chroma_client()
    collection = create_or_reset_collection(client, COLLECTION_NAME)

    all_pages = []
    all_chunks = []
    materials = []

    for idx, file in enumerate(files, start=1):
        file_bytes = await file.read()
        file_name = file.filename or f"uploaded_{idx}.pdf"
        material_id = make_safe_id(f"u{user_id}_m{idx}_{file_name}")

        pages = extract_text_from_pdf(file_bytes=file_bytes, file_name=file_name, material_id=material_id)
        chunks = make_chunks_from_pages(pages, chunk_size=800, chunk_overlap=120)

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
    app_state["tutor_history"] = []

    return {
        "user_id": user_id,
        "message": "PDF 처리 및 ChromaDB 저장 완료",
        "materials": materials,
        "saved_chunks": saved,
        "total_pages": len(all_pages),
        "total_chunks": len(all_chunks)
    }

@app.get("/api/materials/summary")
def get_summary(user_id: int = 1, material_id: str | None = None, file_name: str | None = None):
    if not app_state.get("chunks"):
        return mock_ai.get("summary", {})

    target_chunks = get_material_chunks(material_id=material_id, file_name=file_name)
    if not target_chunks:
        return {"user_id": user_id, "summary": "선택한 자료에서 분석할 수 있는 텍스트를 찾지 못했습니다.", "keywords": [], "concepts": []}

    cache_key = material_id or file_name or "__all__"
    if cache_key in app_state["summary_cache"]:
        return app_state["summary_cache"][cache_key]

    summary_result = generate_material_summary_with_ollama(
        chunks=target_chunks,
        ollama_url=OLLAMA_URL,
        model_name=OLLAMA_MODEL,
        user_id=user_id,
        num_predict=900,
        stream_read_timeout=120
    )
    app_state["summary_cache"][cache_key] = summary_result
    return summary_result

@app.post("/api/tutor/ask")
def ask_tutor(req: TutorRequest):
    if not app_state["collection_ready"]:
        result = mock_ai.get("tutor_example", {}).copy()
        result["question"] = req.question
        result["notice"] = "아직 PDF가 업로드되지 않아 mock 답변을 반환했습니다."
        return result

    embedding_model = load_embedding_model()
    client = get_chroma_client()
    collection = client.get_collection(name=COLLECTION_NAME)

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
        "model": OLLAMA_MODEL,
        "prompt": "/no_think\n" + prompt,
        "stream": True,
        "think": False,
        "options": {"temperature": 0.1, "num_predict": 500}
    }

    try:
        raw_answer = call_ollama_streaming(ollama_url=OLLAMA_URL, payload=payload, stream_read_timeout=90)
        answer = clean_tutor_answer_text(raw_answer)
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
            {"file_name": c.get("file_name", ""), "page": c.get("page"), "evidence": c.get("text", "")[:280]}
            for c in retrieved[:4]
        ]
    }

@app.get("/api/review/questions")
def get_questions(user_id: int = 1):
    return {"user_id": user_id, "questions": get_current_questions()}

@app.post("/api/review/questions")
def generate_question(req: QuestionRequest):
    if not app_state["collection_ready"]:
        return {"user_id": req.user_id, "questions": mock_questions.get("questions", []), "notice": "아직 PDF가 업로드되지 않아 mock 문제를 반환했습니다."}

    embedding_model = load_embedding_model()
    client = get_chroma_client()
    collection = client.get_collection(name=COLLECTION_NAME)

    search_query = f"{req.query} {req.weak_concept}".strip()
    retrieved = search_chunks(collection=collection, query=search_query, embedding_model=embedding_model, top_k=req.top_k)

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
        model_name=OLLAMA_MODEL,
        ollama_url=OLLAMA_URL,
        allowed_pages=allowed_pages,
        expected_difficulty=get_level_number(req.student_level),
        context_text=context_text,
        num_predict=900,
        stream_read_timeout=120,
        banned_terms=[]
    )

    first_source = retrieved[0]
    source_pages = quiz.get("source_pages", [])
    source_page = source_pages[0] if source_pages else first_source.get("page")
    qtype = quiz.get("question_type") or normalize_question_type(req.question_type)

    question = {
        "question_id": len(app_state.get("questions", [])) + 1,
        "user_id": req.user_id,
        "type": qtype,
        "question": quiz.get("question", ""),
        "choices": quiz.get("choices", []),
        "answer": quiz.get("answer", ""),
        "explanation": quiz.get("explanation", ""),
        "concept": quiz.get("concept", ""),
        "source": {
            "file_name": first_source.get("file_name", ""),
            "page": source_page,
            "evidence": quiz.get("evidence_text", "") or first_source.get("text", "")[:280]
        },
        "raw_quiz": quiz
    }

    app_state["questions"].append(question)

    return {"user_id": req.user_id, "questions": app_state["questions"], "latest_question": question}

@app.get("/api/review/flashcards")
def get_flashcards(user_id: int = 1):
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

    is_correct = str(req.selected_answer).strip() == str(target.get("answer", "")).strip()

    raw_quiz = target.get("raw_quiz", {}) or {}

    attempt = {
        "user_id": req.user_id,
        "question_id": req.question_id,
        "question": target.get("question", ""),
        "selected_answer": req.selected_answer,
        "correct_answer": target.get("answer", ""),
        "is_correct": is_correct,
        "concept": target.get("concept", "기타 개념"),
        "elapsed_time": req.elapsed_time,
        "explanation": target.get("explanation", ""),
        "source": target.get("source", {}),
        "raw_quiz": raw_quiz,
        "question_type": target.get("type", raw_quiz.get("question_type", "multiple_choice")),
        "part_summary": raw_quiz.get("part_summary", ""),
        "evidence_text": raw_quiz.get("evidence_text", target.get("source", {}).get("evidence", "")),
        "choice_explanations": raw_quiz.get("choice_explanations", []),
        "source_pages": raw_quiz.get("source_pages", []),
        "hint": raw_quiz.get("hint", ""),
        "grading_criteria": raw_quiz.get("grading_criteria", []),
        "difficulty": raw_quiz.get("difficulty", "")
    }

    app_state["attempts"].append(attempt)
    return attempt

@app.get("/api/wrong-notes")
def get_wrong_notes(user_id: int = 1):
    notes = wrong_notes_from_attempts(user_id)
    if notes:
        return {"user_id": user_id, "wrong_notes": notes}
    return {"user_id": user_id, "wrong_notes": []}

@app.get("/api/weak-concepts")
def get_weak_concepts(user_id: int = 1):
    weak = weak_concepts_from_attempts(user_id)
    return {"user_id": user_id, "weak_concepts": weak}

@app.get("/api/study-plan")
def get_study_plan(user_id: int = 1):
    weak = weak_concepts_from_attempts(user_id)
    if weak:
        targets = [w for w in weak if w.get("wrong_count", 0) > 0][:3] or weak[:2]
        plan = []
        for idx, w in enumerate(targets, start=1):
            plan.append({
                "user_id": user_id,
                "step": idx,
                "title": f"{w['concept']} 복습",
                "time": "10분",
                "reason": w.get("reason", "최근 풀이 기록을 기준으로 복습이 필요합니다.")
            })
        return {"user_id": user_id, "study_plan": plan}
    return {"user_id": user_id, "study_plan": []}

@app.get("/api/dashboard/summary")
def get_dashboard_summary(user_id: int = 1):
    return dashboard_from_attempts(user_id)

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

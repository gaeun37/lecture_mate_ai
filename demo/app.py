
from pathlib import Path
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "mockdata_korean_history_until_goryeo_v3"

def load_json(name: str):
    with open(DATA_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)

mock_chunks = load_json("mock_chunks.json")
mock_ai = load_json("mock_ai_outputs.json")
mock_questions = load_json("mock_questions.json")
mock_attempts = load_json("mock_attempts.json")
mock_analytics = load_json("mock_analytics.json")

app = FastAPI(title="LectureMate AI Connected Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TutorRequest(BaseModel):
    user_id: int = 1
    question: str

class AnswerSubmitRequest(BaseModel):
    user_id: int = 1
    question_id: int
    selected_answer: str
    elapsed_time: int = 30

@app.get("/api/health")
def health():
    return {"status": "ok", "message": "LectureMate AI backend connected"}

@app.get("/api/materials")
def get_materials(user_id: int = 1):
    return {
        "user_id": user_id,
        "materials": mock_chunks["materials"],
        "material_set_name": mock_chunks["material_set_name"]
    }

@app.get("/api/materials/summary")
def get_summary(user_id: int = 1):
    return mock_ai["summary"]

@app.post("/api/tutor/ask")
def ask_tutor(req: TutorRequest):
    # MVP demo: 실제 구현에서는 req.question으로 RAG 검색 후 LLM 답변 생성
    result = mock_ai["tutor_example"].copy()
    result["question"] = req.question
    return result

@app.get("/api/review/questions")
def get_questions(user_id: int = 1):
    return mock_questions

@app.get("/api/review/flashcards")
def get_flashcards(user_id: int = 1):
    return {"user_id": user_id, "flashcards": mock_ai["flashcards"]}

@app.post("/api/answers/submit")
def submit_answer(req: AnswerSubmitRequest):
    target = None
    for q in mock_questions["questions"]:
        if q["question_id"] == req.question_id:
            target = q
            break

    if target is None:
        return {"error": "question_not_found"}

    is_correct = req.selected_answer == target["answer"]
    return {
        "user_id": req.user_id,
        "question_id": req.question_id,
        "selected_answer": req.selected_answer,
        "correct_answer": target["answer"],
        "is_correct": is_correct,
        "concept": target["concept"],
        "elapsed_time": req.elapsed_time,
        "explanation": target["explanation"],
        "source": target["source"]
    }

@app.get("/api/wrong-notes")
def get_wrong_notes(user_id: int = 1):
    return {"user_id": user_id, "wrong_notes": mock_analytics["wrong_notes"]}

@app.get("/api/weak-concepts")
def get_weak_concepts(user_id: int = 1):
    return {"user_id": user_id, "weak_concepts": mock_analytics["weak_concepts"]}

@app.get("/api/study-plan")
def get_study_plan(user_id: int = 1):
    return {"user_id": user_id, "study_plan": mock_analytics["study_plan"]}

@app.get("/api/dashboard/summary")
def get_dashboard_summary(user_id: int = 1):
    return mock_analytics["dashboard_summary"]

# Serve frontend at /
app.mount("/", StaticFiles(directory=BASE_DIR / "frontend", html=True), name="frontend")

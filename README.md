# LectureMate AI Connected FastAPI Demo

HTML UI와 FastAPI 백엔드를 연결한 데모입니다.

## 실행 방법

```bash
pip install fastapi uvicorn pydantic
uvicorn app:app --reload
```

브라우저에서 아래 주소로 접속합니다.

```text
http://127.0.0.1:8000
```


## 현재 상태

현재는 빠른 시연을 위해 mock data를 API가 반환합니다.
추후 mock data를 실제 RAG 검색 / LLM 생성 / DB 저장 결과로 교체예정

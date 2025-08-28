<<<<<<< HEAD
# smart-attendance-api
=======
# Smart Attendance API (Face Embeddings + Cosine Similarity)

Minimal, production-ready FastAPI backend that stores **face embeddings** (not images) and matches incoming faces to enrolled students by **cosine similarity**.

## Project layout
```
smart-attendance-api/
├─ app.py
├─ requirements.txt
├─ .env.example
├─ Dockerfile
└─ README.md
```

## 1) Run locally

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env   # On Windows: copy .env.example .env
# Edit .env (set API_KEY, optionally change DB URL / threshold)

uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open: `http://localhost:8000/docs` or hit `/health`.

### Health check
```bash
curl http://localhost:8000/health
```

## 2) Enroll & Recognize

**Enroll** (stores normalized embedding against a roll number):
```bash
curl -X POST http://localhost:8000/enroll \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"imageBase64":"<data-url-or-b64>", "roll":"23CSE001", "name":"Alice", "cls":"CSE-3A"}'
```

**Recognize** (returns top matches above threshold):
```bash
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"imageBase64":"<data-url-or-b64>", "cls":"CSE-3A"}'
```

Set `SIM_THRESHOLD` in `.env` (start at `0.7`). Increase if you see false accepts; decrease if false rejects.

## 3) Docker

```bash
docker build -t smart-attendance-api .
docker run -p 8000:8000 --env-file .env smart-attendance-api
```

## 4) Deploy notes

- Use Postgres in production (set `DATABASE_URL` accordingly).
- Lock CORS via `ALLOWED_ORIGINS` to your frontend origin.
- Keep a strong `API_KEY`; rotate periodically.
- Add rate limiting / logging at the gateway.
- Backup your DB regularly.
- For smaller images / faster CPU inference, consider switching to **insightface + onnxruntime** later.


import base64
import io
import os
from datetime import datetime
from typing import List, Optional, Tuple
import sqlite3

import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import (create_engine, Column, Integer, String, Date, Time,
                        Float, Text, DateTime)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

# ML
from deepface import DeepFace
import cv2

from fastapi import Depends, Security

from fastapi.security.api_key import APIKeyHeader

import logging
logging.basicConfig(level=logging.DEBUG)


API_KEY = "dafea3cd-45e5-47ea-9e3c-c2bd5d61de12"
api_key_header = APIKeyHeader(name="x-api-key", auto_error=True)


# Add this near the top of your app.py
def api_key_auth(x_api_key: str = Security(api_key_header)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./attendance.db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Facenet512")
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.7"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# ---------- DB ----------
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True)
    roll = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(128), nullable=False)
    cls = Column(String(64), nullable=True)
    embedding = Column(Text, nullable=True)          # stored as CSV of floats
    created_at = Column(DateTime, default=datetime.utcnow)


class Class(Base):
    __tablename__ = "classes"
    code = Column(String(64), primary_key=True)
    title = Column(String(128))
    teacher = Column(String(128))


class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(String(10), index=True)            # "YYYY-MM-DD"
    cls = Column(String(64), index=True)
    roll = Column(String(64), index=True)
    name = Column(String(128))
    status = Column(String(16))                      # Present/Absent
    time = Column(String(8))                         # "HH:MM:SS"


Base.metadata.create_all(bind=engine)

# ---------- FastAPI ----------
app = FastAPI(title="Smart Attendance API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == "*" else [ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Security ----------
def require_api_key(authorization: Optional[str] = Header(None)):
    """Expect: Authorization: Bearer <API_KEY>"""
    if not API_KEY:
        return  # allow if not set (dev)
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# ---------- DB session ----------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- Schemas ----------
class EnrollReq(BaseModel):
    imageBase64: str
    roll: str
    name: str
    cls: Optional[str] = None

class EnrollResp(BaseModel):
    ok: bool
    faceId: Optional[str] = None  # compat with your frontend

class RecognizeReq(BaseModel):
    imageBase64: str
    cls: Optional[str] = None

class RecognizeItem(BaseModel):
    roll: Optional[str]
    name: str
    confidence: float

class RecognizeResp(BaseModel):
    results: List[RecognizeItem]

# ---------- ML helpers ----------
_model = None
def get_model():
    global _model
    if _model is None:
        # Build once
        _model = DeepFace.build_model(EMBEDDING_MODEL)
    return _model

def b64_to_bgr(image_b64: str) -> np.ndarray:
    """Decode data URL / base64 -> OpenCV BGR image."""
    if "," in image_b64:  # handle data URL prefix
        image_b64 = image_b64.split(",", 1)[1]
    data = base64.b64decode(image_b64)
    img_array = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Invalid image")
    return bgr

def get_embedding(image_b64: str) -> Optional[np.ndarray]:
    """Returns a single face embedding (np.array) or None if no face."""
    img = b64_to_bgr(image_b64)
    reps = DeepFace.represent(
        img_path=img,
        model_name=EMBEDDING_MODEL,
        detector_backend="opencv",
        enforce_detection=False
    )
    if not reps:
        return None
    # If multiple faces, take the largest box
    best = max(reps, key=lambda r: (r.get("facial_area", {}).get("w", 0) * r.get("facial_area", {}).get("h", 0)))
    vec = np.array(best["embedding"], dtype=np.float32)
    # Normalize for cosine similarity
    norm = np.linalg.norm(vec) + 1e-10
    return vec / norm

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def parse_embedding(s: str) -> np.ndarray:
    arr = np.fromstring(s, sep=",", dtype=np.float32)
    return arr

def emb_to_text(vec: np.ndarray) -> str:
    return ",".join([f"{x:.8f}" for x in vec.tolist()])

# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"ok": True, "model": EMBEDDING_MODEL, "threshold": SIM_THRESHOLD}

@app.post("/enroll", response_model=EnrollResp, dependencies=[Depends(require_api_key)])
def enroll(req: EnrollReq, db: Session = Depends(get_db)):
    """Create or update a student's face embedding."""
    _ = get_model()
    emb = get_embedding(req.imageBase64)
    if emb is None:
        raise HTTPException(status_code=400, detail="No face detected")

    student = db.query(Student).filter(Student.roll == req.roll).first()
    if student is None:
        student = Student(
            roll=req.roll.strip(),
            name=req.name.strip(),
            cls=(req.cls or "").strip(),
            embedding=emb_to_text(emb)
        )
        db.add(student)
    else:
        if req.name: student.name = req.name.strip()
        if req.cls is not None: student.cls = (req.cls or "").strip()
        student.embedding = emb_to_text(emb)

    db.commit()
    return EnrollResp(ok=True, faceId=student.roll)

@app.post("/recognize", response_model=RecognizeResp, dependencies=[Depends(require_api_key)])
def recognize(req: RecognizeReq, db: Session = Depends(get_db)):
    """Match one (largest) face in the image against enrolled students."""
    probe = get_embedding(req.imageBase64)
    if probe is None:
        return RecognizeResp(results=[])

    q = db.query(Student)
    if req.cls:
        q = q.filter(Student.cls == req.cls)

    candidates = q.all()
    best: List[Tuple[float, Student]] = []
    for s in candidates:
        if not s.embedding:
            continue
        emb = parse_embedding(s.embedding)
        sim = cosine_sim(probe, emb)
        best.append((sim, s))

    best.sort(key=lambda x: x[0], reverse=True)
    top = best[:5]

    results: List[RecognizeItem] = []
    for sim, s in top:
        if sim >= SIM_THRESHOLD:
            results.append(RecognizeItem(roll=s.roll, name=s.name, confidence=round(sim, 4)))

    return RecognizeResp(results=results)
import cv2
from fastapi.responses import JSONResponse


@app.post("/camera/recognize")
async def camera_recognize(
    threshold: float = 0.7,
    api_key: str = Depends(api_key_auth)
):
    try:
        import cv2
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            return {"status": "error", "message": "Could not access camera"}

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return {"status": "error", "message": "Could not capture image"}

        # Save captured frame
        img_path = "captured_frame.jpg"
        cv2.imwrite(img_path, frame)

        # Use DeepFace to recognize
        from deepface import DeepFace
        result = DeepFace.find(img_path, db_path="face_db", model_name="Facenet", distance_metric="cosine")

        if len(result) > 0 and not result[0].empty:
            identity = result[0].iloc[0]['identity']
            return {"status": "success", "recognized": identity}
        else:
            return {"status": "success", "recognized": None}

    except Exception as e:
        return {"status": "error", "message": str(e)}



# ---- Optional: simple CRUD for students/classes/attendance (for admin tools) ----

class StudentIn(BaseModel):
    roll: str
    name: str
    cls: Optional[str] = None

@app.get("/students", dependencies=[Depends(require_api_key)])
def list_students(db: Session = Depends(get_db)):
    rows = db.query(Student).all()
    return [{"roll": s.roll, "name": s.name, "cls": s.cls, "hasEmbedding": bool(s.embedding)} for s in rows]

@app.post("/students", dependencies=[Depends(require_api_key)])
def create_student(body: StudentIn, db: Session = Depends(get_db)):
    if db.query(Student).filter(Student.roll == body.roll).first():
        raise HTTPException(400, "Roll already exists")
    s = Student(roll=body.roll, name=body.name, cls=body.cls or "")
    db.add(s); db.commit()
    return {"ok": True}

class ClassIn(BaseModel):
    code: str
    title: str
    teacher: str

@app.get("/classes", dependencies=[Depends(require_api_key)])
def list_classes(db: Session = Depends(get_db)):
    rows = db.query(Class).all()
    return [{"code": c.code, "title": c.title, "teacher": c.teacher} for c in rows]

@app.post("/classes", dependencies=[Depends(require_api_key)])
def create_class(body: ClassIn, db: Session = Depends(get_db)):
    if db.query(Class).filter(Class.code == body.code).first():
        raise HTTPException(400, "Code exists")
    c = Class(code=body.code, title=body.title, teacher=body.teacher)
    db.add(c); db.commit()
    return {"ok": True}

class AttendanceIn(BaseModel):
    date: str
    cls: str
    roll: str
    name: str
    status: str
    time: str

@app.post("/attendance", dependencies=[Depends(require_api_key)])
def add_attendance(items: List[AttendanceIn], db: Session = Depends(get_db)):
    for it in items:
        db.add(Attendance(date=it.date, cls=it.cls, roll=it.roll, name=it.name, status=it.status, time=it.time))
    db.commit()
    return {"ok": True}

DB_FILE = "attendance.db"
if not os.path.exists(DB_FILE):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    cls TEXT,
                    roll TEXT,
                    name TEXT,
                    status TEXT,
                    time TEXT
                )''')
    conn.commit()
    conn.close()

# Pydantic model
class Attendance(BaseModel):
    date: str
    cls: str
    roll: str
    name: str
    status: str
    time: str

# POST /attendance/mark
@app.post("/attendance/mark")
def mark_attendance(record: Attendance, api_key: str = Depends(api_key_auth)):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO attendance (date, cls, roll, name, status, time) VALUES (?, ?, ?, ?, ?, ?)",
              (record.date, record.cls, record.roll, record.name, record.status, record.time))
    conn.commit()
    conn.close()
    return {"ok": True, "message": f"Attendance marked for {record.name}"}

# GET /attendance/list
@app.get("/attendance/list")
def list_attendance(cls: Optional[str] = None, date: Optional[str] = None, api_key: str = Depends(api_key_auth)):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    query = "SELECT date, cls, roll, name, status, time FROM attendance WHERE 1=1"
    params = []
    
    if cls:
        query += " AND cls = ?"
        params.append(cls)
    if date:
        query += " AND date = ?"
        params.append(date)
    
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    
    return {"ok": True, "records": rows}

from fastapi.responses import StreamingResponse
import io, csv, sqlite3

from fastapi.responses import StreamingResponse
import io
import csv

@app.get("/attendance/export")
def export_attendance(cls: str, date: str):
    try:
        conn = sqlite3.connect("attendance.db")
        cur = conn.cursor()
        cur.execute(
            "SELECT date, cls, roll, name, status, time FROM attendance WHERE cls=? AND date=?",
            (cls, date),
        )
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return {"ok": False, "message": "No records found"}

        # Write CSV into memory
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Date", "Class", "Roll No", "Name", "Status", "Time"])
        writer.writerows(rows)

        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=attendance_{cls}_{date}.csv"},
        )

    except Exception as e:
        # Debug log
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}

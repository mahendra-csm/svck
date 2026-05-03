from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, List, Literal, Optional
from datetime import date, datetime, timedelta, timezone
import bcrypt
import base64
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import json
import secrets

from google.cloud.exceptions import Conflict

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Rate limiter (keyed by remote IP)
limiter = Limiter(key_func=get_remote_address)

# Runtime configuration
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'development').strip().lower()
JWT_ISSUER = os.environ.get('JWT_ISSUER', 'svck-digital')
JWT_EXPIRE_MINUTES = int(os.environ.get('JWT_EXPIRE_MINUTES', '480'))

DEFAULT_FIREBASE_SERVICE_ACCOUNT = ROOT_DIR / 'firebase-service-account.json'


def load_firebase_credentials() -> credentials.Certificate:
    service_account_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
    if service_account_json:
        return credentials.Certificate(json.loads(service_account_json))

    service_account_path = Path(
        os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH', str(DEFAULT_FIREBASE_SERVICE_ACCOUNT))
    )
    if service_account_path.exists():
        return credentials.Certificate(service_account_path)

    raise RuntimeError(
        'Firebase service account credentials were not found. '
        'Set FIREBASE_SERVICE_ACCOUNT_PATH or FIREBASE_SERVICE_ACCOUNT_JSON.'
    )


# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    firebase_admin.initialize_app(load_firebase_credentials())

# Firestore database
db = firestore.client()

# JWT Configuration (still used for admin)
SECRET_KEY = os.environ.get('JWT_SECRET', 'svck-digital-insecure-default-change-me-NOW')
ALGORITHM = "HS256"

# Campus Geo-fencing Configuration
CAMPUS_LATITUDE = float(os.environ.get('CAMPUS_LATITUDE', 14.459705443779649))
CAMPUS_LONGITUDE = float(os.environ.get('CAMPUS_LONGITUDE', 78.81842145279516))
CAMPUS_RADIUS_METERS = float(os.environ.get('CAMPUS_RADIUS_METERS', 100))

# TESTING MODE - Set to True to disable geo-fencing check (read from env, default False)
TESTING_MODE = os.environ.get('TESTING_MODE', 'false').lower() == 'true'

# Face recognition configuration
SUPPORTED_FACE_DETECTION_MODELS = {"hog", "cnn"}
FACE_DETECTION_MODEL = os.environ.get('FACE_DETECTION_MODEL', 'hog').strip().lower() or 'hog'
if FACE_DETECTION_MODEL not in SUPPORTED_FACE_DETECTION_MODELS:
    FACE_DETECTION_MODEL = 'hog'
FACE_MATCH_TOLERANCE = max(0.3, min(0.8, float(os.environ.get('FACE_MATCH_TOLERANCE', '0.55'))))
FACE_DUPLICATE_TOLERANCE = max(0.3, min(0.8, float(os.environ.get('FACE_DUPLICATE_TOLERANCE', '0.45'))))
FACE_REGISTER_MIN_IMAGES = max(1, int(os.environ.get('FACE_REGISTER_MIN_IMAGES', '5')))
FACE_REGISTER_MIN_ENCODINGS = max(1, int(os.environ.get('FACE_REGISTER_MIN_ENCODINGS', '3')))
FACE_REGISTER_TARGET_ENCODINGS = max(
    FACE_REGISTER_MIN_ENCODINGS,
    int(os.environ.get('FACE_REGISTER_TARGET_ENCODINGS', '5')),
)

# Admin Credentials — must be set in env; fallbacks are intentionally weak so
# validate_runtime_configuration() will raise on startup if not overridden.
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', 'admin@svck.edu.in')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'changeme')
ADMIN_PASSWORD_HASH = os.environ.get('ADMIN_PASSWORD_HASH')

# Assistant configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')

# CORS configuration
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get('ALLOWED_ORIGINS', '*').split(',') if o.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]
ALLOW_CREDENTIALS = "*" not in ALLOWED_ORIGINS

@asynccontextmanager
async def lifespan(_: FastAPI):
    validate_runtime_configuration()
    logger.info("SVCK Digital Backend started with Firebase")
    logger.info(
        "Environment=%s, testing_mode=%s, face_detection_model=%s, "
        "face_match_tolerance=%s, campus_radius=%sm, gemini_model=%s, "
        "admin_email=%s, jwt_secret_len=%d",
        ENVIRONMENT,
        TESTING_MODE,
        FACE_DETECTION_MODEL,
        FACE_MATCH_TOLERANCE,
        CAMPUS_RADIUS_METERS,
        GEMINI_MODEL,
        ADMIN_EMAIL,
        len(SECRET_KEY),
    )
    yield

# Create the main app
app = FastAPI(title="SVCK Digital - Face Recognition Attendance System", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Security
security = HTTPBearer()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        stale_connections: List[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                stale_connections.append(connection)
        for connection in stale_connections:
            self.disconnect(connection)

manager = ConnectionManager()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== MODELS ====================

class StudentRegister(BaseModel):
    name: str
    roll_number: str
    password: str
    regulation: str  # R20, R23
    branch: str  # CSE, ECE, CSE (AI & ML)
    section: str = "A"  # A, B, C, D
    year: int  # 1, 2, 3, 4
    college: str = "SVCK"

class StudentLogin(BaseModel):
    roll_number: str
    password: str

class AdminLogin(BaseModel):
    email: str
    password: str

class FaceRegisterRequest(BaseModel):
    face_images: List[str] = Field(min_length=1, max_length=10)

class AttendanceRequest(BaseModel):
    face_image: str = Field(min_length=1, max_length=2_000_000)  # ~1.5 MB max
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)

class StudentResponse(BaseModel):
    id: str
    name: str
    roll_number: str
    regulation: str
    branch: str
    year: int
    college: str
    face_registered: bool

class AttendanceRecord(BaseModel):
    id: str
    student_id: str
    student_name: str
    roll_number: str
    branch: str
    year: int
    date: str
    time: str
    geo_verified: bool
    created_at: datetime

class UpdateProfileRequest(BaseModel):
    year: Optional[int] = None
    name: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str = Field(min_length=1, max_length=4000)


class StudentAssistantRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: List[ChatMessage] = Field(default_factory=list)


class AdminAssistantContext(BaseModel):
    totalStudents: Optional[int] = None
    todayAttendance: Optional[int] = None
    statistics: Optional[dict[str, Any]] = None
    attendanceRecords: Optional[List[dict[str, Any]]] = None
    students: Optional[List[dict[str, Any]]] = None


class AdminAssistantRequest(StudentAssistantRequest):
    model_config = ConfigDict(populate_by_name=True)
    app_context: Optional[AdminAssistantContext] = Field(default=None, alias='appContext')

# ==================== HELPER FUNCTIONS ====================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

import jwt

def normalize_roll_number(roll_number: str) -> str:
    return roll_number.strip().upper()


def build_student_email(roll_number: str) -> str:
    return f"{normalize_roll_number(roll_number).lower()}@svck.edu.in"


def is_production_environment() -> bool:
    return ENVIRONMENT == 'production'


_INSECURE_SECRETS = {
    'svck-digital-insecure-default-change-me-NOW',
    'svck-digital-secret-key-2025',
    'changeme',
    'secret',
    'password',
}
_INSECURE_PASSWORDS = {'changeme', 'admin@123', 'password', 'admin'}


def uses_weak_default_secret() -> bool:
    return SECRET_KEY in _INSECURE_SECRETS or len(SECRET_KEY) < 32


def uses_default_admin_credentials() -> bool:
    plain_weak = ADMIN_PASSWORD in _INSECURE_PASSWORDS and not ADMIN_PASSWORD_HASH
    return plain_weak


def validate_runtime_configuration() -> None:
    issues: List[str] = []

    if uses_weak_default_secret():
        issues.append('JWT_SECRET must be set to a strong random value with at least 32 characters.')

    if is_production_environment():
        if TESTING_MODE:
            issues.append('TESTING_MODE must be false in production.')
        if uses_default_admin_credentials():
            issues.append('Replace the default admin credentials before running in production.')

    if not ADMIN_EMAIL or '@' not in ADMIN_EMAIL:
        issues.append('ADMIN_EMAIL must be a valid email address.')

    if issues:
        raise RuntimeError('Invalid runtime configuration:\n- ' + '\n- '.join(issues))


def create_token(data: dict, expires_minutes: int = JWT_EXPIRE_MINUTES) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        **data,
        "iat": now,
        "nbf": now,
        "exp": now + timedelta(minutes=expires_minutes),
        "iss": JWT_ISSUER,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], issuer=JWT_ISSUER)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def fs_get_doc(doc_ref):
    """Run Firestore doc get in a thread to avoid blocking the event loop."""
    return await run_in_threadpool(doc_ref.get)

async def fs_set_doc(doc_ref, data: dict):
    """Threadpool wrapper for doc.set"""
    await run_in_threadpool(doc_ref.set, data)

async def fs_create_doc(doc_ref, data: dict):
    """Threadpool wrapper for doc.create"""
    await run_in_threadpool(doc_ref.create, data)

async def fs_update_doc(doc_ref, data: dict):
    """Threadpool wrapper for doc.update"""
    await run_in_threadpool(doc_ref.update, data)

async def fs_delete_doc(doc_ref):
    """Threadpool wrapper for doc.delete"""
    await run_in_threadpool(doc_ref.delete)

async def fs_query(query):
    """Threadpool wrapper for query.get"""
    return await run_in_threadpool(query.get)

async def verify_firebase_token(token: str) -> dict:
    """Verify Firebase ID token (runs in threadpool to avoid blocking the event loop)."""
    try:
        decoded_token = await run_in_threadpool(firebase_auth.verify_id_token, token)
        return decoded_token
    except Exception as e:
        logger.error(f"Firebase token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def authenticate_admin_credentials(email: str, password: str) -> bool:
    normalized_email = email.strip().lower()
    if not secrets.compare_digest(normalized_email, ADMIN_EMAIL.lower()):
        return False
    if ADMIN_PASSWORD_HASH:
        return verify_password(password, ADMIN_PASSWORD_HASH)
    return secrets.compare_digest(password, ADMIN_PASSWORD)


async def get_current_user_from_token(token: str) -> dict:
    # First try Firebase token — verify_id_token may do a network call on first run
    # so we run it in a thread to avoid blocking the event loop.
    try:
        payload = await run_in_threadpool(firebase_auth.verify_id_token, token)
        # Get additional user data from Firestore
        uid = payload['uid']
        student_doc = await fs_get_doc(db.collection('students').document(uid))
        if hasattr(student_doc, 'exists') and student_doc.exists:
            student_data = student_doc.to_dict()
            if student_data:
                return {
                    "id": uid,
                    "uid": uid,
                    "roll_number": student_data.get("roll_number"),
                    "role": "student",
                    **student_data
                }
        raise HTTPException(status_code=404, detail="User not found")
    except HTTPException:
        raise
    except Exception:
        # The token may be an admin JWT instead of a Firebase ID token.
        pass
    # Fallback to JWT for admin
    try:
        payload = decode_token(token)
        return payload
    except HTTPException:
        raise


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return await get_current_user_from_token(credentials.credentials)

async def get_current_student(credentials: HTTPAuthorizationCredentials = Depends(security)):
    payload = await get_current_user(credentials)
    if payload.get('role') != 'student':
        raise HTTPException(status_code=403, detail="Student access required")
    return payload

async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    payload = await get_current_user(credentials)
    if payload.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    return payload


def build_chat_prompt(
    system_prompt: str,
    message: str,
    history: List[ChatMessage],
    context: Optional[dict[str, Any]] = None,
) -> str:
    history_lines: List[str] = []
    for entry in history[-10:]:
        role = 'User' if entry.role == 'user' else 'Assistant'
        history_lines.append(f"{role}: {entry.content.strip()}")

    context_lines: List[str] = []
    if context:
        context_lines.append('Context:')
        context_lines.append(json.dumps(context, default=str, ensure_ascii=True, indent=2))

    prompt_parts = [system_prompt]
    if context_lines:
        prompt_parts.append('\n'.join(context_lines))
    if history_lines:
        prompt_parts.append('Conversation so far:\n' + '\n'.join(history_lines))
    prompt_parts.append(f"User: {message.strip()}")
    prompt_parts.append('Assistant:')
    return '\n\n'.join(prompt_parts)


async def generate_gemini_response(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail='AI assistant is not configured on the server.')

    def _generate() -> str:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        text = getattr(response, 'text', None)
        if not text:
            raise RuntimeError('The AI assistant returned an empty response.')
        return text.strip()

    try:
        return await run_in_threadpool(_generate)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Assistant generation failed: {exc}")
        raise HTTPException(status_code=502, detail='AI assistant request failed.')

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in meters using Haversine formula"""
    R = 6371000  # Earth's radius in meters
    
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    
    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c

def validate_geofence(latitude: float, longitude: float) -> tuple:
    """Validate if location is within campus geo-fence"""
    distance = haversine_distance(CAMPUS_LATITUDE, CAMPUS_LONGITUDE, latitude, longitude)
    is_valid = distance <= CAMPUS_RADIUS_METERS
    return is_valid, distance

def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 image to numpy array for face recognition"""
    import warnings
    warnings.filterwarnings('ignore')
    from PIL import Image, ImageOps
    import io as iomodule
    
    # Remove data URL prefix if present
    if 'base64,' in base64_str:
        base64_str = base64_str.split('base64,')[1]
    
    # Add padding if necessary
    padding = 4 - len(base64_str) % 4
    if padding != 4:
        base64_str += '=' * padding
    
    image_data = base64.b64decode(base64_str)
    image = Image.open(iomodule.BytesIO(image_data))
    
    # Fix EXIF orientation (important for mobile camera photos)
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to optimal size for face detection (not too big, not too small)
    target_size = 800
    width, height = image.size
    if max(width, height) > target_size:
        ratio = target_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    elif max(width, height) < 400:
        # Upscale small images so face detector can find them
        ratio = 400 / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    logger.info(f"Decoded image size: {image.size}")
    return np.array(image)

def get_face_encoding(image_array: np.ndarray, model: Optional[str] = None, num_jitters: int = 2):
    """Extract face encoding with a CPU-friendly default and a safe HOG fallback."""
    import warnings
    warnings.filterwarnings('ignore')
    import face_recognition
    from PIL import Image

    original_shape = image_array.shape
    logger.info(f"Processing image array shape: {original_shape}")

    preferred_model = (model or FACE_DETECTION_MODEL).strip().lower()
    models_to_try = [preferred_model]
    if preferred_model != "hog":
        models_to_try.append("hog")

    for active_model in models_to_try:
        working_image = image_array
        try:
            # First attempt: standard detection
            face_locations = face_recognition.face_locations(working_image, model=active_model)
            logger.info(f"{active_model.upper()} model found {len(face_locations)} face(s)")

            # Second attempt: upsample to find smaller faces
            if not face_locations:
                face_locations = face_recognition.face_locations(
                    working_image,
                    number_of_times_to_upsample=2,
                    model=active_model,
                )
                logger.info(f"{active_model.upper()} upsample=2 found {len(face_locations)} face(s)")

            # Third attempt: try rotated images (handles sideways camera shots)
            if not face_locations:
                pil_img = Image.fromarray(working_image)
                for angle in [90, 270, 180]:
                    rotated = pil_img.rotate(angle, expand=True)
                    rotated_array = np.array(rotated)
                    face_locations = face_recognition.face_locations(rotated_array, model=active_model)
                    if face_locations:
                        logger.info(f"Found face after rotating {angle} degrees with {active_model.upper()}")
                        working_image = rotated_array
                        break

            # Fourth attempt: try at a fixed 640px size
            if not face_locations:
                pil_img = Image.fromarray(working_image)
                max_dim = max(pil_img.width, pil_img.height)
                if max_dim != 640:
                    scale = 640 / max_dim
                    resized = pil_img.resize(
                        (int(pil_img.width * scale), int(pil_img.height * scale)),
                        Image.Resampling.LANCZOS
                    )
                    resized_array = np.array(resized)
                    face_locations = face_recognition.face_locations(
                        resized_array,
                        number_of_times_to_upsample=1,
                        model=active_model,
                    )
                    if face_locations:
                        logger.info(f"Resized image detection found {len(face_locations)} face(s) with {active_model.upper()}")
                        working_image = resized_array

            if not face_locations:
                logger.info(f"No faces detected with {active_model.upper()} model")
                continue

            # Use the largest face detected (closest to camera)
            face_locations = [max(face_locations, key=lambda loc: (loc[2]-loc[0]) * (loc[1]-loc[3]))]

            face_encodings = face_recognition.face_encodings(
                working_image, face_locations, num_jitters=num_jitters
            )
            logger.info(f"Generated {len(face_encodings)} encoding(s) with {active_model.upper()}, jitters={num_jitters}")

            if face_encodings:
                return face_encodings[0].tolist()
        except Exception as exc:
            logger.warning(f"Face detection failed with {active_model.upper()} model: {exc}")

    logger.warning("No faces detected with any configured method")
    return None

def compare_faces(known_encodings: List[List[float]], face_encoding: List[float], tolerance: float = FACE_MATCH_TOLERANCE) -> bool:
    """Compare face encoding with known encodings."""
    import warnings
    warnings.filterwarnings('ignore')
    import face_recognition

    if not known_encodings:
        return False

    known_np = [np.array(enc) for enc in known_encodings]
    face_np = np.array(face_encoding)

    logger.info(f"Comparing face with {len(known_np)} known encoding(s)")
    matches = face_recognition.compare_faces(known_np, face_np, tolerance=tolerance)
    logger.info(f"Match results: {matches}")
    return True in matches


def find_best_face_match(known_encodings: List[List[float]], face_encoding: List[float], tolerance: float = FACE_MATCH_TOLERANCE):
    """Find the best matching face using Euclidean distance. Returns (is_match, min_distance)."""
    import warnings
    warnings.filterwarnings('ignore')
    import face_recognition

    if not known_encodings:
        return False, 1.0

    known_np = [np.array(enc) for enc in known_encodings]
    face_np = np.array(face_encoding)

    distances = face_recognition.face_distance(known_np, face_np)
    min_distance = float(min(distances))
    is_match = min_distance <= tolerance

    logger.info(
        f"Face distance check: min_distance={min_distance:.4f}, "
        f"tolerance={tolerance}, is_match={is_match}, "
        f"checked_against={len(known_np)} encodings"
    )
    return is_match, min_distance


def _process_face_image_sync(image_b64: str, num_jitters: int = 3) -> Optional[List[float]]:
    """Synchronous helper: decode + CLAHE + encode a single face image.
    Always call via run_in_threadpool — never call directly in async context."""
    import cv2
    try:
        image_array = decode_base64_image(image_b64)
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) gives better
        # contrast normalisation than plain equalizeHist, especially in uneven lighting.
        img_yuv = cv2.cvtColor(image_array, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        image_array = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return get_face_encoding(image_array, num_jitters=num_jitters)
    except Exception as exc:
        logger.error(f"_process_face_image_sync failed: {exc}")
        return None


def _check_duplicates_sync(face_docs_data: list, new_encodings: list, tolerance: float):
    """Synchronous helper: check if any of new_encodings match existing face records.
    Returns the student_id of the duplicate or None. Always call via run_in_threadpool."""
    for face_data in face_docs_data:
        existing_encs = face_data.get("encodings", [])
        if not existing_encs:
            continue
        for new_enc in new_encodings:
            if compare_faces(existing_encs, new_enc, tolerance=tolerance):
                return face_data.get("student_id")
    return None

# ==================== ROUTES ====================

@api_router.get("/")
async def root():
    return {"message": "SVCK Digital - Face Recognition Attendance System (Firebase)", "status": "active"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat(), "database": "Firebase Firestore"}

# ==================== STUDENT AUTH (Firebase) ====================

@api_router.post("/student/register")
@limiter.limit("10/minute")
async def register_student(request: Request, data: StudentRegister):
    """Register a new student with Firebase Authentication"""
    try:
        roll_number = normalize_roll_number(data.roll_number)

        # Check if roll number already exists in Firestore
        existing_query = await fs_query(db.collection('students').where('roll_number', '==', roll_number).limit(1))
        if len(list(existing_query)) > 0:
            raise HTTPException(status_code=400, detail="Roll number already registered")
        
        # Create Firebase Auth user with email (roll_number@svck.edu.in)
        email = build_student_email(roll_number)
        try:
            firebase_user = await run_in_threadpool(
                firebase_auth.create_user,
                email=email,
                password=data.password,
                display_name=data.name
            )
        except firebase_admin.exceptions.AlreadyExistsError:
            raise HTTPException(status_code=400, detail="Roll number already registered")

        # Store additional student data in Firestore
        student_data = {
            "id": firebase_user.uid,
            "name": data.name,
            "roll_number": roll_number,
            "email": email,
            "regulation": data.regulation,
            "branch": data.branch,
            "section": data.section,
            "year": data.year,
            "college": data.college,
            "face_registered": False,
            "created_at": datetime.now(timezone.utc)
        }

        await fs_set_doc(db.collection('students').document(firebase_user.uid), student_data)

        # Create custom token for immediate login (also a network call)
        custom_token = await run_in_threadpool(firebase_auth.create_custom_token, firebase_user.uid)
        
        return {
            "message": "Registration successful",
            "custom_token": custom_token.decode('utf-8') if isinstance(custom_token, bytes) else custom_token,
            "uid": firebase_user.uid,
            "student": {
                "id": firebase_user.uid,
                "name": data.name,
                "roll_number": roll_number,
                "regulation": data.regulation,
                "branch": data.branch,
                "section": data.section,
                "year": data.year,
                "college": data.college,
                "face_registered": False
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed. Please try again later.")

@api_router.post("/student/login")
async def login_student(data: StudentLogin):
    """
    The mobile app authenticates directly against Firebase using the derived email address.
    This endpoint exists only as a helper for legacy clients and intentionally does not return
    student profile data.
    """
    return {
        "message": "Use Firebase client SDK for authentication",
        "email": build_student_email(data.roll_number),
    }

@api_router.get("/student/profile")
async def get_student_profile(user: dict = Depends(get_current_student)):
    student_doc = await fs_get_doc(db.collection('students').document(user["id"]))
    if not hasattr(student_doc, 'exists') or not student_doc.exists:
        raise HTTPException(status_code=404, detail="Student not found")
    student = student_doc.to_dict() or {}
    # Get last attendance (without order_by to avoid index requirement)
    attendance_query = await fs_query(db.collection('attendance').where('student_id', '==', user["id"]))
    attendance_list = [doc.to_dict() for doc in attendance_query if doc.to_dict()]
    # Sort in Python instead
    if attendance_list:
        attendance_list.sort(key=lambda x: x.get('created_at'), reverse=True)
        last_attendance = attendance_list[0]
    else:
        last_attendance = None
    _ca = last_attendance.get("created_at") if last_attendance else None
    if _ca is not None and not hasattr(_ca, 'isoformat'):
        _ca = None  # discard unparseable value
    return {
        "id": getattr(student_doc, 'id', None),
        "name": student.get("name"),
        "roll_number": student.get("roll_number"),
        "regulation": student.get("regulation"),
        "branch": student.get("branch"),
        "section": student.get("section", "A"),
        "year": student.get("year"),
        "college": student.get("college"),
        "face_registered": student.get("face_registered", False),
        "last_attendance": _ca.isoformat() if _ca else None
    }

@api_router.put("/student/profile")
async def update_student_profile(data: UpdateProfileRequest, user: dict = Depends(get_current_student)):
    student_ref = db.collection('students').document(user["id"])
    student_doc = await fs_get_doc(student_ref)
    if not hasattr(student_doc, 'exists') or not student_doc.exists:
        raise HTTPException(status_code=404, detail="Student not found")
    update_data = {}
    if data.year is not None and data.year in [1, 2, 3, 4]:
        update_data["year"] = data.year
    if data.name is not None and len(data.name.strip()) > 0:
        update_data["name"] = data.name.strip()
    if not update_data:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    await fs_update_doc(student_ref, update_data)
    # Return updated profile
    updated_doc = await fs_get_doc(student_ref)
    updated_student = updated_doc.to_dict() or {}
    return {
        "message": "Profile updated successfully",
        "student": {
            "id": getattr(updated_doc, 'id', None),
            "name": updated_student.get("name"),
            "roll_number": updated_student.get("roll_number"),
            "regulation": updated_student.get("regulation"),
            "branch": updated_student.get("branch"),
            "year": updated_student.get("year"),
            "college": updated_student.get("college"),
            "face_registered": updated_student.get("face_registered", False)
        }
    }

# ==================== FACE REGISTRATION ====================

@api_router.post("/student/register-face")
@limiter.limit("5/minute")
async def register_face(request: Request, data: FaceRegisterRequest, user: dict = Depends(get_current_student)):
    if len(data.face_images) < FACE_REGISTER_MIN_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Please provide at least {FACE_REGISTER_MIN_IMAGES} clear face images "
                "(different angles, lighting, expressions) for best accuracy."
            ),
        )

    # Process all images in parallel via threadpool - uses CLAHE + higher jitters for quality
    logger.info(f"Processing {len(data.face_images)} face images for registration (user={user['id']})")
    encoding_tasks = [
        run_in_threadpool(_process_face_image_sync, img, 3)
        for img in data.face_images
    ]
    results = await asyncio.gather(*encoding_tasks, return_exceptions=True)

    encodings = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Error processing image {i}: {result}")
            continue
        if result is not None:
            encodings.append(result)
        if len(encodings) >= FACE_REGISTER_TARGET_ENCODINGS:
            logger.info("Collected enough face encodings for registration, stopping early")
            break

    logger.info(f"Successfully extracted {len(encodings)}/{len(data.face_images)} face encodings")

    if len(encodings) < FACE_REGISTER_MIN_ENCODINGS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Face not clearly detected. We found a face in only {len(encodings)} of "
                f"{len(data.face_images)} images (need at least {FACE_REGISTER_MIN_ENCODINGS}). "
                "Tips: ensure good lighting, face the camera directly, remove glasses/mask, "
                "and avoid very dark or very bright backgrounds."
            ),
        )

    # CHECK FOR DUPLICATE FACE — run the CPU-heavy comparison off the event loop
    face_docs = await fs_query(db.collection('face_encodings'))
    other_face_data = [
        {**(doc.to_dict() or {}), "student_id": doc.id}
        for doc in face_docs
        if doc.id != user["id"]
    ]

    if other_face_data:
        dup_student_id = await run_in_threadpool(
            _check_duplicates_sync, other_face_data, encodings, FACE_DUPLICATE_TOLERANCE
        )
        if dup_student_id:
            existing_student = await fs_get_doc(db.collection('students').document(dup_student_id))
            existing_roll = "unknown"
            if hasattr(existing_student, 'exists') and existing_student.exists:
                existing_data = existing_student.to_dict() or {}
                existing_roll = existing_data.get("roll_number", "unknown")
            raise HTTPException(
                status_code=400,
                detail=(
                    f"This face is already registered with another account (Roll: {existing_roll}). "
                    "Each person can only have one account."
                ),
            )

    # Store face encodings in Firestore
    await fs_set_doc(db.collection('face_encodings').document(user["id"]), {
        "student_id": user["id"],
        "encodings": encodings,
        "created_at": datetime.now(timezone.utc)
    })

    # Update student face_registered status
    await fs_update_doc(db.collection('students').document(user["id"]), {
        "face_registered": True
    })

    logger.info(f"Face registration complete for user={user['id']}, encodings={len(encodings)}")
    return {"message": "Face registration successful", "encodings_count": len(encodings)}

# ==================== ATTENDANCE ====================

@api_router.post("/student/mark-attendance")
@limiter.limit("10/minute")
async def mark_attendance(request: Request, data: AttendanceRequest, user: dict = Depends(get_current_student)):
    # STEP 1: Validate geo-fence (BACKEND VALIDATION - skip in TESTING_MODE)
    is_valid_location, distance = validate_geofence(data.latitude, data.longitude)
    
    if not TESTING_MODE and not is_valid_location:
        raise HTTPException(
            status_code=400, 
            detail=f"You are {distance:.0f} meters away from campus. Attendance can only be marked within {CAMPUS_RADIUS_METERS} meters of campus."
        )
    
    if TESTING_MODE:
        logger.info(f"TESTING_MODE: Geo-fence check bypassed. Distance: {distance:.0f}m")
    
    # STEP 2: Determine today's attendance document id
    today = date.today().isoformat()
    
    # STEP 3: Get student info
    student_doc = await fs_get_doc(db.collection('students').document(user["id"]))
    if not hasattr(student_doc, 'exists') or not student_doc.exists:
        raise HTTPException(status_code=404, detail="Student not found")
    
    student = student_doc.to_dict()
    
    if not student.get("face_registered"):
        raise HTTPException(status_code=400, detail="Please register your face before marking attendance")
    
    # STEP 4: Get stored face encodings
    face_doc = await fs_get_doc(db.collection('face_encodings').document(user["id"]))
    if not face_doc.exists:
        raise HTTPException(status_code=400, detail="Face data not found. Please register your face again.")
    
    face_data = face_doc.to_dict()
    
    # STEP 5: Face Recognition — run CPU-heavy work in threadpool
    try:
        current_encoding = await run_in_threadpool(_process_face_image_sync, data.face_image, 2)

        if not current_encoding:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No face detected in the image. Please ensure: "
                    "good lighting on your face, face the camera directly, "
                    "remove glasses/mask if possible, and avoid backlighting."
                ),
            )

        stored_encodings = face_data.get("encodings", [])
        if not stored_encodings:
            raise HTTPException(
                status_code=400,
                detail="No stored face encodings found. Please re-register your face."
            )

        # Distance-based matching: find minimum Euclidean distance across all stored encodings
        is_match, min_distance = await run_in_threadpool(
            find_best_face_match, stored_encodings, current_encoding, FACE_MATCH_TOLERANCE
        )

        if not is_match:
            logger.info(
                f"Attendance face match FAILED for user={user['id']}: "
                f"best_distance={min_distance:.4f}, tolerance={FACE_MATCH_TOLERANCE}"
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    "Face verification failed. Tips: look directly at the camera, "
                    "ensure good lighting, and hold your phone steady. "
                    "If this keeps failing, re-register your face from the Profile tab."
                ),
            )

        logger.info(
            f"Attendance face match OK for user={user['id']}: "
            f"best_distance={min_distance:.4f}, tolerance={FACE_MATCH_TOLERANCE}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face recognition error: {e}")
        raise HTTPException(status_code=500, detail="Face recognition failed. Please try again.")
    
    # STEP 6: Mark attendance
    attendance_id = f"{user['id']}_{today}"
    now = datetime.now(timezone.utc)
    attendance_record = {
        "id": attendance_id,
        "student_id": user["id"],
        "student_name": student["name"],
        "roll_number": student["roll_number"],
        "branch": student["branch"],
        "year": student["year"],
        "date": today,
        "time": now.strftime("%H:%M:%S"),
        "geo_verified": is_valid_location,
        "latitude": data.latitude,
        "longitude": data.longitude,
        "distance_from_campus": distance,
        "created_at": now
    }

    try:
        await fs_create_doc(db.collection('attendance').document(attendance_id), attendance_record)
    except Conflict:
        raise HTTPException(status_code=400, detail="Attendance already marked for today")
    
    # Broadcast to admin dashboard
    await manager.broadcast({
        "type": "new_attendance",
        "data": {
            "id": attendance_id,
            "student_name": student["name"],
            "roll_number": student["roll_number"],
            "branch": student["branch"],
            "year": student["year"],
            "time": now.strftime("%H:%M:%S"),
            "date": today
        }
    })
    
    return {
        "message": "Attendance marked successfully",
        "attendance": {
            "id": attendance_id,
            "date": today,
            "time": now.strftime("%H:%M:%S"),
            "geo_verified": is_valid_location
        }
    }

@api_router.get("/student/attendance-history")
async def get_attendance_history(user: dict = Depends(get_current_student)):
    # Fetch without order_by to avoid index requirement, then sort in Python
    records_query = await fs_query(db.collection('attendance').where('student_id', '==', user["id"]))
    records = [doc.to_dict() for doc in records_query]
    # Sort by created_at descending in Python
    records.sort(key=lambda x: x.get('created_at'), reverse=True)
    records = records[:100]  # Limit to 100
    
    # Calculate monthly statistics
    total_records = len(records)
    geo_verified_count = sum(1 for r in records if r.get("geo_verified", False))
    
    return {
        "records": [{
            "id": r["id"],
            "date": r["date"],
            "time": r["time"],
            "geo_verified": r.get("geo_verified", False)
        } for r in records],
        "statistics": {
            "total_attendance": total_records,
            "geo_verified_percentage": (geo_verified_count / total_records * 100) if total_records > 0 else 0
        }
    }

# ==================== ADMIN AUTH ====================

@api_router.post("/admin/login")
@limiter.limit("10/minute")
async def login_admin(request: Request, data: AdminLogin):
    if not authenticate_admin_credentials(data.email, data.password):
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
    
    token = create_token({
        "email": ADMIN_EMAIL,
        "role": "admin"
    })
    
    return {
        "message": "Admin login successful",
        "token": token,
        "admin": {
            "email": ADMIN_EMAIL,
            "role": "admin"
        }
    }

# ==================== ADMIN DASHBOARD ====================

@api_router.get("/admin/departments")
async def get_departments(user: dict = Depends(get_current_admin)):
    return {
        "departments": ["CSE", "ECE", "CSE (AI & ML)"],
        "years": [1, 2, 3, 4]
    }

@api_router.get("/admin/students")
async def get_students(
    branch: Optional[str] = None,
    year: Optional[int] = None,
    section: Optional[str] = None,
    user: dict = Depends(get_current_admin)
):
    query = db.collection('students')
    
    if branch:
        query = query.where('branch', '==', branch)
    if year:
        query = query.where('year', '==', year)
    if section:
        query = query.where('section', '==', section)
    
    students = await fs_query(query)
    
    result = []
    for s in students:
        s_dict = s.to_dict() or {}
        result.append({
            "id": getattr(s, 'id', None),
            "name": s_dict.get("name"),
            "roll_number": s_dict.get("roll_number"),
            "branch": s_dict.get("branch"),
            "section": s_dict.get("section", "A"),
            "year": s_dict.get("year"),
            "regulation": s_dict.get("regulation"),
            "face_registered": s_dict.get("face_registered", False)
        })
    
    # Sort by roll_number (safe against None values)
    result.sort(key=lambda x: x.get("roll_number") or "")
    
    return result

@api_router.delete("/admin/student/{student_id}")
async def delete_student(student_id: str, user: dict = Depends(get_current_admin)):
    """Completely remove a student: auth account, face encodings, attendance history."""
    student_ref = db.collection('students').document(student_id)
    student_doc = await fs_get_doc(student_ref)
    if not student_doc.exists:
        raise HTTPException(status_code=404, detail="Student not found")

    # Delete attendance history
    attendance_docs = await fs_query(db.collection('attendance').where('student_id', '==', student_id))
    deleted_attendance = 0
    for doc in attendance_docs:
        await fs_delete_doc(doc.reference)
        deleted_attendance += 1

    # Delete face encodings
    await fs_delete_doc(db.collection('face_encodings').document(student_id))

    # Delete student profile
    await fs_delete_doc(student_ref)

    # Delete Firebase auth user (ignore if already removed)
    # Run in threadpool so we don't block the async event loop
    try:
        await run_in_threadpool(firebase_auth.delete_user, student_id)
    except firebase_auth.UserNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Failed to delete Firebase user {student_id}: {e}")

    return {
        "message": "Student deleted successfully",
        "deleted_attendance_records": deleted_attendance
    }

@api_router.get("/admin/attendance")
async def get_all_attendance(
    branch: Optional[str] = None,
    year: Optional[int] = None,
    date_filter: Optional[str] = None,
    user: dict = Depends(get_current_admin)
):
    query_date = date_filter or date.today().isoformat()
    
    query = db.collection('attendance').where('date', '==', query_date)
    
    if branch:
        query = query.where('branch', '==', branch)
    if year:
        query = query.where('year', '==', year)
    
    records = await fs_query(query)
    result = []
    for r in records:
        r_dict = r.to_dict() or {}
        result.append({
            "id": r_dict.get("id"),
            "student_id": r_dict.get("student_id"),
            "student_name": r_dict.get("student_name"),
            "roll_number": r_dict.get("roll_number"),
            "branch": r_dict.get("branch"),
            "year": r_dict.get("year"),
            "time": r_dict.get("time"),
            "geo_verified": r_dict.get("geo_verified", False)
        })
    return {
        "date": query_date,
        "records": result,
        "total": len(result)
    }

@api_router.get("/admin/statistics")
async def get_statistics(
    branch: Optional[str] = None,
    year: Optional[int] = None,
    user: dict = Depends(get_current_admin)
):
    # Get student count
    student_query = db.collection('students')
    if branch:
        student_query = student_query.where('branch', '==', branch)
    if year:
        student_query = student_query.where('year', '==', year)
    
    students = list(await fs_query(student_query))
    total_students = len(students)
    
    # Get today's attendance
    today = date.today().isoformat()
    attendance_query = db.collection('attendance').where('date', '==', today)
    if branch:
        attendance_query = attendance_query.where('branch', '==', branch)
    if year:
        attendance_query = attendance_query.where('year', '==', year)
    
    attendance = list(await fs_query(attendance_query))
    today_attendance = len(attendance)
    
    return {
        "total_students": total_students,
        "today_attendance": today_attendance,
        "attendance_percentage": (today_attendance / total_students * 100) if total_students > 0 else 0
    }

@api_router.get("/admin/export-attendance")
async def export_attendance(
    branch: Optional[str] = None,
    year: Optional[int] = None,
    date_filter: Optional[str] = None,
    user: dict = Depends(get_current_admin)
):
    """Export attendance report as structured data for PDF generation"""
    from datetime import datetime as dt
    
    query = db.collection('attendance')
    
    if date_filter:
        query = query.where('date', '==', date_filter)
    if branch:
        query = query.where('branch', '==', branch)
    if year:
        query = query.where('year', '==', year)
    
    records = [doc.to_dict() for doc in await fs_query(query)]
    
    # Get all students for the filter
    student_query = db.collection('students')
    if branch:
        student_query = student_query.where('branch', '==', branch)
    if year:
        student_query = student_query.where('year', '==', year)
    all_students = list(await fs_query(student_query))
    
    # Group by branch and year
    grouped_data = {}
    for record in records:
        key = f"{record['branch']}_{record['year']}"
        if key not in grouped_data:
            grouped_data[key] = {
                "branch": record["branch"],
                "year": record["year"],
                "records": []
            }
        grouped_data[key]["records"].append({
            "roll_number": record["roll_number"],
            "student_name": record["student_name"],
            "date": record["date"],
            "time": record["time"],
            "geo_verified": record.get("geo_verified", False)
        })
    
    # Calculate statistics
    total_present = len(records)
    total_students = len(all_students)
    
    return {
        "report_generated": dt.utcnow().isoformat(),
        "filters": {
            "branch": branch or "All",
            "year": year or "All",
            "date": date_filter or "All dates"
        },
        "summary": {
            "total_students": total_students,
            "total_present": total_present,
            "attendance_rate": (total_present / total_students * 100) if total_students > 0 else 0
        },
        "grouped_data": list(grouped_data.values()),
        "all_records": [{
            "roll_number": r["roll_number"],
            "student_name": r["student_name"],
            "branch": r["branch"],
            "year": r["year"],
            "date": r["date"],
            "time": r["time"]
        } for r in records]
    }

# ==================== AI ASSISTANTS ====================

@api_router.post("/assistant/student")
async def student_assistant_chat(
    data: StudentAssistantRequest,
    user: dict = Depends(get_current_student),
):
    first_name = (user.get("name") or "Student").split()[0]
    prompt = build_chat_prompt(
        (
            "You are a helpful academic assistant for college students using the SVCK Digital app. "
            "Provide concise, accurate, study-focused help. If a question is unrelated to academics "
            "or coding, steer the user back politely."
        ),
        data.message,
        data.history,
        context={"student_name": first_name, "roll_number": user.get("roll_number")},
    )
    return {"response": await generate_gemini_response(prompt)}


@api_router.post("/assistant/admin")
async def admin_assistant_chat(
    data: AdminAssistantRequest,
    user: dict = Depends(get_current_admin),
):
    context = data.app_context.model_dump(exclude_none=True) if data.app_context else {}
    prompt = build_chat_prompt(
        (
            "You are the analytics assistant for the SVCK Digital admin dashboard. "
            "Answer using the provided attendance context, be concise, and call out when data is missing "
            "instead of inventing numbers."
        ),
        data.message,
        data.history,
        context=context,
    )
    return {"response": await generate_gemini_response(prompt)}

# ==================== WEBSOCKET ====================

@api_router.websocket("/ws/admin")
async def admin_websocket(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="Admin token required")
        return

    try:
        payload = await get_current_user_from_token(token)
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
    except HTTPException:
        await websocket.close(code=1008, reason="Invalid admin token")
        return

    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Note: No shutdown needed for Firestore - it handles connections automatically

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    is_dev = ENVIRONMENT == 'development'
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=is_dev)

"""Local educational review API. Never deploy publicly without authentication."""
import hashlib
import io
import json
import os
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, UnidentifiedImageError
from pydantic import BaseModel, Field
from pydicom.errors import InvalidDicomError

DATA = Path(os.environ.get('DATA_DIR', 'data/runtime'))
LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural effusion']
LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()
Image.MAX_IMAGE_PIXELS = 16_000_000


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def db():
    conn = sqlite3.connect(DATA / 'review.db', timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys=ON')
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def initialize():
    DATA.mkdir(parents=True, exist_ok=True)
    with db() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS studies (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, digest TEXT UNIQUE,
            created TEXT NOT NULL, status TEXT NOT NULL, demo INTEGER NOT NULL,
            result TEXT, error TEXT);
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT, study_id TEXT NOT NULL REFERENCES studies(id),
            created TEXT NOT NULL, reviewer TEXT NOT NULL, decision TEXT NOT NULL,
            comment TEXT NOT NULL, model_version TEXT NOT NULL);
        CREATE TRIGGER IF NOT EXISTS reviews_no_update BEFORE UPDATE ON reviews
        BEGIN SELECT RAISE(ABORT, 'Audit events are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS reviews_no_delete BEFORE DELETE ON reviews
        BEGIN SELECT RAISE(ABORT, 'Audit events are immutable'); END;
        ''')
        columns = {r[1] for r in c.execute('PRAGMA table_info(studies)')}
        if 'kind' not in columns:
            c.execute("ALTER TABLE studies ADD COLUMN kind TEXT NOT NULL DEFAULT 'radiograph'")
            c.execute('ALTER TABLE studies ADD COLUMN slices INTEGER NOT NULL DEFAULT 0')
        c.execute("UPDATE studies SET status='failed', error='Processing interrupted. Retry analysis.' WHERE status='processing'")


@asynccontextmanager
async def lifespan(app):
    initialize()
    yield


app = FastAPI(title='Lumen • Educational Imaging Review', version='0.2.0', lifespan=lifespan)
from backend.auth import install, configured
install(app)


def study(study_id):
    with db() as c:
        row = c.execute('SELECT * FROM studies WHERE id=?', (study_id,)).fetchone()
        if row is None:
            raise HTTPException(404, 'Study not found')
        item = dict(row)
        item['demo'] = bool(item['demo'])
        item['result'] = json.loads(item['result']) if item['result'] else None
        item['reviews'] = [dict(r) for r in c.execute('SELECT * FROM reviews WHERE study_id=? ORDER BY id', (study_id,))]
        item.pop('digest')
        return item


@app.get('/api/health')
def health():
    return {'status': 'ok', 'education_only': True, 'classifier_configured': bool(os.environ.get('CLASSIFIER_CHECKPOINT')), 'segmentation_configured': bool(os.environ.get('SEGMENTATION_CHECKPOINT')), 'authentication': 'account-session' if configured() else 'none-local-demo-only'}


@app.get('/api/studies')
def studies():
    with db() as c:
        ids = [r[0] for r in c.execute('SELECT id FROM studies ORDER BY created DESC')]
    return [study(s) for s in ids]


@app.get('/api/studies/{study_id}')
def get_study(study_id: str):
    return study(study_id)


def read_image(raw, suffix):
    if suffix == '.dcm':
        import pydicom
        from pydicom.pixels import apply_modality_lut, apply_voi_lut
        ds = pydicom.dcmread(io.BytesIO(raw))
        if int(ds.Rows) * int(ds.Columns) > 16_000_000:
            raise ValueError('DICOM exceeds 16 million pixels')
        if ds.Modality not in ('DX', 'CR') or int(getattr(ds, 'NumberOfFrames', 1)) != 1:
            raise ValueError('Only single-frame DX/CR DICOM is supported in this milestone')
        if getattr(ds, 'PhotometricInterpretation', '') not in ('MONOCHROME1', 'MONOCHROME2'):
            raise ValueError('Only monochrome radiographs are supported')
        arr = np.asarray(apply_voi_lut(apply_modality_lut(ds.pixel_array, ds), ds), dtype=np.float32)
        if arr.ndim != 2 or not np.isfinite(arr).all():
            raise ValueError('Invalid pixel array')
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / max(hi - lo, 1e-8)
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            arr = 1 - arr
        return Image.fromarray((arr * 255).astype('uint8')).convert('RGB')
    if suffix not in ('.png', '.jpg', '.jpeg'):
        raise ValueError('Upload PNG, JPEG or single-frame chest DX/CR DICOM')
    img = Image.open(io.BytesIO(raw))
    if img.width * img.height > 16_000_000:
        raise ValueError('Image exceeds 16 million pixels')
    img.load()
    if img.width < 32 or img.height < 32:
        raise ValueError('Image must be at least 32 × 32 pixels')
    return img.convert('RGB')


@app.post('/api/studies', status_code=201)
async def upload(tasks: BackgroundTasks, file: UploadFile = File(...)):
    raw = await file.read(20 * 1024 * 1024 + 1)
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, 'Maximum upload size is 20 MB')
    try:
        img = read_image(raw, Path(file.filename or '').suffix.lower())
    except (ValueError, AttributeError, KeyError, RuntimeError, NotImplementedError, OSError, InvalidDicomError, UnidentifiedImageError, Image.DecompressionBombError) as e:
        raise HTTPException(422, f'Unsupported or invalid image: {e}') from e
    digest = hashlib.sha256(raw).hexdigest()
    with LOCK:
        with db() as c:
            existing = c.execute('SELECT id FROM studies WHERE digest=?', (digest,)).fetchone()
            if existing:
                return study(existing[0])
            sid = uuid.uuid4().hex
            folder = DATA / sid
            folder.mkdir()
            img.save(folder / 'image.png')
            status = 'processing' if os.environ.get('CLASSIFIER_CHECKPOINT') else 'awaiting_model'
            # Store an opaque display name; source filenames may contain patient identifiers.
            c.execute('INSERT INTO studies(id,name,digest,created,status,demo,result,error) VALUES (?,?,?,?,?,?,?,?)', (sid, f'Uploaded scan {sid[:6]}', digest, now(), status, 0, None, None))
    if status == 'processing':
        tasks.add_task(analyze, sid)
    return study(sid)


def analyze(sid):
    # Serialize model loading/inference to bound GPU use in the single-process local app.
    with INFERENCE_LOCK:
        _analyze(sid)


def _analyze(sid):
    try:
        if study(sid)['kind'] == 'ct':
            from ml.segmentation import segment
            result = segment(DATA / sid / 'volume.nii.gz', Path(os.environ['SEGMENTATION_CHECKPOINT']), DATA / sid)
        else:
            from ml.inference import classify
            result = classify(DATA / sid / 'image.png', Path(os.environ['CLASSIFIER_CHECKPOINT']), DATA / sid)
        with db() as c:
            c.execute("UPDATE studies SET status='ready', result=?, error=NULL WHERE id=?", (json.dumps(result), sid))
    except Exception:
        import logging
        logging.exception('Inference failed for %s', sid)
        with db() as c:
            c.execute("UPDATE studies SET status='failed', error='Inference failed. Check the backend log and checkpoint, then retry.' WHERE id=?", (sid,))


@app.post('/api/studies/{study_id}/retry')
def retry(study_id: str, tasks: BackgroundTasks):
    item = study(study_id)
    if item['demo'] or item['status'] not in ('failed', 'awaiting_model'):
        raise HTTPException(409, 'This study cannot be retried')
    setting = 'SEGMENTATION_CHECKPOINT' if item['kind'] == 'ct' else 'CLASSIFIER_CHECKPOINT'
    if not os.environ.get(setting):
        raise HTTPException(409, f'Configure {setting} first')
    with db() as c:
        updated = c.execute("UPDATE studies SET status='processing', error=NULL WHERE id=? AND status IN ('failed','awaiting_model')", (study_id,)).rowcount
        if not updated:
            raise HTTPException(409, 'Analysis already started')
    tasks.add_task(analyze, study_id)
    return study(study_id)


@app.post('/api/demo', status_code=201)
def demo():
    sid = uuid.uuid4().hex
    folder = DATA / sid
    folder.mkdir()
    img = Image.new('RGB', (512, 512), '#111d2b')
    draw = ImageDraw.Draw(img)
    draw.ellipse((112, 70, 245, 425), fill='#536b80')
    draw.ellipse((267, 70, 400, 425), fill='#536b80')
    draw.line((256, 55, 256, 450), fill='#a3b5c5', width=10)
    for y in range(110, 390, 40):
        draw.arc((85, y - 60, 427, y + 40), 10, 170, fill='#8298aa', width=3)
    draw.text((20, 475), 'SYNTHETIC WORKFLOW FIXTURE - NOT A MEDICAL SCAN', fill='white')
    img.save(folder / 'image.png')
    scores = [.18, .32, .12, .09, .26]
    for i in range(5):
        overlay = Image.new('RGBA', img.size)
        d = ImageDraw.Draw(overlay)
        d.ellipse((130 + i * 12, 230, 250 + i * 12, 390), fill=(255, 166, 40, 120))
        overlay.save(folder / f'heatmap-{i}.png')
    result = {'model_version': 'synthetic-fixture-v1', 'kind': 'synthetic', 'findings': [{'label': label, 'score': score, 'heatmap': f'heatmap-{i}.png'} for i, (label, score) in enumerate(zip(LABELS, scores))], 'note': 'Invented scores and overlays for testing the interface. No medical inference was performed.'}
    with db() as c:
        c.execute('INSERT INTO studies(id,name,digest,created,status,demo,result,error) VALUES (?,?,?,?,?,?,?,?)', (sid, 'Synthetic chest illustration', None, now(), 'ready', 1, json.dumps(result), None))
    return study(sid)


@app.get('/api/studies/{study_id}/files/{filename}')
def asset(study_id: str, filename: str):
    study(study_id)
    if filename not in ['image.png'] + [f'heatmap-{i}.png' for i in range(5)]:
        raise HTTPException(404, 'File not found')
    path = DATA / study_id / filename
    if not path.is_file():
        raise HTTPException(404, 'File not found')
    return FileResponse(path, headers={'Cache-Control': 'no-store'})


class Review(BaseModel):
    reviewer: str = Field(min_length=2, max_length=80, pattern=r'\S')
    decision: str = Field(pattern='^(approve|reject)$')
    comment: str = Field(default='', max_length=2000)
    model_version: str = Field(min_length=1, max_length=200)


@app.post('/api/studies/{study_id}/reviews', status_code=201)
def review(study_id: str, body: Review, request:Request):
    item = study(study_id)
    if item['status'] != 'ready' or not item['result']:
        raise HTTPException(409, 'A completed prediction is required')
    if item['result']['model_version'] != body.model_version:
        raise HTTPException(409, 'Prediction version changed; refresh before reviewing')
    with db() as c:
        reviewer=request.state.account['username'] if request.state.account else body.reviewer.strip()
        c.execute('INSERT INTO reviews(study_id,created,reviewer,decision,comment,model_version) VALUES (?,?,?,?,?,?)', (study_id, now(), reviewer, body.decision, body.comment.strip(), body.model_version))
    return study(study_id)


@app.post('/api/volumes', status_code=201)
async def upload_volume(tasks: BackgroundTasks, file: UploadFile = File(...)):
    from backend.volume_io import validate_volume
    if not (file.filename or '').lower().endswith('.nii.gz'):
        raise HTTPException(422, 'Upload a de-identified abdominal CT volume in .nii.gz format')
    raw = await file.read(150 * 1024 * 1024 + 1)
    if len(raw) > 150 * 1024 * 1024:
        raise HTTPException(413, 'CT upload limit is 150 MB')
    try:
        volume = validate_volume(raw)
    except (ValueError, OSError, EOFError) as e:
        raise HTTPException(422, str(e)) from e
    return store_volume(volume, raw, tasks)


@app.post('/api/ct-series', status_code=201)
async def upload_ct_series(tasks: BackgroundTasks, file: UploadFile = File(...)):
    from backend.dicom_series import assemble
    import zipfile
    raw=await file.read(150*1024*1024+1)
    if len(raw)>150*1024*1024:
        raise HTTPException(413,'CT archive limit is 150 MB')
    try:
        volume=assemble(raw)
    except (ValueError, OSError, KeyError, AttributeError, EOFError, zipfile.BadZipFile, InvalidDicomError, NotImplementedError, RuntimeError) as error:
        raise HTTPException(422,'Unsupported CT series: '+str(error)) from error
    return store_volume(volume, raw, tasks)


def store_volume(volume, raw, tasks):
    digest = hashlib.sha256(raw).hexdigest()
    with LOCK:
        with db() as c:
            existing = c.execute('SELECT id FROM studies WHERE digest=?', (digest,)).fetchone()
            if existing:
                return study(existing[0])
            sid = uuid.uuid4().hex
            folder = DATA / sid; folder.mkdir()
            # Re-encode pixels and affine only, discarding free-text NIfTI header metadata.
            import nibabel as nib
            nib.save(volume, folder / 'volume.nii.gz')
            status = 'processing' if os.environ.get('SEGMENTATION_CHECKPOINT') else 'awaiting_model'
            c.execute('INSERT INTO studies(id,name,digest,created,status,demo,kind,slices) VALUES (?,?,?,?,?,?,?,?)',
                      (sid, f'Abdominal CT {sid[:6]}', digest, now(), status, 0, 'ct', volume.shape[2]))
    if status == 'processing':
        tasks.add_task(analyze, sid)
    return study(sid)


@app.get('/api/studies/{study_id}/slices/{index}')
def volume_slice(study_id: str, index: int, overlay: bool = False):
    from backend.volume_io import slice_png
    from fastapi.responses import Response
    item = study(study_id)
    if item['kind'] != 'ct' or index < 0 or index >= item['slices']:
        raise HTTPException(404, 'Slice not found')
    folder = DATA / study_id
    path = folder / ('mask.nii.gz' if overlay else 'volume.nii.gz')
    if not path.is_file():
        raise HTTPException(404, 'Segmentation is not available')
    return Response(slice_png(path, index, overlay), media_type='image/png', headers={'Cache-Control':'no-store'})

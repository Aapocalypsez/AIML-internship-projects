"""Local account sessions for an educational deployment; use HTTPS for remote access."""
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from pathlib import Path
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

SESSIONS={}
ATTEMPTS={}
LOCK=threading.Lock()


def password_hash(password,salt):
    return hashlib.pbkdf2_hmac('sha256',password.encode(),bytes.fromhex(salt),600_000).hex()


def configured():
    return bool(os.environ.get('REVIEW_USERS_FILE'))


def install(app):
    @app.middleware('http')
    async def guard(request:Request,call_next):
        request.state.account=None
        path=request.url.path
        if not configured() or not path.startswith('/api/') or path in ('/api/health','/api/auth/login'):
            return await call_next(request)
        token=request.cookies.get('lumen_session','')
        token_hash=hashlib.sha256(token.encode()).hexdigest()
        with LOCK:
            session=SESSIONS.get(token_hash)
            if session and session['expires']<time.time():
                SESSIONS.pop(token_hash,None); session=None
        if not session or session['expires']<time.time():
            return JSONResponse({'detail':'Sign in to continue'},status_code=401)
        request.state.account=session
        if request.method not in ('GET','HEAD','OPTIONS'):
            allowed=os.environ.get('ALLOWED_ORIGINS','http://127.0.0.1:5173,http://127.0.0.1:8080').split(',')
            if request.headers.get('origin') and request.headers['origin'] not in allowed:
                return JSONResponse({'detail':'Untrusted request origin'},status_code=403)
            if path!='/api/auth/logout' and session['role'] not in ('reviewer','admin'):
                return JSONResponse({'detail':'Reviewer permission required'},status_code=403)
        return await call_next(request)

    class Login(BaseModel):
        username:str=Field(min_length=1,max_length=80)
        password:str=Field(min_length=1,max_length=200)

    @app.post('/api/auth/login')
    def login(body:Login,request:Request):
        allowed=os.environ.get('ALLOWED_ORIGINS','http://127.0.0.1:5173,http://127.0.0.1:8080').split(',')
        if request.headers.get('origin') and request.headers['origin'] not in allowed:
            raise HTTPException(403,'Untrusted request origin')
        if not configured():
            raise HTTPException(409,'Account authentication is not configured')
        key=request.client.host if request.client else 'unknown'
        with LOCK:
            ATTEMPTS[key]=[t for t in ATTEMPTS.get(key,[]) if time.time()-t<60]
            if len(ATTEMPTS[key])>=5:
                raise HTTPException(429,'Too many sign-in attempts; wait one minute')
            ATTEMPTS[key].append(time.time())
        users=json.loads(Path(os.environ['REVIEW_USERS_FILE']).read_text())
        user=users.get(body.username)
        salt=user['salt'] if user else '00'*16
        digest=password_hash(body.password,salt)
        if not user or not hmac.compare_digest(digest,user['password_hash']):
            raise HTTPException(401,'Incorrect username or password')
        if user['role'] not in ('viewer','reviewer','admin'):
            raise HTTPException(403,'Invalid account role')
        token=secrets.token_urlsafe(32)
        session={'username':body.username,'role':user['role'],'expires':time.time()+8*3600}
        with LOCK:
            for expired in [key for key,value in SESSIONS.items() if value['expires']<time.time()]:
                SESSIONS.pop(expired,None)
            SESSIONS[hashlib.sha256(token.encode()).hexdigest()]=session
            ATTEMPTS.pop(key,None)
        response=JSONResponse({'username':body.username,'role':user['role']})
        response.set_cookie('lumen_session',token,httponly=True,samesite='strict',max_age=8*3600,
                            secure=os.environ.get('COOKIE_SECURE')=='true')
        return response

    @app.get('/api/auth/session')
    def current(request:Request):
        account=request.state.account
        return {'enabled':configured(),'username':account['username'] if account else None,'role':account['role'] if account else None}

    @app.post('/api/auth/logout')
    def logout(request:Request):
        token=request.cookies.get('lumen_session','')
        with LOCK:
            SESSIONS.pop(hashlib.sha256(token.encode()).hexdigest(),None)
        response=JSONResponse({'signed_out':True}); response.delete_cookie('lumen_session')
        return response

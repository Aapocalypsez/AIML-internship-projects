import json
import pytest
from fastapi.testclient import TestClient
from backend import app as module
from backend.auth import password_hash, ATTEMPTS


@pytest.fixture
def secured(tmp_path,monkeypatch):
    salt='01'*16
    users={name:{'salt':salt,'password_hash':password_hash('test-password',salt),'role':role} for name,role in [('alice','reviewer'),('reader','viewer')]}
    path=tmp_path/'users.json'; path.write_text(json.dumps(users))
    monkeypatch.setenv('REVIEW_USERS_FILE',str(path)); monkeypatch.delenv('CLASSIFIER_CHECKPOINT',raising=False)
    monkeypatch.setattr(module,'DATA',tmp_path); ATTEMPTS.clear()
    with TestClient(module.app) as client: yield client


def test_authentication_required_and_logout(secured):
    assert secured.get('/api/studies').status_code==401
    assert secured.post('/api/auth/login',json={'username':'alice','password':'incorrect'}).status_code==401
    response=secured.post('/api/auth/login',json={'username':'alice','password':'test-password'})
    assert response.status_code==200 and 'HttpOnly' in response.headers['set-cookie']
    assert secured.get('/api/studies').status_code==200
    assert secured.post('/api/auth/logout').status_code==200
    assert secured.get('/api/studies').status_code==401


def test_account_identity_overrides_submitted_name(secured):
    secured.post('/api/auth/login',json={'username':'alice','password':'test-password'})
    study=secured.post('/api/demo').json()
    response=secured.post('/api/studies/'+study['id']+'/reviews',json={'reviewer':'Spoofed name','decision':'reject','comment':'test','model_version':study['result']['model_version']})
    assert response.status_code==201 and response.json()['reviews'][0]['reviewer']=='alice'
    assert secured.post('/api/demo',headers={'Origin':'https://untrusted.example'}).status_code==403


def test_viewer_cannot_upload_or_review(secured):
    secured.post('/api/auth/login',json={'username':'reader','password':'test-password'})
    assert secured.get('/api/studies').status_code==200
    assert secured.post('/api/demo').status_code==403
    assert secured.post('/api/auth/logout').status_code==200


def test_sign_in_rate_limit(secured):
    for _ in range(5):
        assert secured.post('/api/auth/login',json={'username':'alice','password':'incorrect'}).status_code==401
    assert secured.post('/api/auth/login',json={'username':'alice','password':'incorrect'}).status_code==429


def test_login_rejects_cross_origin_requests(secured):
    response=secured.post('/api/auth/login',headers={'Origin':'https://untrusted.example'},json={'username':'alice','password':'test-password'})
    assert response.status_code==403 and 'lumen_session' not in secured.cookies


def test_expired_session_is_rejected_and_removed(secured):
    import hashlib
    from backend.auth import SESSIONS
    secured.post('/api/auth/login',json={'username':'alice','password':'test-password'})
    fingerprint=hashlib.sha256(secured.cookies['lumen_session'].encode()).hexdigest()
    SESSIONS[fingerprint]['expires']=0
    assert secured.get('/api/studies').status_code==401
    assert fingerprint not in SESSIONS

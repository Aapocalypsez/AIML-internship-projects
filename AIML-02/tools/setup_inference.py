"""Prepare private service credentials and active weight copies for local containers."""
import json
import secrets
import shutil
from pathlib import Path
from backend.auth import password_hash
from tools.create_local_account import main as create_owner


def main():
    create_owner()
    path=Path('data/auth/users.json'); users=json.loads(path.read_text())
    environment=Path('.env'); text=environment.read_text() if environment.exists() else ''
    if 'orthanc-importer' not in users:
        password=secrets.token_urlsafe(24); salt=secrets.token_hex(16)
        users['orthanc-importer']={'salt':salt,'password_hash':password_hash(password,salt),'role':'reviewer'}
        path.write_text(json.dumps(users,indent=2))
        text+='\nAPI_REVIEW_USERNAME=orthanc-importer\nAPI_REVIEW_PASSWORD='+password+'\n'
    if 'ORTHANC_PASSWORD=' not in text: text+='\nORTHANC_PASSWORD='+secrets.token_urlsafe(24)+'\n'
    environment.write_text(text)
    active=Path('models/active'); active.mkdir(parents=True,exist_ok=True)
    for label in ('classifier','spleen'):
        destination=active/(label+'.pt')
        if not destination.exists(): shutil.copyfile(Path('models/releases/20261107')/(label+'.pt'),destination)
    print('Local account, private importer credentials and active model copies are prepared.')


if __name__=='__main__': main()

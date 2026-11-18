"""Create an initial local reviewer account without printing its password."""
import json
import secrets
from pathlib import Path
from backend.auth import password_hash


def main():
    folder=Path('data/auth'); folder.mkdir(parents=True,exist_ok=True)
    path=folder/'users.json'
    if path.exists():
        print('Existing local accounts preserved.'); return
    password=secrets.token_urlsafe(20); salt=secrets.token_hex(16)
    path.write_text(json.dumps({'project-owner':{'salt':salt,'password_hash':password_hash(password,salt),'role':'admin'}},indent=2))
    (folder/'LOCAL_ACCESS.txt').write_text('Local educational application access\nUsername: project-owner\nPassword: '+password+'\nDo not share or commit this file.\n')
    print('Local account created. Credentials are in data/auth/LOCAL_ACCESS.txt.')


if __name__=='__main__':
    main()

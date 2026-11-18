"""Package source and documentation, never datasets, credentials or model weights."""
import json
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=root/'outputs/code-review-20261107'
inventory=json.loads((out/'inventory.json').read_text())
paths={root/row[0] for row in inventory['files']}
paths.update((root/'docs').glob('*.md'))
paths.update(root/name for name in ('README.md','frontend/package-lock.json','ml/requirements-lock.txt','ml/requirements-serving-lock.txt','backend/requirements-lock.txt'))
paths.update(root/name for name in ('.dvcignore','.dvc/.gitignore','.dvc/config'))
paths.update((root/'data/raw/chexpert').glob('*.dvc'))
paths.update((root/'data/raw/spleen').glob('*.dvc'))
paths.update((root/'data/splits').glob('*.dvc'))
with zipfile.ZipFile(out/'Lumen_Source_Code.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(paths):
        if file.is_file():
            archive.write(file,file.relative_to(root))
print('Packaged source and documentation:',len(paths),'files')

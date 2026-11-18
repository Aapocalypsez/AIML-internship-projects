"""Verify and extract user-supplied archives without modifying the originals."""
import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def safe_target(root, name):
    relative = PurePosixPath(name.replace('\\', '/'))
    if relative.is_absolute() or '..' in relative.parts or any(':' in part for part in relative.parts):
        raise ValueError(f'Unsafe archive path: {name}')
    target = (root / str(relative)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f'Archive path escapes destination: {name}')
    return target


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / '.extracted.json'
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['sha256'] != digest:
            raise ValueError('Archive changed. Extract to a new destination to preserve existing data.')
        return saved
    count = 0
    with (zipfile.ZipFile(archive) if archive.suffix == '.zip' else tarfile.open(archive)) as source:
        members = source.infolist() if isinstance(source, zipfile.ZipFile) else source.getmembers()
        size = sum(m.file_size if hasattr(m, 'file_size') else m.size for m in members)
        if shutil.disk_usage(destination).free < size + 2 * 1024**3:
            raise ValueError('Insufficient disk space for safe extraction')
        for member in members:
            name = member.filename if hasattr(member, 'filename') else member.name
            target = safe_target(destination, name)
            if any(p.startswith('._') or p == '__MACOSX' for p in PurePosixPath(name).parts):
                continue
            if isinstance(source, zipfile.ZipFile):
                if member.is_dir():
                    continue
                reader = source.open(member)
            else:
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f'Archive contains non-regular file: {name}')
                reader = source.extractfile(member)
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + '.partial')
            with reader, temp.open('wb') as output:
                shutil.copyfileobj(reader, output)
            temp.replace(target)
            count += 1
            if count % 20000 == 0:
                print(f'{archive.name}: {count} files extracted', flush=True)
    result = {'archive': str(archive), 'sha256': digest, 'files': count, 'uncompressed_archive_bytes': size}
    marker.write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    results = {}
    for name, pattern in [('chexpert', '*.zip'), ('spleen', '*.tar')]:
        archives = list((Path('data/raw') / name).glob(pattern))
        if len(archives) != 1:
            raise ValueError(f'Expected exactly one {pattern} in data/raw/{name}')
        results[name] = extract(archives[0], Path('data/raw') / name / 'extracted')
    Path('reports').mkdir(exist_ok=True)
    Path('reports/archive_verification.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)

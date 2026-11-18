import zipfile
from pathlib import Path

import pytest
from ml.extract_data import extract, safe_target


@pytest.mark.parametrize('name',['../escape','/absolute','C:/windows/file','a/../../bad','a\\..\\..\\bad'])
def test_reject_archive_traversal(tmp_path,name):
    with pytest.raises(ValueError):
        safe_target(tmp_path,name)


def test_archive_extraction_preserves_original_and_skips_resource_forks(tmp_path):
    archive=tmp_path/'sample.zip'
    with zipfile.ZipFile(archive,'w') as stream:
        stream.writestr('images/one.txt','hello')
        stream.writestr('images/._one.txt','metadata')
    original=archive.read_bytes()
    result=extract(archive,tmp_path/'out')
    assert result['files']==1
    assert (tmp_path/'out/images/one.txt').read_text()=='hello'
    assert extract(archive,tmp_path/'out')==result
    assert archive.read_bytes()==original

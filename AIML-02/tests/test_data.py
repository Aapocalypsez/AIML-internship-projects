import pytest
from ml.prepare_data import partition, normalize_label, chexpert, LABELS


def test_patient_partition_is_reproducible_and_grouped():
    patients=[f'patient{i}' for i in range(20)]
    split=partition(patients)
    assert split==partition(list(reversed(patients))+patients)
    assert [list(split.values()).count(s) for s in ('train','val','test')]==[14,3,3]


def test_tiny_dataset_fails_explicitly():
    with pytest.raises(ValueError):
        partition(['one-patient'])


def test_chexpert_float_labels_preserve_uncertainty():
    assert [normalize_label(v) for v in ('1.0','0.0','-1.0','')]==['1','0','-1','']
    with pytest.raises(ValueError):
        normalize_label('0.5')


def test_flat_kaggle_archive_layout_and_float_labels(tmp_path):
    import csv
    source=tmp_path/'train.csv'
    with source.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['Path','Frontal/Lateral']+LABELS)
        writer.writeheader()
        for i in range(10):
            relative=f'train/patient{i:05}/study1/view1_frontal.jpg'
            image=tmp_path/relative; image.parent.mkdir(parents=True); image.write_bytes(b'fixture')
            writer.writerow({'Path':'CheXpert-v1.0-small/'+relative,'Frontal/Lateral':'Frontal',**{k:'1.0' for k in LABELS}})
    output=tmp_path/'manifest.csv'
    chexpert(source,tmp_path,output)
    with output.open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==10 and {r['split'] for r in rows}=={'train','val','test'}
    assert all(r[LABELS[0]]=='1' for r in rows)


def test_csv_image_path_cannot_escape_root(tmp_path):
    import csv
    source=tmp_path/'train.csv'
    with source.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['Path','Frontal/Lateral']+LABELS)
        writer.writeheader()
        writer.writerow({'Path':'../patient00001/image.jpg','Frontal/Lateral':'Frontal',**{k:'0' for k in LABELS}})
    with pytest.raises(ValueError,match='escapes'):
        chexpert(source,tmp_path,tmp_path/'out.csv')

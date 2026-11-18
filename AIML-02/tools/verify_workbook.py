"""Check exported source text against the builder input without an Excel dependency."""
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

out=Path('outputs/code-review-20261107')
expected=json.loads((out/'inventory.json').read_text())['source']
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(out/'Lumen_Code_Reference.xlsx') as archive:
    strings=[]
    if 'xl/sharedStrings.xml' in archive.namelist():
        doc=ET.fromstring(archive.read('xl/sharedStrings.xml'))
        strings=[''.join(e.itertext()) for e in doc.findall('s:si',ns)]
    sheet=ET.fromstring(archive.read('xl/worksheets/sheet6.xml'))
    cells={cell.attrib['r']:cell for cell in sheet.findall('.//s:c',ns)}
    def value(address):
        cell=cells.get(address)
        if cell is None: return ''
        if cell.attrib.get('t')=='inlineStr': return ''.join(cell.find('s:is',ns).itertext())
        element=cell.find('s:v',ns)
        if element is None: return ''
        return strings[int(element.text)] if cell.attrib.get('t')=='s' else element.text
    for i,row in enumerate(expected,6):
        assert value(f'A{i}')==row[0]
        assert int(value(f'B{i}'))==row[1]
        assert int(value(f'C{i}'))==row[2]
        actual=value(f'D{i}')
        assert actual==row[3] or (row[3].startswith('=') and actual=="'"+row[3]), f'Source mismatch at {i}'
print(f'Export verification passed: {len(expected)} source chunks match the snapshot.')

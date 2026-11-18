import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = path.resolve(import.meta.dirname, '../..');
const output = path.join(root, 'outputs/code-review-20261107');
const inventory = JSON.parse(await fs.readFile(path.join(output, 'inventory.json'), 'utf8'));
const workbook = Workbook.create();
const sheets = [];

function table(name, title, headers, rows, widths) {
  const sheet = workbook.worksheets.add(name);
  sheets.push(sheet);
  sheet.showGridLines = false;
  sheet.getCell(1, 0).values = [[title]];
  sheet.getCell(1, 0).format.font = {name:'Arial',size:15,bold:true,color:'#17324D'};
  const data = [headers, ...rows.map(row => row.map(value => typeof value === 'string' && value.startsWith('=') ? "'"+value : value))];
  const area = sheet.getRangeByIndexes(4,0,data.length,headers.length);
  area.values = data;
  area.format.font = {name:'Arial',size:11,color:'#25374A'};
  area.format.verticalAlignment = 'top';
  area.format.rowHeight = name === 'Source code' ? 20 : 55;
  area.format.wrapText = name !== 'Source code';
  const head = sheet.getRangeByIndexes(4,0,1,headers.length);
  head.format = {fill:'#17324D',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},rowHeight:30,verticalAlignment:'center'};
  widths.forEach((width,index) => sheet.getRangeByIndexes(0,index,data.length+4,1).format.columnWidthPx = width);
  sheet.tables.add(`A5:${String.fromCharCode(64+headers.length)}${data.length+4}`,true,name.replaceAll(' ','')+'Table');
  sheet.freezePanes.freezeRows(5);
  sheet.freezePanes.freezeColumns(1);
  const last = headers.length-1;
  if(headers[last].includes('Your')) {
    sheet.getRangeByIndexes(5,last,rows.length,1).format.fill='#FFF4CF';
  }
  return sheet;
}

table('Start here','Lumen code review and modification guide', ['Topic','Explanation','Your notes'],[
  ['Snapshot',`Source snapshot generated ${new Date().toISOString()}. Use the original project files to change application behavior.`,''],
  ['File guide','Every included source/config file has its purpose, line count, suggested edit areas and SHA-256 fingerprint.',''],
  ['Functions','Python functions/classes and React functions/types are indexed by actual start/end line numbers.',''],
  ['Settings','Training and application controls, their locations, and the effect of changing them.',''],
  ['Source code','Full application, training, tests and support source. Filter by file. Long lines use consecutive numbered chunks. Concatenate chunks without adding characters.',''],
  ['Editing','Yellow cells are for your proposed changes and review notes. Workbook edits do not automatically modify or run project code.',''],
  ['Excluded files','Medical images, patient-level CSV rows, weights, credentials, installed libraries, runtime database and generated dependency lockfiles are not embedded.',''],
  ['Scientific status','Read the current reports and HANDOFF for measured results. Development baselines are not clinical validation.',''],
  ['Safety','For education only, not clinical use. Preserve patient splits, unknown-label masks, native image geometry and immutable reviews.',''],
  ['How to verify edits','Run pytest using .venv-ml, run npm build in frontend, and repeat only the data/model checks affected by the change.','']
], [180,780,340]);
table('File guide','Source files and responsibilities',['File','Area','Lines','Purpose','Where to change carefully','SHA-256','Your proposed change'],inventory.files,[280,160,70,430,450,470,320]);
table('Functions','Functions, classes and React types',['File','Function or class','Start line','End line','Behavior','Your proposed change'],inventory.functions,[280,200,90,90,700,350]);
table('Settings','Controls for training and local operation',['Control','Current/default value','Location','Effect and constraint','Your proposed value'],[
  ['Classifier epochs',5,'ml/train_classifier.py --epochs','Total epochs, including epochs already completed when resuming.',''],
  ['Classifier batch size',4,'ml/train_classifier.py --batch-size','Larger values use more GPU memory.',''],
  ['Known-label balancing',true,'ml/train_classifier.py --balance-known','Derives positive loss weights from training known-label counts only.',''],
  ['CT patch size option','64 or 96','ml/train_segmentation.py --roi-size','Saved in the checkpoint and read by inference. A resume cannot change patch geometry.',''],
  ['Authentication file','data/auth/users.json','REVIEW_USERS_FILE','Private salted hashes; configured by native launcher and model-enabled containers.',''],
  ['Secure cookies','false for local HTTP','COOKIE_SECURE','Set true with HTTPS for remote access. This does not configure TLS.',''],
  ['Browser origins','127.0.0.1 ports 5173 and 8080','ALLOWED_ORIGINS','Explicit comma-separated allowed origins for browser writes.',''],
  ['Inference device','Automatic GPU/CPU','INFERENCE_DEVICE','Use cpu while GPU training is active; CPU containers need no NVIDIA runtime.',''],
  ['Frozen backbone',false,'ml/train_classifier.py --freeze-backbone','Train the five-label head only. Useful development baseline; record this mode in results.',''],
  ['Classifier resume',false,'ml/train_classifier.py --resume','Restore last.pt optimizer/model/RNG state. Manifest and training mode must match.',''],
  ['Classifier input pixels','224 × 224','ml/classifier.py transform','Changing resolution changes the model input distribution and needs evaluation.',''],
  ['Unknown label policy','Ignore missing and -1','ml/train_classifier.py Images','Only known 0/1 labels contribute to loss and evaluation.',''],
  ['Split seed',42,'ml/prepare_data.py partition','Assign patients before selecting images. Never modify after final evaluation.',''],
  ['Split proportions','70% / 15% / remainder','ml/prepare_data.py partition','Patient counts are rounded down for train/val; remainder is test.',''],
  ['Development cohort','2048 / 512 / 512 images','ml/development_subset.py','Small selected cohort. Do not describe its metrics as full CheXpert benchmark results.',''],
  ['Spleen epochs',50,'ml/train_segmentation.py --epochs','Default long run. Actual executed epoch count is in model run.json.',''],
  ['CT patch size','64 × 64 × 64','ml/train_segmentation.py and ml/segmentation.py','Shared patch geometry; larger values require more memory.',''],
  ['CT spacing (mm)','1.5 × 1.5 × 2.0','ml/segmentation.py preprocessing','Resampling must be inverted for native-space output masks.',''],
  ['CT training window (HU)','-57 to 164','ml/segmentation.py preprocessing','Clip and scale before training/inference. Viewer display window is separate.',''],
  ['X-ray checkpoint','Environment variable','CLASSIFIER_CHECKPOINT','Use the saved classifier best.pt from the intended experiment.',''],
  ['CT checkpoint','Environment variable','SEGMENTATION_CHECKPOINT','Use a compatible spleen U-Net checkpoint.',''],
  ['Runtime storage','data/runtime','backend/app.py DATA_DIR','Contains local studies, files and review.db. Keep outside Git.',''],
  ['Upload limits','20 MB X-ray / 150 MB compressed CT','backend/app.py','CT decompression and voxel counts have additional bounds in volume_io.py.',''],
  ['Backend port',8000,'uvicorn command and frontend/vite.config.ts','Update both server and proxy together. Bind to 127.0.0.1 for local use.','']
], [190,230,330,650,250]);
const tests=inventory.functions.filter(r=>r[1].startsWith('test_')).map(r=>[r[0],r[1],r[4],'Run pytest; see reports/test-results-final.txt for the 49-test execution evidence.','']);
table('Tests','Regression coverage and verification',['File','Test','Behavior checked','Evidence','Your notes'],tests,[280,430,650,450,250]);
table('Source code','Full source snapshot',['File','Line','Chunk','Source text','Your proposed change'],inventory.source,[300,70,70,1450,450]);

workbook.recalculate();
const inspected=await workbook.inspect({kind:'table',range:"'Start here'!A5:C8",include:'values',tableMaxRows:4,tableMaxCols:3,maxChars:2500});
console.log(inspected.ndjson);
for(const sheet of sheets) {
  const image=await workbook.render({sheetName:sheet.name,range:sheet.name==='Source code'?'A1:D12':'A1:E10',scale:1,format:'png'});
  await fs.writeFile(path.join(output,`${sheet.name.replaceAll(' ','_')}-preview.png`),new Uint8Array(await image.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(workbook)).save(path.join(output,'Lumen_Code_Reference.xlsx'));
console.log(JSON.stringify({sheets:sheets.length,files:inventory.files.length,functions:inventory.functions.length,sourceRows:inventory.source.length}));

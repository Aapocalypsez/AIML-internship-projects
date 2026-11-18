import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';

const root=path.resolve(import.meta.dirname,'../..');
const out=path.join(root,'outputs/submission');
const build=path.join(root,'outputs/.presentation-build');
const skill='C:/Users/atulk/.codex/plugins/cache/openai-primary-runtime/presentations/26.1105.12137/skills/presentations';
const runtime='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1105-1297-0_26.1105.1297.0_x64__3k8sg7r9htsxt/dependencies';
process.env.RUNTIME_NODE_MODULES=path.join(runtime,'node/node_modules');
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const read=async file=>JSON.parse(await fs.readFile(path.join(root,file),'utf8'));
const preview=process.argv.includes('--preview');
const summary=await read(preview?'reports/deck_preview_summary.json':'reports/final_summary.json');
const presentation=Presentation.create({slideSize:{width:1280,height:720}});
const durations=[30,45,50,50,50,60,60,55,55,50,50,45];
const script=[];
function text(slide,value,x,y,w,h,size=28,color='#273D4D',bold=false){
  const shape=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=value;shape.text.style={fontFamily:'Arial',fontSize:size,color,bold};return shape;
}
function slide(title,body,notes){
  const s=presentation.slides.add();s.background.fill='#F4F7F8';
  text(s,title,70,55,1140,115,44,'#102C3B',true);
  if(body)text(s,body,75,215,1110,390);
  text(s,'For education only, not clinical use.',75,658,1000,28,18,'#526875');
  const index=presentation.slides.items.length; text(s,String(index).padStart(2,'0'),1160,658,50,28,18,'#526875');
  s.speakerNotes.text=`Suggested time: ${durations[index-1]} seconds.\n${notes}\nSources: local measured reports/final_summary.json, docs/RESULTS.md and project source. No clinical validation is claimed.`;
  script.push(`## Slide ${index}: ${title} (${durations[index-1]} seconds)\n\n${notes}\n`);
  return s;
}
slide('Lumen\nAI-assisted medical imaging','Chest X-ray findings and spleen CT masks\nAn educational workflow with recorded human review',
  'Introduce the supported tasks: five chest X-ray findings and spleen anatomy segmentation. Explain that this is an educational prototype. The demonstration connects real trained models to a review workflow. It does not establish diagnostic safety. State that the presentation will show both measured successes and failures.');
slide('Data and patient separation','CheXpert-small: frontal radiographs and report-derived labels.\n\nDecathlon Task09: 41 labeled CT volumes.\n\nPatients stay in one partition; uncertain X-ray labels are masked.',
  'Distinguish the downloaded datasets from trained checkpoints. Explain how patient grouping prevents the same patient from appearing in training and testing. CheXpert missing and uncertain findings are ignored rather than silently labeled negative. Spleen volumes have paired masks, and geometry was checked before training. The larger classifier experiment excludes all 478 patients whose test examples were examined during the baseline. Dataset files are preserved locally and excluded from Git and the source archive.');
const architecture=slide('Application architecture','',
  'Walk through a single upload. React submits a supported image, NIfTI volume or a complete DICOM CT series. FastAPI checks size, geometry and permissions, then schedules the appropriate model. DenseNet supplies scores and Grad-CAM; MONAI supplies a binary mask. SQLite and local storage preserve result metadata, image files and versioned reviews. Orthanc stores DICOM and the importer waits for a stable, explicitly complete CT series. Sessions identify the application account rather than verifying medical qualifications.');
for(const [i,label]of ['React viewer','FastAPI','Model inference','Stored results'].entries()){
  const box=architecture.shapes.add({geometry:'rect',position:{left:65+i*305,top:265,width:235,height:110},fill:'#E2EEF0',line:{fill:'#478D91',width:1}});
  box.text=label;box.text.style={fontFamily:'Arial',fontSize:27,color:'#153A49',bold:true};
  if(i<3){const arrow=architecture.shapes.add({geometry:'rightArrow',position:{left:310+i*305,top:305,width:45,height:25},fill:'#387F83',line:{fill:'none',width:0}});}
}
text(architecture,'Orthanc → complete-series importer → API\n\nAccount sessions · checkpoint fingerprints · append-only reviews',90,455,1100,150,27);
slide('Chest X-ray model','DenseNet121 transfer learning with five independent outputs.\n\nLarger cohort: 8,192 training and 2,048 validation images.\n\nKnown-label balancing and validation-selected checkpoints.',
  'Explain transfer learning: ImageNet supplies initial visual features, while the output head and backbone are fine-tuned for five findings. Independent sigmoid scores allow multiple findings in one image. The improved training cohort includes known negative examples for rare labels; validation and test sampling remain natural within their original partitions. Class imbalance is handled using counts from training only. Checkpoints are selected using validation AUROC, before the fresh test cohort is evaluated. These remain internal research experiments.');
const classification=slide('Classifier evaluation','',
  'Read the measured AUROC chart without converting it into an accuracy claim. Per-label known-positive and negative counts determine how stable each result is. Patient-bootstrap intervals account for multiple images from a patient. Calibration is fitted using validation labels only; raw and calibrated Brier scores and calibration error are evaluated on test. Small subgroup counts can make demographic comparisons unavailable. Refer to the report for full intervals, confusion matrices and subgroup counts. The fresh test was held aside until checkpoint selection.');
const findings=summary.classifier.findings;
const labels=Object.keys(findings).filter(k=>findings[k].auroc!=null);
const chart=classification.charts.add('bar',{position:{left:90,top:190,width:1090,height:380},categories:labels.map(k=>k.replace('Pleural Effusion','Effusion')),series:[{name:'AUROC',values:labels.map(k=>Number(findings[k].auroc.toFixed(3))),fill:'#248C8A'}],hasLegend:false,dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000',textStyle:{fontSize:22}},xAxis:{textStyle:{fontSize:20}},yAxis:{min:0,max:1,textStyle:{fontSize:20}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text(classification,'Internal fresh patient cohort; see report for confidence intervals and label counts.',90,595,1100,50,22);
const explanation=slide('Grad-CAM and model scores','',
  'Explain that Grad-CAM uses gradients of a finding logit with respect to convolutional features. Each finding has its own overlay and the viewer can adjust opacity. An overlay can highlight irrelevant anatomy or acquisition devices; a visually attractive map does not prove a correct diagnosis. A sigmoid value is also not automatically a calibrated clinical probability. Discuss the difference between software alignment checks, calibration measurements and independent explanation faithfulness validation.');
text(explanation,'Grad-CAM shows regions associated with a selected finding.\n\nIt does not draw a verified disease boundary.',75,230,520,300,30);
if(!preview){for(const file of ['xray.png','heatmap-3.png'])explanation.images.add({blob:new Uint8Array(await fs.readFile(path.join(root,'reports/presentation-example',file))),contentType:'image/png',fit:'contain',position:{left:680,top:190,width:500,height:390}});}
text(explanation,'Validation example · explanation aid',680,590,500,35,20);
const segmentation=slide('Spleen segmentation results','',
  'Describe three-dimensional U-Net patch training and full-volume sliding-window inference. Shared preprocessing handles orientation, spacing and CT intensity windowing. Prediction geometry is inverted back to the source scan before evaluation. Compare the original low-Dice baseline with the measured improved checkpoint. Report uncertainty and per-case failures rather than hiding them. The seven CT test cases were already examined during baseline development, so this is a development re-evaluation, not an untouched final clinical assessment.');
text(segmentation,`Baseline Dice: 0.063\n\nSelected model Dice: ${summary.segmentation.mean_dice.toFixed(3)}\n\nSeven cases; source geometry preserved.`,80,210,520,330,32);
if(!preview){for(const file of ['ct.png','ct-overlay.png'])segmentation.images.add({blob:new Uint8Array(await fs.readFile(path.join(root,'reports/presentation-example',file))),contentType:'image/png',fit:'contain',position:{left:680,top:190,width:500,height:390}});}
text(segmentation,'Validation slice · predicted anatomy mask',680,590,500,35,20,'#526875');
slide('Complete CT DICOM series','Order slices using patient coordinates.\n\nValidate orientation, spacing, identities and rescaled intensities.\n\nReject missing slices, mixed series, irregular spacing and gantry tilt.',
  'Explain that file names and instance numbering are not reliable slice geometry. The assembler uses DICOM image position, direction cosines and pixel spacing, then converts LPS coordinates to RAS. The supported input is a classic single-frame CT stack with consistent geometry. The ZIP includes an expected slice count; Orthanc automation requires stable status, Complete status and a known matching count. Stability alone cannot prove that every slice arrived. Enhanced multi-frame CT and irregular or tilted stacks are explicitly rejected. Source: https://dicom.nema.org/medical/dicom/2023d/output/chtml/part03/sect_C.7.6.2.html');
slide('Authenticated review and audit trail','Signed-in accounts identify reviewers.\n\nViewer permissions restrict writes; sessions use HTTP-only cookies.\n\nReviews preserve the prediction fingerprint and prior decisions.',
  'Explain the difference between account identity and medical credentials. Passwords are stored as salted PBKDF2 hashes. Sign-in is rate-limited, sessions expire, and the server uses the account identity even if a request submits a different reviewer name. Viewer accounts can read but cannot upload or review. Rejection adds a new event rather than erasing previous decisions. SQLite triggers help prevent ordinary edits; an administrator controlling the database can still tamper. HTTPS and deployment controls remain necessary for remote use.');
slide('Docker and integration checks','CPU model inference is packaged with read-only checkpoint mounts.\n\nOrthanc, importer, API and viewer run as local services.\n\nSynthetic fixtures test integration; real inputs test saved models.',
  'Distinguish the lightweight demo image from the model-enabled Compose override. The inference image installs CPU PyTorch, torchvision and MONAI, then mounts active checkpoints and local accounts. Tests verify genuine saved-checkpoint inference rather than invented scores. Synthetic DICOM fixtures exercise sorting, missing slices, duplicate handling and import state without patient identifiers. Local container tests establish a reproducible development workflow on this machine, not a production security certification.');
slide('Verification and known limitations',`${summary.tests} automated tests passed.\n\nReview the per-case masks, subgroup counts and confidence intervals.\n\nClinical validation, production operations and MRI support remain outside the evidence.`,
  'Summarize the actual regression result and real-model API tests. Show the recording or perform the supported local workflow. Discuss label uncertainty, small known-negative counts, limited CT cases, scanner and acquisition shifts, and uncalibrated outputs where calibration could not be fitted. No radiologist validation or medical approval is claimed. Authentication is an engineering control rather than evidence of clinical validity. The model card and results report describe the practical limits.');
slide('Ownership and next steps','AI-assisted implementation with inspectable source and experiment records.\n\nAtul supplied the brief, local-GPU preference and datasets.\n\nNext: independent review, broader data and external validation.',
  'Close by discussing future directions: broader data, stronger calibration and subgroup evidence, independent held-out cohorts, qualified clinical review and operational hardening. No clinical validation or independent radiologist approval is claimed.');
await fs.mkdir(build,{recursive:true});
for(const [i,s]of presentation.slides.items.entries()){
  const png=await presentation.export({slide:s,format:'png',scale:1});
  await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
const candidatePath=path.join(build,'candidate.pptx');
await(await PresentationFile.exportPptx(presentation)).save(candidatePath);
if(!preview)await finalizePresentation({workspaceDir:root,candidatePath,finalPath:path.join(out,'Lumen_Final_Presentation.pptx'),
  pythonExecutable:path.join(runtime,'python/python.exe'),integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
  explicitTotalSlideCount:12,materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
await fs.writeFile(path.join(out,'Presentation_Speaker_Notes.md'),'# Lumen — ten-minute presentation\n\n'+script.join('\n'));
console.log('Exported editable 12-slide presentation and ten-minute speaker notes.');

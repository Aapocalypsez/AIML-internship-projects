const ts = require('../frontend/node_modules/typescript');
const fs = require('fs');
const file = 'frontend/src/main.tsx';
const source = ts.createSourceFile(file, fs.readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
const explanations = {
  api: 'Send a same-origin API request and turn non-success responses into readable errors.',
  App: 'Manage the study queue, upload controls, image/CT viewer, model findings and review history.',
  refresh: 'Reload study details and preserve the current selection when polling.',
  act: 'Apply shared busy/error handling around a mutation and refresh the study list afterward.',
  choose: 'Select a study and reset the finding, CT slice and review-comment controls.',
  upload: 'Send NIfTI volumes to /volumes and supported radiographs to /studies.',
  review: 'Submit the decision, reviewer, notes and displayed prediction version.',
  Finding: 'Type contract for a finding label, model score and heatmap filename.',
  Study: 'Type contract for study state, modality, CT slice count, results and review events.'
};
const rows=[];
function visit(node) {
  if ((ts.isFunctionDeclaration(node) || ts.isTypeAliasDeclaration(node)) && node.name) {
    const name=node.name.text;
    rows.push([file,name,source.getLineAndCharacterOfPosition(node.getStart()).line+1,
      source.getLineAndCharacterOfPosition(node.end).line+1,explanations[name] || 'React source declaration.','']);
  }
  ts.forEachChild(node,visit);
}
visit(source);
process.stdout.write(JSON.stringify(rows));

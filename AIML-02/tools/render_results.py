"""Create reproducible scientific figures from saved data and experiment evidence."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image

reports=Path('reports'); reports.mkdir(exist_ok=True)
with Path('data/splits/classifier_development.csv').open(newline='') as f:
    samples=[r for r in csv.DictReader(f) if r['split']=='val'][:3]
ct_records=json.loads(Path('data/splits/spleen.json').read_text())
ct_record=next(r for r in ct_records if r['split']=='val')
volume=nib.as_closest_canonical(nib.load(ct_record['image']))
mask=nib.as_closest_canonical(nib.load(ct_record['label']))
image=np.asarray(volume.dataobj); labels=np.asarray(mask.dataobj)
index=int(np.argmax(labels.sum(axis=(0,1))))
fig,axes=plt.subplots(1,4,figsize=(14,4),layout='constrained')
for i,row in enumerate(samples):
    axes[i].imshow(Image.open(row['path']),cmap='gray'); axes[i].set_title(f'Validation radiograph {i+1}'); axes[i].axis('off')
axes[3].imshow(image[:,:,index].T[::-1,::-1],cmap='gray',vmin=-160,vmax=240)
axes[3].contour(labels[:,:,index].T[::-1,::-1],levels=[.5],colors=['#ffad32'],linewidths=1)
axes[3].set_title('CT with supplied spleen mask'); axes[3].axis('off')
fig.suptitle('Dataset verification examples — supplied labels, not model predictions',fontsize=13)
fig.savefig(reports/'data_preview.png',dpi=140); plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,name,metric,label in [(axes[0],'classifier-development','training_loss','X-ray training loss'),(axes[1],'spleen-development','mean_dice','Spleen validation Dice')]:
    history_path=Path('models')/name/'history.json'
    if history_path.exists():
        rows=json.loads(history_path.read_text()); ax.plot([r['epoch'] for r in rows],[r[metric] for r in rows],marker='o')
    ax.set_title(label); ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
axes[1].set_ylim(0,1)
fig.suptitle('Development runs — not clinical validation')
fig.savefig(reports/'learning_curves.png',dpi=140); plt.close(fig)
print('Saved data_preview.png and learning_curves.png')

"""Show every held-out spleen mask beside its ground truth on one fixed-rule slice."""
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--report',default='reports/segmentation_evaluation.json'); parser.add_argument('--output',default='reports/segmentation_predictions.png'); args=parser.parse_args()
    report=json.loads(Path(args.report).read_text())
    records={r['patient_id']:r for r in json.loads(Path('data/splits/spleen.json').read_text())}
    fig,axes=plt.subplots(2,4,figsize=(15,8),facecolor='#101b29')
    for index,(ax,result) in enumerate(zip(axes.flat,report['cases']),1):
        record=records[result['case']]
        source=nib.as_closest_canonical(nib.load(record['image']))
        truth=nib.as_closest_canonical(nib.load(record['label'])).get_fdata()>0
        prediction=nib.as_closest_canonical(nib.load(Path(report.get('prediction_root','data/predictions/spleen'))/result['case']/'mask.nii.gz')).get_fdata()>0
        assert source.shape==truth.shape==prediction.shape
        z=int(truth.sum(axis=(0,1)).argmax())
        orient=lambda array: array[:,:,z].T[::-1,::-1]
        ax.imshow(orient(source.get_fdata()),cmap='gray',vmin=-150,vmax=250,
                  aspect=float(source.header.get_zooms()[1]/source.header.get_zooms()[0]))
        for mask,color in ((truth,'#29d3ba'),(prediction,'#ffad55')):
            if orient(mask).any():
                ax.contour(orient(mask),levels=[.5],colors=[color],linewidths=.8)
        ax.set_title(f"Test case {index} | volume Dice {result['dice']:.3f}",color='white',fontsize=11)
        ax.axis('off')
    for ax in list(axes.flat)[len(report['cases']):]:
        ax.axis('off')
    fig.suptitle('Spleen development evaluation: all seven cases',color='white',fontsize=19)
    fig.text(.5,.025,'Teal: ground truth | Orange: prediction | Slice chosen by largest ground-truth spleen area\nFor education only. One slice cannot represent the full 3D segmentation.',ha='center',color='white',fontsize=11)
    fig.subplots_adjust(top=.90,bottom=.12,wspace=.15,hspace=.2)
    fig.savefig(args.output,dpi=140,facecolor=fig.get_facecolor())
    plt.close(fig)
    print('Rendered all held-out cases using a fixed slice-selection rule.')


if __name__=='__main__':
    main()

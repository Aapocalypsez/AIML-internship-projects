"""Export scientific ROC, reliability and learning-curve figures from saved evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from ml.classifier import LABELS


def main():
    report=json.loads(Path('reports/classifier_final_evaluation.json').read_text())
    data=np.load('reports/classifier_final_evaluation.npz')
    scores,targets,known=data['test_scores'],data['test_targets'],data['test_known']
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for index,label in enumerate(LABELS):
        ax=axes.flat[index]; y=targets[known[:,index],index]; p=scores[known[:,index],index]
        if len(np.unique(y))==2:
            fpr,tpr,_=roc_curve(y,p); ax.plot(fpr,tpr,color='#167D82',label=f"AUROC {report['findings'][label]['auroc']:.3f}")
        ax.plot([0,1],[0,1],ls='--',color='#b5bfc5'); ax.set(xlim=(0,1),ylim=(0,1),title=label,xlabel='False positive rate',ylabel='True positive rate'); ax.legend(loc='lower right'); ax.grid(alpha=.15)
    axes.flat[-1].axis('off'); axes.flat[-1].text(.05,.6,'Fresh patient-disjoint internal test cohort\nOnly known labels are evaluated.\nConfidence intervals and counts in report.\nNot an official CheXpert benchmark.',va='center',fontsize=12)
    fig.savefig('reports/classifier_roc.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for index,label in enumerate(LABELS):
        ax=axes.flat[index]; y=targets[known[:,index],index]; raw=scores[known[:,index],index]
        p=raw.copy(); coefficients=report['calibrators'].get(label)
        if coefficients:
            logits=np.log(np.clip(raw,1e-6,1-1e-6)/(1-np.clip(raw,1e-6,1-1e-6)))
            p=1/(1+np.exp(-(coefficients['slope']*logits+coefficients['intercept'])))
        for values,name,color in ((raw,'Raw','#b46b23'),(p,'Validation calibrated' if coefficients else 'Calibration unavailable','#167D82')):
            xpoints=[]; ypoints=[]
            for lower in np.arange(0,1,.1):
                mask=(values>=lower)&(values<lower+.1 if lower<.89 else values<=1)
                if mask.any(): xpoints.append(values[mask].mean()); ypoints.append(y[mask].mean())
            ax.plot(xpoints,ypoints,marker='o',label=name,color=color)
        ax.plot([0,1],[0,1],ls='--',color='#b5bfc5'); ax.set(xlim=(0,1),ylim=(0,1),title=label,xlabel='Mean model score',ylabel='Observed positive fraction'); ax.legend(fontsize=8); ax.grid(alpha=.15)
    axes.flat[-1].axis('off'); axes.flat[-1].text(.05,.6,'Ten equal-width bins; empty bins omitted.\nCalibration fitted on validation only.\nObserved fractions condition on known labels.\nSmall bins can be unstable.',va='center',fontsize=12)
    fig.savefig('reports/classifier_reliability.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    cxr=json.loads(Path('models/classifier-improved/history.json').read_text())
    axes[0].plot([r['epoch'] for r in cxr],[r['training_loss'] for r in cxr],marker='o',color='#167D82'); axes[0].set_title('Larger classifier: training loss')
    axes[1].plot([r['epoch'] for r in cxr],[np.mean([v['auroc'] for v in r['validation'].values() if v['auroc'] is not None]) for r in cxr],marker='o',color='#167D82'); axes[1].set_title('Larger classifier: validation AUROC'); axes[1].set_ylim(0,1)
    offset=0
    for folder,label in [('spleen-development','Initial baseline'),('spleen-improved','64-cube warm start'),('spleen-context','96-cube context')]:
        rows=json.loads((Path('models')/folder/'history.json').read_text()); axes[2].plot([offset+r['epoch'] for r in rows],[r['mean_dice'] for r in rows],label=label); offset+=len(rows)
    axes[2].set(title='Spleen: validation Dice',ylim=(0,1)); axes[2].legend(fontsize=8)
    for ax in axes: ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    fig.savefig('reports/final_learning_curves.png',dpi=150); plt.close(fig)
    print('Exported ROC, calibration and learning-curve figures from saved measurements.')


if __name__=='__main__': main()

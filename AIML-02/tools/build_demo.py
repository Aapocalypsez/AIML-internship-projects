"""Assemble a captioned walkthrough from real, locally captured application screens."""
import json
import shutil
from pathlib import Path


def main():
    project=Path('videos/lumen-demo'); assets=project/'assets'; assets.mkdir(exist_ok=True)
    scenes=[
        ('sign-in','01-sign-in.jpg','Sign in to review.','An account identifies each saved decision.'),
        ('upload','03-xray-upload.jpg','Upload a supported scan.','The saved model analyzes this validation example.'),
        ('explain','04-xray-output.jpg','Inspect each finding.','Grad-CAM shows influence, not a verified disease boundary.'),
        ('volume','05-ct-slices.jpg','Browse the CT volume.','Image and mask use the same scan geometry.'),
        ('mask','06-ct-overlay.jpg','Inspect imperfect masks.','A working viewer does not guarantee accurate anatomy.'),
        ('review','07-review-history.jpg','Record a versioned review.','This is a software demonstration, not a clinical assessment.'),
        ('results','08-results.jpg','Report measured limits.','Keep failures and uncertainty visible in the submission.'),
    ]
    composition=project/'compositions'; composition.mkdir(exist_ok=True)
    hosts=[]; provenance=[]
    for index,(identifier,name,title,description) in enumerate(scenes):
        source=Path('reports/demo-assets')/name
        if not source.exists(): raise FileNotFoundError('Real UI capture is missing: '+str(source))
        shutil.copyfile(source,assets/name)
        hosts.append(f'<div id="{identifier}" class="clip" data-composition-id="{identifier}" data-composition-src="compositions/{identifier}.html" data-start="{index*12}" data-duration="12" data-track-index="1" data-track-kind="graphics" data-width="1920" data-height="1080"></div>')
        (composition/(identifier+'.html')).write_text(f'''<template>
<style>
#root{{position:absolute;inset:0;background:#F1F5F6;color:#163441;font-family:Inter,sans-serif}}
#{identifier}-screen{{position:absolute;left:100px;top:140px;right:auto;bottom:auto;width:1000px;height:760px;object-fit:contain}}
#{identifier}-words{{position:absolute;left:1170px;top:250px;right:auto;bottom:auto;width:550px;height:auto}}
#{identifier}-title{{font-size:62px;line-height:1.12;font-weight:700;margin:0 0 38px}}
#{identifier}-detail{{font-size:36px;line-height:1.35;margin:0;color:#375661}}
#{identifier}-notice{{position:absolute;left:195px;top:930px;right:auto;bottom:auto;width:1530px;height:40px;font-size:28px;line-height:1.3;color:#375661}}
</style>
<div id="root" data-composition-id="{identifier}" data-width="1920" data-height="1080" data-hold="Read the genuine application screen and its explanation">
<img id="{identifier}-screen" class="clip" data-start="0" data-duration="12" src="assets/{name}" alt="Actual local Lumen application capture"/>
<div id="{identifier}-words" class="clip" data-start="0" data-duration="12"><h1 id="{identifier}-title">{title}</h1><p id="{identifier}-detail">{description}</p></div>
<div id="{identifier}-notice" class="clip" data-start="0" data-duration="12">For education only, not clinical use. Captioned walkthrough from actual application captures.</div>
</div>
<script>const tl=gsap.timeline({{paused:true}});tl.fromTo('#{identifier}-words',{{opacity:0,x:18}},{{opacity:1,x:0,duration:.4,ease:'power3.out'}},.2);window.__timelines['{identifier}']=tl;</script>
</template>''',encoding='utf-8')
        provenance.append({'scene':identifier,'capture':str(source),'method':'Actual local application screenshot; paced still sequence, not continuous screen recording','seconds':12})
    (project/'index.html').write_text('''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=1920,height=1080"><script src="https://cdn.jsdelivr.net/npm/gsap@3.15.0/dist/gsap.min.js"></script><style>*{box-sizing:border-box}html,body{margin:0;width:1920px;height:1080px;overflow:hidden;background:#F1F5F6}.clip{position:absolute;inset:0}</style></head><body><div id="root" data-composition-id="main" data-start="0" data-duration="84" data-width="1920" data-height="1080">'''+''.join(hosts)+'''</div><script>window.__timelines=window.__timelines||{};window.__timelines.main=gsap.timeline({paused:true});</script></body></html>''',encoding='utf-8')
    (project/'CAPTURE_PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
    print('Prepared seven real-capture scenes, 84 seconds total; silent readable walkthrough.')


if __name__=='__main__': main()

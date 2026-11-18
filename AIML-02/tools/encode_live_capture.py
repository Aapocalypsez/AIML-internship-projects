"""Encode genuine timestamped browser frames, retaining their original pacing."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from PIL import Image


def main():
    p=argparse.ArgumentParser(); p.add_argument('--ffmpeg',default=shutil.which('ffmpeg')); p.add_argument('--ffprobe',default=shutil.which('ffprobe')); a=p.parse_args()
    if not a.ffmpeg or not a.ffprobe: raise FileNotFoundError('FFmpeg and FFprobe are required')
    folder=Path('reports/live-capture').resolve(); metadata=json.loads((folder/'timing.json').read_text())
    frames=metadata['frames']; assert len(frames)>=2
    manifest=['ffconcat version 1.0']; dimensions=set()
    for index,frame in enumerate(frames):
        if not re.fullmatch(r'frame-\d{3}\.jpg',frame['file']): raise ValueError('Unexpected capture filename')
        image=folder/frame['file']
        with Image.open(image) as source: dimensions.add(source.size)
        end=frames[index+1]['elapsed_ms'] if index+1<len(frames) else metadata['duration_ms']
        duration=(end-frame['elapsed_ms'])/1000
        if duration<=0: raise ValueError('Capture times must increase')
        manifest.extend([f"file '{frame['file']}'",f'duration {duration:.6f}'])
    assert len(dimensions)==1,'Browser viewport changed during recording'
    manifest.append(f"file '{frames[-1]['file']}'")
    concat=folder/'frames.ffconcat'; concat.write_text('\n'.join(manifest)+'\n')
    output=Path('outputs/submission/Lumen_Live_Review.mp4').resolve()
    subprocess.run([a.ffmpeg,'-y','-loglevel','error','-safe','0','-f','concat','-i',str(concat),'-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2,fps=30','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],check=True)
    result=subprocess.run([a.ffprobe,'-v','error','-show_entries','format=duration:stream=codec_name,width,height,avg_frame_rate','-of','json',str(output)],check=True,capture_output=True,text=True)
    probe=json.loads(result.stdout); duration=float(probe['format']['duration'])
    assert abs(duration-metadata['duration_ms']/1000)<1.0
    evidence={'file':str(output),'duration_seconds':duration,'source_frames':len(frames),'source_duration_seconds':metadata['duration_ms']/1000,
              'method':metadata['method'],'events':metadata['events'],'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'probe':probe}
    Path('reports/live_recording_result.json').write_text(json.dumps(evidence,indent=2))
    subprocess.run([a.ffmpeg,'-y','-loglevel','error','-ss',str(duration/2),'-i',str(output),'-frames:v','1','reports/live_recording_midpoint.jpg'],check=True)
    print(json.dumps({'duration_seconds':duration,'source_frames':len(frames),'encoding':'H.264, original capture pacing retained'}))


if __name__=='__main__': main()

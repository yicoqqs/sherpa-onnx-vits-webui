"""Fetch verified Chinese base weights through a byte-range capable mirror."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import sys
import urllib.request
ROOT=Path(__file__).resolve().parent.parent

def main():
    manifest=json.loads((ROOT/'vendor/vits/base-models.json').read_text())['models']['C'];destination=ROOT/'data/base/c';destination.mkdir(parents=True,exist_ok=True)
    for name,info in manifest.items():
        target=destination/name;size=int(info['size']);checksum=info['oid'].split(':')[1]
        if target.exists() and hashlib.file_digest(target.open('rb'),'sha256').hexdigest()==checksum:continue
        part=target.with_suffix('.mirror.part')
        with part.open('wb') as output:output.truncate(size)
        url='https://hf-mirror.com/datasets/Plachta/sampled_audio4ft/resolve/main/VITS-Chinese/'+name
        parts=8;width=(size+parts-1)//parts
        def fetch(index):
            first=index*width;last=min(size-1,first+width-1)
            request=urllib.request.Request(url,headers={'Range':f'bytes={first}-{last}'})
            with urllib.request.urlopen(request,timeout=45) as response,part.open('r+b') as output:
                expected=f'bytes {first}-{last}/{size}'
                if response.status!=206 or response.headers.get('Content-Range')!=expected:raise ValueError('Mirror returned an unexpected byte range: '+str(response.headers.get('Content-Range')))
                output.seek(first);remaining=last-first+1
                while remaining:
                    chunk=response.read(min(1024*1024,remaining))
                    if not chunk:raise ValueError('Incomplete byte range')
                    output.write(chunk);remaining-=len(chunk)
            return index
        with concurrent.futures.ThreadPoolExecutor(max_workers=parts) as pool:
            for completed,future in enumerate(concurrent.futures.as_completed([pool.submit(fetch,i) for i in range(parts)]),1):
                future.result();print(f'{name}: {completed}/{parts} ranges complete',flush=True)
        with part.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=checksum:raise ValueError(name+' SHA256 mismatch; refusing to install')
        part.replace(target);print(name+': SHA256 verified',flush=True)

if __name__=='__main__':main()

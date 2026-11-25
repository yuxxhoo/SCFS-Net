"""Download external OpenAI CLIP initialization; never included in submission."""
import hashlib
import os
from pathlib import Path
from urllib.request import urlretrieve
HASH='5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
URL=f'https://openaipublic.azureedge.net/clip/models/{HASH}/ViT-B-16.pt'
def main():
    target=Path('pretrained/ViT-B-16.pt'); target.parent.mkdir(exist_ok=True)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest()==HASH:
            print('Verified:',target); return
        raise RuntimeError('Existing initialization has an unexpected checksum; move it before retrying.')
    temporary=target.with_suffix('.download')
    urlretrieve(URL,temporary)
    digest=hashlib.sha256(temporary.read_bytes()).hexdigest()
    if digest!=HASH:
        temporary.unlink(); raise RuntimeError('Download checksum mismatch')
    os.replace(temporary,target); print('Downloaded and verified:',target)
if __name__=='__main__': main()

"""Package saved vision drafts/revisions without rerendering or hiding failures."""
import json
import argparse
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='artifacts/plasma_painter/rusty-vision/vision-context')
    args=parser.parse_args()
    root=Path(args.root)
    manifest=json.loads((root/'manifest.json').read_text())
    artists=list(dict.fromkeys(r['artist'] for r in manifest['records']))
    columns=max((r['attempt'] for r in manifest['records']),default=0)+1
    if not artists:raise ValueError('No attempts to display')
    boards=[Image.new('RGB',(columns*384,len(artists)*288),'#f7f0df') for _ in range(8)]
    for row,artist in enumerate(artists):
        for col in range(columns):
            item=next((r for r in manifest['records'] if r['artist']==artist and r['attempt']==col),None)
            x,y=col*384,row*288
            for i,board in enumerate(boards):
                label=artist+' / attempt '+str(col+1)
                if item and 'construction' in item:
                    label+=' / '+('modified marks' if item['construction']['different_construction'] else 'COPY')
                draw=ImageDraw.Draw(board);draw.text((x+8,y+10),label,fill='#252525')
                if not item or 'render' not in item:
                    draw.text((x+8,y+100),'No valid render (not replaced)',fill='#a03030');continue
                original=item['render']['frames'][i]
                relative=Path(original.split('/vision-context/',1)[1])
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('invalid render path')
                with Image.open(root/relative) as image:board.paste(image.convert('RGB').resize((384,256)),(x,y+32))
    boards[0].save(root/'comparison.png')
    imageio.mimsave(root/'comparison.gif',[np.asarray(b) for b in boards],duration=.35,loop=0)
    print(root.resolve()/'comparison.gif')


if __name__=='__main__':main()

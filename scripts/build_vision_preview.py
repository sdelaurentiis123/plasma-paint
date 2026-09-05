"""Package saved vision drafts/revisions without rerendering or hiding failures."""
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw


def main():
    root=Path('artifacts/plasma_painter/rusty-vision/vision-context')
    manifest=json.loads((root/'manifest.json').read_text())
    boards=[Image.new('RGB',(768,864),'#f7f0df') for _ in range(8)]
    artists=['Vincent van Gogh','Georges Seurat','Édouard Manet']
    for row,artist in enumerate(artists):
        for col in range(2):
            item=next((r for r in manifest['records'] if r['artist']==artist and r['attempt']==col),None)
            x,y=col*384,row*288
            for i,board in enumerate(boards):
                draw=ImageDraw.Draw(board);draw.text((x+8,y+10),artist+(' / draft' if col==0 else ' / second attempt'),fill='#252525')
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

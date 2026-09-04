"""Collect a bounded public-domain reference pool from the Art Institute API.

One sequential image download at a time, with >=1 second between requests,
following https://api.artic.edu/docs/ . No plasma files are read or transmitted.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image

ARTISTS = [
    'Albrecht Dürer', 'Lucas Cranach', 'El Greco', 'Peter Paul Rubens',
    'Rembrandt', 'Anthony van Dyck', 'Claude Lorrain', 'Giovanni Battista Tiepolo',
    'Francisco de Goya', 'William Blake', 'John Constable', 'Joseph Mallord William Turner',
    'Eugène Delacroix', 'Gustave Courbet', 'Jean-François Millet', 'Camille Corot',
    'Édouard Manet', 'Claude Monet', 'Edgar Degas', 'Berthe Morisot', 'Mary Cassatt',
    'Camille Pissarro', 'Alfred Sisley', 'Pierre-Auguste Renoir', 'Georges Seurat',
    'Paul Signac', 'Vincent van Gogh', 'Paul Cézanne', 'Paul Gauguin',
    'Henri de Toulouse-Lautrec', 'Odilon Redon', 'Gustave Moreau', 'Edvard Munch',
    'Wassily Kandinsky', 'Paul Klee', 'Piet Mondrian', 'Pablo Picasso',
    'Georges Braque', 'Henri Matisse', 'Georgia O’Keeffe', 'Marsden Hartley',
    'Winslow Homer', 'John Singer Sargent', 'James McNeill Whistler',
    'Katsushika Hokusai', 'Utagawa Hiroshige', 'Kitagawa Utamaro', 'Kawanabe Kyōsai',
]
TOPICS = ['Chinese landscape painting', 'Japanese ink painting', 'Persian manuscript',
          'Indian miniature painting', 'medieval manuscript', 'Italian Renaissance painting',
          'Islamic geometric textile', 'Chinese calligraphy']
FIELDS = 'id,title,artist_title,artist_display,date_display,date_start,date_end,medium_display,style_titles,artwork_type_title,is_public_domain,image_id,copyright_notice'


def collect(limit=280, per_query=6):
    root=Path('data/art_references/aic'); images=root/'images'; images.mkdir(parents=True,exist_ok=True)
    static=Path('plasma_painter/web/demo/static')
    manifest_path=root/'manifest.json'
    previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records=previous.get('records',[])
    known={r['id'] for r in records}; failures=previous.get('failures',[]); coverage=previous.get('coverage',[])
    session=requests.Session(); session.headers['User-Agent']='PlasmaPainterReferenceStudy/0.1 (local academic reference collection)'
    def save():
        payload={'source':'Art Institute of Chicago','api_documentation':'https://api.artic.edu/docs/',
                 'collected_utc':datetime.now(timezone.utc).isoformat(),'purpose':'reference exploration; not automatically training targets',
                 'public_domain_filter':True,'limit':limit,'records':records,'coverage':coverage,'failures':failures}
        manifest_path.write_text(json.dumps(payload,indent=2)+'\n')
        (static/'art-library.generated.json').write_text(json.dumps(payload,separators=(',',':'))+'\n')
    for query in ARTISTS+TOPICS:
        if len(records)>=limit: break
        remaining=max(0,per_query-sum(r['query_group']==query for r in records))
        if not remaining: continue
        is_artist=query in ARTISTS
        body={'query':{'bool':{'filter':[{'term':{'is_public_domain':True}},{'exists':{'field':'image_id'}}],
                             'must':[{'match':{'artist_title':{'query':query,'operator':'and'}}}] if is_artist else [{'multi_match':{'query':query,'fields':['title','style_titles','medium_display'],'operator':'or'}}]}},
              'limit':min(60,per_query*4),'fields':FIELDS.split(',')}
        added=0
        try:
            response=session.get('https://api.artic.edu/api/v1/artworks/search',params={'params':json.dumps(body,separators=(',',':'))},timeout=25);response.raise_for_status();result=response.json()
            base=result['config']['iiif_url']
            for item in result['data']:
                if added>=remaining or len(records)>=limit: break
                if item['id'] in known or item.get('is_public_domain') is not True or not item.get('image_id'): continue
                url=f"{base}/{item['image_id']}/full/843,/0/default.jpg"
                filename=images/f"{item['id']}.jpg"
                time.sleep(1.05)
                try:
                    response=session.get(url,timeout=30);response.raise_for_status()
                    if not response.headers.get('Content-Type','').startswith('image/') or len(response.content)>20_000_000: raise ValueError('invalid image response')
                    filename.write_bytes(response.content)
                    thumb=f"art-reference-{item['id']}.generated.webp"
                    with Image.open(filename) as image:
                        image.thumbnail((400,400));image.convert('RGB').save(static/thumb,'WEBP',quality=82)
                    record={**item,'query_group':query,'group_kind':'artist' if is_artist else 'search_topic',
                            'source_page':f"https://www.artic.edu/artworks/{item['id']}",'image_url':url,
                            'file':str(filename),'sha256':hashlib.sha256(response.content).hexdigest(),
                            'thumbnail':thumb,'rights_evidence':{'is_public_domain':True,'metadata_license_links':result.get('info',{}).get('license_links',[])}}
                    records.append(record);known.add(item['id']);added+=1;save()
                except (requests.RequestException,ValueError,OSError) as error: failures.append({'query':query,'id':item['id'],'error':str(error)})
            coverage.append({'query':query,'added':added,'available_matches':result.get('pagination',{}).get('total')})
            print(f"{query}: +{added}, total {len(records)}",flush=True)
        except (requests.RequestException,ValueError,KeyError) as error:
            failures.append({'query':query,'error':str(error)});print(f"Query failed: {query}",flush=True)
        save();time.sleep(1.05)
    save();print(f"COMPLETE: {len(records)} images; {len(failures)} failures",flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--limit',type=int,default=280);p.add_argument('--per-query',type=int,default=6);a=p.parse_args();collect(a.limit,a.per_query)

"""Add requested artist museum links, without downloading restricted artwork images."""
import json
import time
from pathlib import Path

import requests


def main():
    records=[]
    for artist in ('Pablo Picasso','Andy Warhol'):
        query={'query':{'match':{'artist_title':{'query':artist,'operator':'and'}}},'limit':12,
               'fields':['id','title','artist_title','date_display','medium_display','style_titles','is_public_domain','copyright_notice']}
        response=requests.get('https://api.artic.edu/api/v1/artworks/search',params={'params':json.dumps(query)},timeout=25)
        response.raise_for_status()
        for item in response.json()['data']:
            if item.get('artist_title')!=artist: continue
            records.append({**item,'query_group':artist,'group_kind':'artist','reference_only':True,
                            'training_eligible':False,'source_page':f"https://www.artic.edu/artworks/{item['id']}"})
        time.sleep(1.05)
    payload={'source':'Art Institute of Chicago','purpose':'Museum links only; not training inputs','records':records}
    root=Path('data/art_references/artist_links');root.mkdir(parents=True,exist_ok=True)
    (root/'manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
    Path('plasma_painter/web/demo/static/artist-links.generated.json').write_text(json.dumps(payload)+'\n')
    print({artist:sum(r['artist_title']==artist for r in records) for artist in ('Pablo Picasso','Andy Warhol')})


if __name__=='__main__': main()

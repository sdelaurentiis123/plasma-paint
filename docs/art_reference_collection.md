# Art reference collection

Local-only reference exploration from the [Art Institute of Chicago API](https://api.artic.edu/docs/). Each downloaded record must have `is_public_domain: true` and an image identifier. Collection is sequential with a delay of at least one second between image requests. No plasma data is read or sent by the collector.

```
python3 scripts/collect_art_references.py --limit 320 --per-query 6
```

The initial search covers 48 named artists and eight broader topic queries, at up to six unique images per query. Repeated runs respect the per-group quota. Topic groups are search prompts, not verified movement classifications. Actual museum artist, medium, dates and style labels remain in each record. Automated retrieval still needs human triage; some matches can be prints, textiles, or other media rather than paintings.

Images and metadata live in ignored `data/art_references/aic/`. The manifest records source URLs, image URLs, image SHA-256 values, public-domain evidence, API metadata-license links, query coverage and failed requests. Ignored thumbnails and a generated JSON index feed the local `/art-library.html` browser. Source code is tracked; image files are not pushed to Git.

The source did not return public-domain image matches for several requested artists, including Picasso, Braque, Kandinsky, Klee and O'Keeffe. Their absence must not be presented as coverage. HTTP 403 image failures are recorded, not bypassed. This is a broad first collection from one museum, not a comprehensive or stylistically balanced corpus.

No images are automatically approved as training targets. No preference ratings, style adapter, or RL update is produced by downloading references. See `medium_studies.md` for the separation between image inspiration, executable code supervision, and preference training.

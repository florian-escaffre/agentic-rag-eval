# db/

Local Chroma vector index, created by `python src/ingest.py`.

The index is generated data and is not committed to git. To rebuild it:

```bash
python src/ingest.py
```

Re-running is safe: documents are stored under stable ids and overwritten, not duplicated. If you change the embedding model in `config.py`, delete this folder's contents first: vectors from different models are not comparable.
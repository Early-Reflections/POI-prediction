## TL;DR: Single-user prediction (predict_user.py)

- **Basic (latest sample, auto-latest checkpoint)**

```bash
python predict_user.py --user_id 123 -f best_conf/nyc.yml
```

- **Predict all test samples for the user**

```bash
python predict_user.py --user_id 123 -f best_conf/nyc.yml --all
```

- **Select GPU explicitly**

```bash
python predict_user.py --user_id 123 -f best_conf/nyc.yml --gpu 0
```

- **Print names (category labels)**

venv
```bash
python predict_user.py --user_id 123 -f best_conf/nyc.yml --names
```

docker
```bash
docker compose run --rm app python predict_user.py --user_id 123 -f best_conf/nyc.yml --names
```

- **Use a specific checkpoint explicitly**

```bash
python predict_user.py --user_id 123 -f best_conf/nyc.yml --checkpoint tensorboard/<timestamp>/<dataset> --names
```

- **Find an encoded user id (top 5 by frequency in test set)**

```bash
python -c "import pandas as pd; df=pd.read_csv('data/nyc/preprocessed/test_sample.csv'); print(df['UserId'].value_counts().head())"
```

Notes:
- The script expects the encoded `UserId` from `data/<dataset>/preprocessed/test_sample.csv`.
- If `--checkpoint` is not provided, it automatically uses the most recent `tensorboard/<timestamp>/<dataset>/checkpoint.pt`.

### Interpreting the output
- **True POI (encoded/original)**: encoded label and original dataset POI ID (string or int).
- **Top-K (encoded/original ids)**: predictions as encoded IDs and their mapped original POI IDs.
- **Top-K (names)**: category labels from raw files (PoiCategoryName/venueCategory). Requires raw files in `data/<dataset>/raw`; pass `--names`.
- **Top-K (scores)**: softmax probabilities. If scores are ~1/num_poi (e.g., ~0.0002 for ~5k POIs), the checkpoint is likely early/weak; train longer or pass `--checkpoint` to a stronger run.

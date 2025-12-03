# POI Mapping Demo – TL;DR

This file only covers the **mapping demo**. For training the POI model and running
`run_test.py`, see the root `README.md`.

## 1. (Optional) Refresh real-data scenarios

Prerequisite: you already have a predictions file like:

```text
log/<timestamp>/<dataset>/test_predictions_top20.csv
```

where `dataset` is one of `nyc`, `tky`, or `ca` (matching what you used in
`best_conf/{dataset}.yml`).

To rebuild the real-data demo JSON from the latest predictions, run from the
repo root:

```bash
# NYC (default)
docker compose run --rm app python mapping/export_nyc_demo.py

# Tokyo
docker compose run --rm app python mapping/export_nyc_demo.py --dataset tky
```

This writes, for example:

```text
mapping/web/public/data/demo-nyc-real.json
mapping/web/public/data/demo-tky-real.json
```

If you just want to use the built-in synthetic demo, you can skip this step and
stick with `mapping/web/public/data/demo-nyc.json`.

## 2. Choose data source in the frontend

In `mapping/web/src/App.jsx`:

```js
// Synthetic demo data
// fetch("/data/demo-nyc.json")

// Real data from export_nyc_demo.py
fetch("/data/demo-nyc-real.json")
```

Switch the `fetch` line as desired, then rebuild the web image.

## 3. Build and run the mapping web UI

From repo root:

```bash
docker compose build mapping-web
docker compose up mapping-web
```

Then open:

```text
http://localhost:4173
```

You should see the animated trajectories cycling through scenarios with
predicted POIs.

## Video 

```bash
ffmpeg -i tokyoPOI-raw.webm \
  -vf "scale=-2:1080" \
  -c:v libx264 -preset slow -crf 15 \
  -c:a aac -b:a 160k \
  output_1080p-m.mp4
```
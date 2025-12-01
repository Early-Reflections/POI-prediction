# POI Mapping Demo – TL;DR

This file only covers the **mapping demo**. For training the POI model and running
`run_test.py`, see the root `README.md`.

## 1. (Optional) Refresh real-data scenarios

Prerequisite: you already have a predictions file like:

```text
log/<timestamp>/nyc/test_predictions_top20.csv
```

To rebuild the real-data demo JSON from the latest predictions, run from the
repo root:

```bash
docker compose run --rm app python mapping/export_nyc_demo.py
```

This writes:

```text
mapping/web/public/data/demo-nyc-real.json
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

You should see the animated NYC trajectories cycling through scenarios with
predicted POIs.

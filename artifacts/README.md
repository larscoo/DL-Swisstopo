# Artifacts

Dieses Verzeichnis enthaelt die Modelle, Metriken und Inferenz-Ausgaben des Projekts.

## Schnelle Orientierung

- `runs/`
- `analysis/`
- `experiments/`
- `predictions/`
- `default_checkpoint.txt`
- `default_threshold.txt`
- `README.md`

Die Root-Ebene von `artifacts/` enthaelt bewusst keine einzelnen Modell- oder Metrics-Dateien mehr.

## Ordnerstruktur

```text
artifacts/
  README.md
  default_checkpoint.txt
  default_threshold.txt
  runs/
    local/
    server/
      seeds/
  analysis/
  experiments/
  predictions/
```

## Bedeutung der Unterordner

- `runs/local/`
  Abgelegte lokale Trainingslaeufe, z. B. fruehere Mac/MPS-Runs.

- `runs/server/`
  Vollstaendige Hauptlaeufe vom Server mit fertigem `best_model.pt`, `metrics.json` und `manifest.csv`.

- `runs/server/seeds/`
  Seed-Serien zur Stabilitaetspruefung der besten Kandidaten.

- `analysis/`
  Fehleranalysen aus `analyze_errors.py`, z. B. `false_positives.csv`, `false_negatives.csv` und `summary.json`.

- `experiments/`
  Kurzlaeufe, Mini-Subset-Vergleiche und sonstige Testexperimente.

- `predictions/`
  CSV-Ausgaben aus `predict.py`.

- `default_checkpoint.txt`
  Legt explizit fest, welches Modell App und `predict.py` standardmaessig verwenden.

- `default_threshold.txt`
  Legt den Inferenz-Threshold fuer das Default-Modell fest, falls er vom gespeicherten Checkpoint-Threshold abweichen soll.

## Namenskonvention

Empfohlen fuer neue `--output-dir` Werte:

- Vollstaendige lokale Runs:
  `artifacts/runs/local/<datum>-<modell>-<notiz>`

- Vollstaendige Server-Runs:
  `artifacts/runs/server/<datum>-<gpu>-<modell>-<notiz>`

- Vergleichs- oder Testlaeufe:
  `artifacts/experiments/<datum>-<vergleichsname>`

## Aktueller Stand

- `runs/server/sampler_only_effb0/`
  Aktuell gesetztes Default-Modell fuer App und Inferenz.

- `runs/server/seeds/`
  Enthaelt die Seed-Runs fuer `sampler_only` und `both`.

- `experiments/`
  Enthaelt die kleinen Vergleichslaeufe aus der Analysephase.

- `runs/local/root-default-run/`
  Frueherer lokaler Standard-Run, der vorher direkt im Root von `artifacts/` lag.

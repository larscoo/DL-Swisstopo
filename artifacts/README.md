# Artifacts

Dieses Verzeichnis enthält die Modelle, Metriken und Inferenz-Ausgaben des Projekts.

## Schnelle Orientierung

- `runs/`
- `analysis/`
- `experiments/`
- `predictions/`
- `default_checkpoint.txt`
- `default_threshold.txt`
- `README.md`

Die Root-Ebene von `artifacts/` enthält bewusst keine einzelnen Modell- oder Metrics-Dateien mehr.

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
  Abgelegte lokale Trainingsläufe, z. B. frühere Mac/MPS-Runs.

- `runs/server/`
  Vollständige Hauptläufe vom Server mit fertigem `best_model.pt`, `metrics.json` und `manifest.csv`.

- `runs/server/seeds/`
  Seed-Serien zur Stabilitätsprüfung der besten Kandidaten.

- `analysis/`
  Fehleranalysen aus `analyze_errors.py`, z. B. `false_positives.csv`, `false_negatives.csv` und `summary.json`.

- `experiments/`
  Kurzläufe, Mini-Subset-Vergleiche und sonstige Testexperimente.

- `predictions/`
  CSV-Ausgaben aus `predict.py`.

- `default_checkpoint.txt`
  Legt explizit fest, welches Modell App und `predict.py` standardmässig verwenden.

- `default_threshold.txt`
  Legt den Inferenz-Threshold für das Default-Modell fest, falls er vom gespeicherten Checkpoint-Threshold abweichen soll.

## Namenskonvention

Empfohlen für neue `--output-dir` Werte:

- Vollständige lokale Runs:
  `artifacts/runs/local/<datum>-<modell>-<notiz>`

- Vollständige Server-Runs:
  `artifacts/runs/server/<datum>-<gpu>-<modell>-<notiz>`

- Vergleichs- oder Testläufe:
  `artifacts/experiments/<datum>-<vergleichsname>`

## Aktueller Stand

- `runs/server/sampler_only_effb0/`
  Aktuell gesetztes Default-Modell für App und Inferenz.

- `runs/server/seeds/`
  Enthält die Seed-Runs für `sampler_only` und `both`.

- `experiments/`
  Enthält die kleinen Vergleichsläufe aus der Analysephase.

- `runs/local/root-default-run/`
  Früherer lokaler Standard-Run, der vorher direkt im Root von `artifacts/` lag.

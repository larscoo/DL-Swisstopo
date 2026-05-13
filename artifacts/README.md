# Artifacts

Dieses Verzeichnis enthaelt die Modelle, Metriken und Inferenz-Ausgaben des Projekts.

## Schnelle Orientierung

- `runs/`
- `experiments/`
- `predictions/`
- `README.md`

Die Root-Ebene von `artifacts/` enthaelt bewusst keine einzelnen Modell- oder Metrics-Dateien mehr.

## Ordnerstruktur

```text
artifacts/
  README.md
  runs/
    local/
    server/
      completed/
      incomplete/
  experiments/
  predictions/
```

## Bedeutung der Unterordner

- `runs/local/`
  Abgelegte lokale Trainingslaeufe, z. B. fruehere Mac/MPS-Runs.

- `runs/server/completed/`
  Vollstaendige Server-Runs mit fertigem `best_model.pt`, `metrics.json` und `manifest.csv`.

- `runs/server/incomplete/`
  Angefangene oder abgebrochene Server-Runs, bei denen noch nicht alle Artefakte vorliegen.

- `experiments/`
  Kurzlaeufe, Mini-Subset-Vergleiche und sonstige Testexperimente.

- `predictions/`
  CSV-Ausgaben aus `predict.py`.

## Namenskonvention

Empfohlen fuer neue `--output-dir` Werte:

- Vollstaendige lokale Runs:
  `artifacts/runs/local/<datum>-<modell>-<notiz>`

- Vollstaendige Server-Runs:
  `artifacts/runs/server/completed/<datum>-<gpu>-<modell>-<notiz>`

- Vergleichs- oder Testlaeufe:
  `artifacts/experiments/<datum>-<vergleichsname>`

## Aktueller Stand

- `runs/server/completed/a100_baseline_effb0/`
  Vollstaendiger A100-Baseline-Run mit EfficientNet-B0.

- `runs/server/incomplete/a100_classaware_both/`
  Noch unvollstaendiger Server-Run fuer class-aware Augmentation plus kombinierte Imbalance-Strategie.

- `experiments/`
  Enthaelt die kleinen Vergleichslaeufe aus der Analysephase.

- `runs/local/root-default-run/`
  Frueherer lokaler Standard-Run, der vorher direkt im Root von `artifacts/` lag.

# DL-Swisstopo

Toolkit zum Laden, Labeln, Trainieren und Testen eines Modells für Fussgängerstreifen auf Swisstopo-Tiles.

## Überblick

Der aktuelle Workflow ist:

1. Punkte erzeugen oder vorbereiten
2. Tiles nach `data/unlabeled/` herunterladen
3. Bilder im Web-UI labeln
4. Gelabelte Bilder werden automatisch nach `data/y` oder `data/n` verschoben
5. Modell mit `train.py` trainieren
6. Modell mit `predict.py` auf neue Bilder anwenden

## Projektstruktur

```text
data/
  unlabeled/     Noch nicht gelabelte Bilder für das Web-UI
  y/             Positive Bilder: Fussgängerstreifen vorhanden
  n/             Negative Bilder: kein Fussgängerstreifen
  labels.csv     Metadaten für offene Bilder in data/unlabeled

artifacts/
  best_model.pt
  metrics.json
  manifest.csv

generate_points_grid.py
download_tiles.py
app.py
train.py
predict.py
```

## Bedeutungen der Labels

- `1` oder positiv: Fussgängerstreifen vorhanden
- `0` oder negativ: kein Fussgängerstreifen vorhanden

## Voraussetzungen

Mindestens benötigt:

- Python 3
- `requests` für den Tile-Download
- `flask`, `pillow`, `torch`
- optional `torchvision`

Beispiel:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install requests flask pillow torch torchvision
```

## Quickstart

### 1. Punkte erzeugen

Beispiel mit Preset:

```bash
python3 generate_points_grid.py \
  --preset glarus \
  --sampling blocks \
  --num-blocks 64 \
  --block-size 4 \
  --region-id region_glarus_01 \
  --output points.csv
```

Eigener Ausschnitt:

```bash
python3 generate_points_grid.py \
  --xmin 2722500 \
  --ymin 1207000 \
  --xmax 2725500 \
  --ymax 1209500 \
  --step 25 \
  --sampling blocks \
  --num-blocks 64 \
  --block-size 4 \
  --region-id region_custom_01 \
  --output points.csv
```

Hinweis:

- Koordinaten sind in `EPSG:2056`
- `points.csv` braucht mindestens `x`, `y`, `region_id`

### 2. Tiles herunterladen

```bash
python3 download_tiles.py
```

Das Skript:

- laedt Bilder nach `data/unlabeled/<region_id>/`
- speichert sie als Koordinaten-Dateinamen wie `2725075_1208825.jpg`
- schreibt Metadaten nach `data/labels.csv`

Wichtige Optionen:

- `--input-csv points.csv`
- `--output-dir data/unlabeled`
- `--output-labels-csv data/labels.csv`
- `--overwrite`

### 3. Bilder im Web-UI labeln

Webserver starten:

```bash
python3 app.py
```

Dann im Browser die angezeigte lokale URL öffnen.

Der Labeling-Tab zeigt nur offene Bilder aus `data/unlabeled`.

Beim Speichern gilt:

- `1` -> Datei wird nach `data/y` verschoben
- `0` -> Datei wird nach `data/n` verschoben

Danach verschwindet das Bild aus der offenen Queue.

Wichtig:

- Neue Bilder in `data/unlabeled` werden automatisch erkannt
- Unterordner in `data/unlabeled` sind erlaubt, zum Beispiel `data/unlabeled/zuerich/`
- Aus solchen Unterordnern wird automatisch eine lesbare `region_id`

### 4. Modell trainieren

Einfacher Testlauf auf dem Mac:

```bash
python3 train.py \
  --device cpu \
  --model simple_cnn \
  --pretrained off \
  --epochs 2 \
  --batch-size 32 \
  --lr 1e-4
```

Empfohlener grösserer Lauf:

```bash
python3 train.py \
  --balanced-sampling \
  --model auto \
  --pretrained auto \
  --device auto \
  --epochs 12 \
  --batch-size 64 \
  --spatial-bin-size 1000 \
  --output-dir artifacts
```

Was `train.py` macht:

- trainiert direkt aus `data/y` und `data/n`
- liest Koordinaten aus Dateinamen wie `x_y.png`
- erzeugt einen räumlichen Split für `train`, `val`, `test`
- speichert das beste Modell in `artifacts/best_model.pt`

Wichtige Optionen:

- `--device auto|cpu|mps|cuda`
- `--model auto|simple_cnn|resnet18|efficientnet_b0`
- `--pretrained auto|on|off`
- `--balanced-sampling`
- `--spatial-bin-size 1000`
- `--csv-path data/labels.csv`

Outputs:

- `artifacts/best_model.pt`
- `artifacts/metrics.json`
- `artifacts/manifest.csv`

## Inferenz

Einzelbild:

```bash
python3 predict.py \
  --checkpoint artifacts/best_model.pt \
  --input data/y/2599350_1200900.png
```

Ganzer Ordner:

```bash
python3 predict.py \
  --checkpoint artifacts/best_model.pt \
  --input data/y \
  --output-csv artifacts/predictions.csv
```

Die Ausgabe enthält:

- `score`: Modellwahrscheinlichkeit für Fussgängerstreifen
- `prediction`: `1` oder `0`
- `threshold`: verwendete Entscheidungsschwelle

## Formate

### `points.csv`

```csv
x,y,label,region_id,block_id
2725075.0,1208825.0,,region_glarus_01,block_0001
2725100.0,1208825.0,crosswalk,region_glarus_01,block_0001
```

Hinweise:

- `label` ist optional
- `block_id` ist optional
- gültige Labelwerte für CSV-basiertes Training sind `1`, `0`, `crosswalk`, `no_crosswalk`, `ignore`

### `data/labels.csv`

```csv
image_path,label,region_id,x,y,block_id
data/unlabeled/region_glarus_01/2725075_1208825.jpg,,region_glarus_01,2725075,1208825,block_0001
```

Hinweise:

- die Datei beschreibt nur die offene Labeling-Queue
- gelabelte Bilder bleiben für das Training in `data/y` und `data/n`
- offene Bilder haben ein leeres `label`

## Web-UI

Die Web-App hat zwei Bereiche:

- `Labeling`: offene Bilder sichten und nach `data/y` oder `data/n` verschieben
- `Modell-Test`: einzelne Bilder hochladen und mit dem gespeicherten Modell prüfen

Features:

- Drag-and-Drop für Bild-Upload im Modell-Test
- Sortierung und Filterung der Vorhersagen
- automatische Queue-Synchronisation für `data/unlabeled`

## Hinweise zum aktuellen Stand

- Der aktive Workflow nutzt `data/unlabeled`, `data/y`, `data/n` und `data/labels.csv`
- Alte `images/...`- und Regions-Praefix-Flows sind nicht mehr Teil des aktiven Setups

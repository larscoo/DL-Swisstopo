# DL-Swisstopo Hackathon Toolkit

Mini-Toolkit für den Deep-Learning-Hackathon:  
Punkte erzeugen -> 25m x 25m Tiles von Swisstopo laden -> labels.csv aufbauen -> fürs Training nutzen.

## Was dieses Repo macht

- Batch-Download von Luftbildern (WMS `ch.swisstopo.swissimage`)
- Automatische Generierung von Rasterpunkten (inkl. Preset für Glarus)
- Fortschrittsanzeige mit Prozent + ETA beim Download
- Ausgabe in einem sauberen Format für ML (`images/` + `labels.csv`)

## Dateien

- `generate_points_grid.py`: erstellt `points.csv` mit Koordinaten
- `download_tiles.py`: lädt Bilder und schreibt `labels.csv`
- `points.csv`: Eingabe mit Punkten (`x,y,label,region_id`)
- `images/`: heruntergeladene Tiles (pro Region in `images/<region_id>/`)
- `labels.csv`: Datensatz-Metadaten fürs Training

Orte mit gelabelten Daten:
- Basel Innenstadt: `blc_00001.jpg`
- Bern (to be continued)
- Fribourg (to be continued)
- Locarno (to be continued)
- Genf (to be continued)

## Setup

```bash
python3 -m pip install requests
```

## Quickstart (Glarus)

1. Punkte für Glarus generieren (zusammenhängende 4x4-Blöcke):

```bash
python3 generate_points_grid.py \
  --preset glarus \
  --sampling blocks \
  --num-blocks 64 \
  --block-size 4 \
  --region-id region_glarus_01 \
  --output points.csv
```

2. Tiles herunterladen:

```bash
python3 download_tiles.py
```

Standardverhalten: `labels.csv` wird erweitert (Append), damit bestehende Labels nicht verloren gehen.  
Nur wenn du bewusst neu starten willst:

```bash
python3 download_tiles.py --overwrite
```

3. Ergebnis:
- Bilder in `images/<region_id>/`
- Metadaten in `labels.csv`

## Eigene Gebiete statt Preset

Beispiel für Basel-Stadt (zusammenhängende 4x4-Blöcke):

```bash
python3 generate_points_grid.py \
  --preset basel_stadt \
  --sampling blocks \
  --num-blocks 64 \
  --block-size 4 \
  --region-id region_basel_stadt_01 \
  --output points.csv
```

Innenstadt-Fokus (engeres Preset):

```bash
python3 generate_points_grid.py \
  --preset basel_stadt_core \
  --sampling blocks \
  --num-blocks 64 \
  --block-size 4 \
  --region-id region_basel_stadt_core_01 \
  --output points.csv
```

```bash
python3 generate_points_grid.py \
  --xmin 2722500 --ymin 1207000 \
  --xmax 2725500 --ymax 1209500 \
  --step 25 \
  --sampling blocks \
  --num-blocks 64 \
  --block-size 4 \
  --region-id region_custom_01 \
  --output points.csv
```

Hinweis: Koordinaten sind in `EPSG:2056`.

## Format von `points.csv`

```csv
x,y,label,region_id
2725075.0,1208825.0,,region_glarus_01
2722950.0,1207150.0,crosswalk,region_glarus_01
```

- `label` ist optional beim Download.
- `block_id` ist optional. Wenn gesetzt (z. B. mit `--sampling blocks`), kann das Labeling-UI echte 4x4-Nachbarschaften seitenweise anzeigen.
- Empfohlenes Labeling:
  - `crosswalk`
  - `no_crosswalk`
  - optional `ignore` für unklare Fälle

## Format von `labels.csv`

```csv
image_path,label,region_id,x,y
images/tile_00001.jpg,,region_glarus_01,2725075.0,1208825.0
```

## Download-Fortschritt

`download_tiles.py` zeigt live:
- `Progress: i/total`
- Prozent
- Elapsed
- ETA

## Tipps für gute Datenqualität

- Nutze mehrere Regionen (nicht nur ein Quartier)
- Sammle harte Negativbeispiele (Straßenmarkierungen ohne Zebrastreifen)
- Verteile Train/Val/Test nach `region_id` (kein Spatial Leakage)
- Unklare Bilder als `ignore` markieren statt raten

## Troubleshooting

- `Input file 'points.csv' not found`  
  -> zuerst `generate_points_grid.py` ausführen oder eigene `points.csv` anlegen.

- Sehr viele ähnliche Bilder  
  -> `step` erhöhen (z. B. `50`) oder `max-points` reduzieren.

- Download langsam  
  -> normal bei vielen Requests; Fortschritt/ETA beobachten.

---

Wenn du willst, kann als nächstes ein kleines Labeling-Tool mit Tastaturkürzeln (`1/0/i`) ergänzt werden.

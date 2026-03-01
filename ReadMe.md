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
- `images/`: heruntergeladene Tiles
- `labels.csv`: Datensatz-Metadaten fürs Training

## Setup

```bash
python3 -m pip install requests
```

## Quickstart (Glarus)

1. Punkte für Glarus generieren:

```bash
python3 generate_points_grid.py --preset glarus --max-points 120 --region-id region_glarus_01 --output points.csv
```

2. Tiles herunterladen:

```bash
python3 download_tiles.py
```

3. Ergebnis:
- Bilder in `images/`
- Metadaten in `labels.csv`

## Eigene Gebiete statt Preset

```bash
python3 generate_points_grid.py \
  --xmin 2722500 --ymin 1207000 \
  --xmax 2725500 --ymax 1209500 \
  --step 25 \
  --max-points 300 \
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

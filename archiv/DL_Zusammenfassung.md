# DeepLearningKurs Beurteilung 2026: Antwortblatt

Diese Datei ist als Vorbereitung für Fragen aus dem Bewertungsraster gedacht. Sie enthält nicht das Raster selbst, sondern die inhaltlichen Antworten, die man zu den fünf Bereichen des Projekts kennen sollte.

## 1. Eckdaten Datensatz

### Was ist der Datensatz?

Der Datensatz besteht aus Swisstopo-Tiles, die auf das Vorhandensein von Fussgängerstreifen klassifiziert werden. Jedes Bild ist binär gelabelt:

- `1`: Fussgängerstreifen vorhanden
- `0`: kein Fussgängerstreifen vorhanden

Die Bilder werden im Workflow zuerst nach `data/unlabeled/` geladen, dann im Web-UI manuell gelabelt und anschliessend nach `data/y` oder `data/n` verschoben.

### Wie gross ist der Datensatz?

Die vollständige Hauptauswertung basiert auf insgesamt `107922` Bildern:

- Train: `75097`
- Validation: `16412`
- Test: `16413`

Klassenverteilung über alle Splits:

- Positiv: `3159`
- Negativ: `104763`

Das bedeutet:

- Positiver Anteil: rund `2.9 %`
- Negativer Anteil: rund `97.1 %`

Die Daten sind also stark unausgeglichen. Genau das ist eines der zentralen Probleme des Projekts.

### Wie werden die Daten aufgeteilt?

Die Aufteilung erfolgt nicht zufällig pro Bild, sondern räumlich über Gruppen in `EPSG:2056`. Aus den Koordinaten wird eine räumliche Gruppe gebildet. Standardmässig ist die Bin-Grösse `1000` Meter.

Warum das wichtig ist:

- Benachbarte Tiles sehen sich oft sehr ähnlich.
- Ein normaler Zufallssplit würde leicht zu Datenleck führen.
- Ein räumlicher Split ist realistischer, weil das Modell auf geografisch getrennten Bereichen getestet wird.

Die Split-Ratios sind:

- Train: `70 %`
- Validation: `15 %`
- Test: `15 %`

Im Hauptlauf wurden `81` räumliche Gruppen verwendet.

### Wie ist die Datenqualität zu beurteilen?

Die Datenqualität ist insgesamt gut genug für das Training, aber nicht perfekt.

Stärken:

- Die Bilder stammen aus einer konsistenten Quelle.
- Die Labels sind inhaltlich klar definiert.
- Die Koordinaten sind strukturiert vorhanden.
- Defekte Bilder werden vor dem Training validiert und übersprungen.

Schwächen:

- Die Klassen sind stark unausgeglichen.
- Manuelles Labeling kann Fehler enthalten.
- Grenzfälle sind möglich, zum Beispiel schlecht sichtbare oder teilweise verdeckte Fussgängerstreifen.
- Räumliche Cluster können zu Verteilungen führen, die regional unterschiedlich sind.

### Welche Risiken oder Biases gibt es?

- Klassenimbalance: Das Modell sieht viel mehr negative als positive Beispiele.
- Regionaler Bias: Bestimmte Regionen können visuell dominieren.
- Visueller Bias: Beleuchtung, Jahreszeit, Perspektive und Schatten können das Modell beeinflussen.
- Label-Bias: Uneinheitliche Entscheidungen bei schwer erkennbaren Fällen sind möglich.

### Kurzantwort für die Prüfung

Der Datensatz ist fachlich passend, aber stark unausgeglichen. Die wichtigste methodische Entscheidung war deshalb, den Split räumlich statt rein zufällig zu machen und die Klassenimbalance gezielt im Training zu behandeln.

## 2. Architektur des Modells

### Welches Modell wurde verwendet?

Die stärksten Läufe im Repository verwenden `EfficientNet-B0` mit vortrainierten ImageNet-Gewichten.

Zusätzlich existieren im Code:

- `simple_cnn`
- `resnet18`
- `efficientnet_b0`
- `auto`, was wenn möglich `efficientnet_b0` wählt

### Warum wurde EfficientNet-B0 gewählt?

EfficientNet-B0 ist ein guter Kompromiss aus:

- Modellgrösse
- Rechenaufwand
- Genauigkeit
- guter Verfügbarkeit in `torchvision`

Für diese Aufgabe ist das sinnvoll, weil Luft- oder Orthofotos visuelle Muster enthalten, die ein modernes CNN gut lernen kann, ohne dass gleich ein sehr grosses Modell nötig ist.

### Warum vortrainierte Gewichte?

Vortrainierte Gewichte helfen, weil:

- das Modell bereits allgemeine visuelle Merkmale kennt
- weniger Daten nötig sind als bei vollständigem Training von null
- das Training stabiler und schneller konvergiert

Das ist besonders nützlich, wenn die positive Klasse selten ist.

### Wie wurde das Modell angepasst?

Bei `efficientnet_b0` wurde der Klassifikationskopf auf eine binäre Ausgabe umgebaut:

- letzter Layer ersetzt
- Ausgabe ist ein einzelner Logit
- Training mit `BCEWithLogitsLoss`

### Warum ist die Architektur passend zur Aufgabe?

Die Aufgabe ist eine binäre Bildklassifikation. Ein CNN ist dafür passend, weil es lokale Muster, Kanten, Kontraste und Formen erkennen kann. Fussgängerstreifen haben visuelle Strukturen, die sich mit CNNs gut erfassen lassen.

EfficientNet-B0 ist hier sinnvoller als ein sehr einfaches CNN, weil:

- es stärkere Feature-Extraktion bietet
- vortrainiert genutzt werden kann
- es bei moderatem Ressourcenbedarf starke Resultate liefert

### Kurzantwort für die Prüfung

Wir haben uns für EfficientNet-B0 mit vortrainierten Gewichten entschieden, weil das Modell für Bildklassifikation stark, effizient und pragmatisch ist. Für diese binäre Aufgabe ist das eine passende Architektur mit gutem Verhältnis aus Genauigkeit und Rechenaufwand.

## 3. Optimierungen

### Was war die wichtigste technische Herausforderung?

Die wichtigste Herausforderung war die starke Klassenimbalance. Nur rund `2.9 %` der Bilder sind positiv.

### Welche Optimierungsstrategien wurden getestet?

Im Projekt wurden mehrere Strategien verglichen:

- `pos_weight`
- `sampler`
- `both`
- `class_aware` Augmentation
- grössere Bildgrösse mit `img_size=320`

### Was bedeutet `pos_weight`?

`pos_weight` in `BCEWithLogitsLoss` gewichtet positive Beispiele stärker. Fehler auf der seltenen positiven Klasse werden dadurch im Loss stärker bestraft.

### Was bedeutet `sampler`?

Mit `WeightedRandomSampler` werden positive und negative Beispiele im Training ausgeglichener gezogen. Das hilft, damit die seltene Klasse im Training häufiger sichtbar wird.

### Was bedeutet `both`?

`both` kombiniert:

- gewichteten Loss
- balanciertes Sampling

Das war im Repository die stärkste Variante nach `F1`.

### Was bedeutet `class_aware` Augmentation?

Bei `class_aware` erhalten positive Bilder stärkere Augmentierungen als negative. Für positive Beispiele werden unter anderem genutzt:

- Horizontal Flip
- Vertical Flip
- Rotation
- Perspective Transform
- Color Jitter
- Gaussian Blur

Die Idee dahinter ist, die seltene positive Klasse robuster zu machen.

### Welche Standard-Augmentierungen wurden verwendet?

Im Standardmodus werden unter anderem genutzt:

- Resize
- Horizontal Flip
- Vertical Flip
- Rotationen um 90, 180 und 270 Grad
- Color Jitter
- Gaussian Blur

### Wie wurde der Threshold gewählt?

Der Entscheidungsthreshold wurde nicht fix auf `0.5` gesetzt, sondern über die Validation optimiert. Wenn kein fester Threshold angegeben wird, sucht das Skript zwischen `0.05` und `0.95` in `0.05`-Schritten den besten Threshold nach `F1`.

Das ist sinnvoll, weil bei unausgeglichenen Daten der beste operative Threshold oft nicht bei `0.5` liegt.

### Welche weiteren methodischen Entscheidungen sind wichtig?

- Räumlicher Split statt Zufallssplit
- Validierung unlesbarer Bilder vor dem Training
- Nutzung vortrainierter Gewichte
- Vergleich mehrerer Imbalance-Strategien statt nur eines Runs
- Seed-Runs zur Stabilitätsprüfung

### Kurzantwort für die Prüfung

Die Optimierungen waren direkt auf das Hauptproblem des Datensatzes ausgerichtet: Klassenimbalance und Generalisierbarkeit. Deshalb wurden Sampling, gewichteter Loss, Augmentierung und Threshold-Tuning systematisch verglichen.

## 4. Resultate

### Welcher Lauf ist der beste?

Nach `F1` ist `a100_balanced_sampling` der beste Lauf im Repository.

Kennzahlen:

- Accuracy: `0.9969`
- Precision: `0.9280`
- Recall: `0.9687`
- F1: `0.9479`
- PR-AUC: `0.9759`
- Threshold: `0.95`

Konfusionsmatrix:

- TP: `464`
- TN: `15898`
- FP: `36`
- FN: `15`

### Welche anderen gute Läufe gibt es?

`sampler_only_effb0`:

- Accuracy: `0.9968`
- Precision: `0.9383`
- Recall: `0.9520`
- F1: `0.9451`
- PR-AUC: `0.9720`

`a100_baseline_effb0`:

- Accuracy: `0.9966`
- Precision: `0.9343`
- Recall: `0.9499`
- F1: `0.9420`
- PR-AUC: `0.9808`

Wichtige Interpretation:

- Der beste `F1` kommt von `both`.
- Die beste `PR-AUC` kommt vom Baseline-Lauf mit `pos_weight`.
- Der `img320`-Lauf hat zwar die höchste Precision (`0.9462`), verliert aber bei Recall und F1.

### Warum ist Accuracy allein hier nicht genug?

Weil der Datensatz extrem unausgeglichen ist. Bei fast nur negativen Beispielen kann Accuracy sehr hoch sein, auch wenn die positive Klasse schlecht erkannt wird.

Deshalb sind wichtiger:

- Precision
- Recall
- F1
- PR-AUC

### Was sagen die Resultate fachlich aus?

Die Resultate zeigen, dass das Modell Fussgängerstreifen sehr zuverlässig erkennt. Besonders wichtig ist der hohe Recall des besten Laufs, weil nur `15` positive Testfälle verpasst wurden.

Gleichzeitig zeigt die Zahl der False Positives (`36`), dass das Modell gelegentlich ähnliche Strukturen fälschlich als Fussgängerstreifen interpretiert.

### Welche Schlussfolgerung ist fachlich fair?

Das Modell ist stark und praktisch brauchbar, aber nicht fehlerfrei. Es funktioniert überzeugend auf dem Testsplit, dennoch bleiben Fehlerquellen wie regionale Unterschiede, Grenzfälle und Labelunsicherheiten bestehen.

### Kurzantwort für die Prüfung

Die Resultate sind überzeugend, weil sie nicht nur eine hohe Accuracy, sondern auch sehr gute Werte bei F1, Recall und PR-AUC zeigen. Besonders wichtig ist, dass die Bewertung auf einem räumlich getrennten Testsplit erfolgt und damit realistischer ist als ein einfacher Zufallssplit.

## 5. Persönliche Erkenntnisse

Hinweis: Dieser Teil sollte mündlich möglichst persönlich formuliert werden. Die folgenden Punkte sind gute, projektspezifische Antwortvorlagen.

### Was war eine zentrale Erkenntnis?

Eine zentrale Erkenntnis war, dass bei diesem Problem nicht nur die Modellarchitektur zählt. Viel entscheidender waren saubere Datenaufbereitung, sinnvoller Split und der Umgang mit Klassenimbalance.

### Was war überraschend?

Überraschend war, dass kleine methodische Änderungen wie:

- balanciertes Sampling
- gewichteter Loss
- Threshold-Tuning

einen grossen Einfluss auf die Qualität der Resultate haben können, obwohl das Grundmodell gleich bleibt.

### Was habt ihr über Evaluation gelernt?

Wir haben gelernt, dass Accuracy bei stark unausgeglichenen Daten zu optimistisch wirken kann. Für diese Aufgabe sind F1, Recall und PR-AUC deutlich aussagekräftiger.

### Was habt ihr über Daten gelernt?

Wir haben gelernt, dass räumlich benachbarte Bilddaten ein echtes Risiko für Data Leakage sind. Ein räumlicher Split ist darum methodisch sauberer und realistischer als ein normaler Zufallssplit.

### Was würdet ihr als Nächstes verbessern?

- Fehleranalyse der False Positives und False Negatives systematischer auswerten
- mehr positive Beispiele sammeln
- regionale Vielfalt weiter erhöhen
- feinere Threshold-Wahl je nach Anwendungsziel
- eventuell modernere Backbones oder Ensembles testen

### Wo liegen die Grenzen eures Ansatzes?

- Das Modell ist von der Qualität der Labels abhängig.
- Seltene oder schlecht sichtbare Fussgängerstreifen bleiben schwierig.
- Gute Testwerte garantieren nicht automatisch perfekte Generalisierung auf alle Regionen.
- Binäre Klassifikation sagt nur, ob ein Fussgängerstreifen vorhanden ist, aber nicht wo genau.

### Gute Abschlussformulierung für die Prüfung

Die wichtigste Erkenntnis aus dem Projekt war, dass robuste Resultate nicht allein aus einem guten Modell entstehen. Entscheidend waren die Kombination aus sauberem räumlichem Split, passender Imbalance-Behandlung, sinnvoller Evaluation und kritischer Interpretation der Metriken.

## Sehr kurze Prüfungsfassung

Wenn du extrem knapp antworten musst, kannst du dir diese fünf Kernaussagen merken:

1. Der Datensatz ist gross genug, aber stark unausgeglichen, deshalb war der Umgang mit der positiven Klasse zentral.
2. EfficientNet-B0 mit vortrainierten Gewichten war ein sinnvoller Kompromiss aus Leistung und Effizienz.
3. Die wichtigsten Optimierungen waren räumlicher Split, balanciertes Sampling, gewichteter Loss und Threshold-Tuning.
4. Der beste Lauf erreichte `F1 = 0.9479` bei hohem Recall auf einem räumlich getrennten Testsplit.
5. Die wichtigste Erkenntnis war, dass Datenaufbereitung und Evaluation mindestens so wichtig sind wie die reine Modellarchitektur.

# DL-Swisstopo: Prüfungszusammenfassung

## Ziel des Projekts

Mein Projekt klassifiziert Swisstopo-Bildkacheln binär in:

- `1`: Fussgängerstreifen vorhanden
- `0`: kein Fussgängerstreifen vorhanden

Die Kernidee war, aus georeferenzierten Luftbildern ein robustes Bildklassifikationsmodell zu bauen, das auch auf räumlich getrennten Gebieten gut generalisiert. Ich habe mich bewusst nicht nur auf ein "starkes Modell" konzentriert, sondern auf die ganze Pipeline: Datenerzeugung, manuelles Labeling, sauberer Split, Imbalance-Behandlung, Training, Evaluation und Fehleranalyse.

## Was im Bewertungsraster voraussichtlich zentral ist

Für die Prüfung würde ich meinen Projektweg entlang dieser Punkte erklären:

1. Problem und Zielsetzung
2. Datensatz und Labeling
3. Wahl der Modellarchitektur
4. Trainingsstrategie und Optimierung
5. Evaluation und Metriken
6. Reflexion: Was funktioniert hat, was schwierig war und was ich gelernt habe

Genau auf diese Punkte ist die Zusammenfassung unten ausgerichtet.

## Projektablauf in 8 Schritten

### 1. Punkte für Bildabfragen erzeugen

Ich habe zuerst Koordinatenpunkte in `EPSG:2056` erzeugt. Dafür gibt es `generate_points_grid.py`.

Warum dieser Schritt sinnvoll war:

- Ich brauchte eine systematische und reproduzierbare Art, Bildausschnitte zu sammeln.
- Die Punkte konnten entweder zufällig oder blockweise erzeugt werden.
- Die blockweise Erzeugung war praktisch, um räumlich zusammenhängende Gebiete zu beproben.

Warum ich diesen Ansatz gewählt habe:

- Das Projekt basiert auf räumlichen Daten. Deshalb war es sinnvoll, die Datengewinnung bereits räumlich strukturiert aufzubauen.
- Durch `region_id` und `block_id` konnte ich später Herkunft und räumliche Struktur der Daten weiterverwenden.

### 2. Swisstopo-Tiles herunterladen

Mit `download_tiles.py` habe ich aus den Punkten einzelne Bildkacheln über den WMS-Service von Swisstopo geladen.

Technisch relevant:

- Layer: `ch.swisstopo.swissimage`
- Tile-Grösse: `25 m`
- Bildgrösse: `256 x 256`
- Speicherung unter `data/unlabeled/<region_id>/`
- Metadaten zusätzlich in `data/labels.csv`

Warum ich das so gemacht habe:

- Ich wollte die Datenerfassung automatisieren, statt Bilder manuell zusammenzustellen.
- Die Dateinamen enthalten direkt die Koordinaten (`x_y.jpg`). Das ist später wichtig für Split und Nachvollziehbarkeit.
- Durch `labels.csv` bleibt der Zusammenhang zwischen Bild, Koordinaten und Region erhalten.

### 3. Bilder manuell labeln

Für das Labeling habe ich eine eigene Flask-Webapp gebaut.

Die Webapp macht Folgendes:

- zeigt offene Bilder aus `data/unlabeled`
- erlaubt das Labeln mit `1` oder `0`
- verschiebt die Bilder danach nach `data/y` oder `data/n`
- hält `labels.csv` synchron

Warum ich eine eigene Webapp gebaut habe:

- Das Labeling war ein zentraler Teil des Projekts. Mit einem einfachen UI konnte ich effizient und konsistent arbeiten.
- Ich wollte einen klaren Workflow: herunterladen -> labeln -> trainieren.
- Der Ansatz ist pragmatisch und reproduzierbar, ohne externe Tools zu brauchen.

Prüfungsrelevante Reflexion:

- Die Labels sind fachlich sinnvoll, aber manuelles Labeling kann Fehler enthalten.
- Gerade bei kleinen, schwer erkennbaren Fussgängerstreifen oder Grenzfällen besteht ein Risiko für Inkonsistenzen.

### 4. Datensatz für Deep Learning aufbauen

Der Hauptdatensatz umfasst laut meinem Lernzettel `107922` Bilder:

- Positiv: `3159`
- Negativ: `104763`

Das bedeutet:

- positive Klasse nur etwa `2.93 %`
- sehr starke Klassenimbalance

Warum das wichtig ist:

- Das war die grösste methodische Herausforderung des Projekts.
- Ein Modell könnte leicht sehr hohe Accuracy erreichen, wenn es fast immer `0` vorhersagt.

Meine wichtigste Datensatz-Entscheidung:

- Ich habe keinen zufälligen Split pro Bild gemacht.
- Stattdessen habe ich einen räumlichen Split verwendet.

## Warum räumlicher Split statt Zufallssplit?

Das ist einer der wichtigsten Prüfungspunkte.

Meine Begründung:

- Benachbarte Tiles sehen sich oft extrem ähnlich.
- Bei einem normalen Zufallssplit könnten fast identische Bildausschnitte gleichzeitig in Train und Test landen.
- Das wäre Data Leakage.
- Die Testleistung wäre dadurch künstlich zu gut.

Umsetzung:

- In `train.py` wird aus den Koordinaten eine räumliche Gruppe gebildet.
- Standardmässig geschieht das über räumliche Bins (`spatial_bin_size=1000`).
- Diese Gruppen werden dann auf `train`, `val`, `test` verteilt.

Verwendete Ratios:

- Train: `70 %`
- Validation: `15 %`
- Test: `15 %`

Warum ich diesen Ansatz gewählt habe:

- Die Evaluation soll realistisch sein.
- Das Modell soll nicht nur bekannte Nachbarbilder wiedererkennen, sondern auf neue Gebiete generalisieren.

## Unterrichtsthema: Introduction to Deep Learning

Was ich davon konkret umgesetzt habe:

- supervised learning
- Klassifikation statt Regression
- End-to-End-Lernproblem mit Eingabe Bild -> Ausgabe Klasse
- datengetriebener Ansatz statt handgebauter Bildregeln
- Fokus auf Generalisierung statt bloss auf Trainingsleistung
- klare Trennung zwischen Training, Validation und Test

Wichtige Begriffe aus der Vorlesung und Bezug zu meinem Projekt:

- `Artificial Intelligence` ist der Oberbegriff.
- `Machine Learning` lernt Muster aus Daten.
- `Deep Learning` ist der Teilbereich, der tiefe neuronale Netze verwendet.
- Mein Projekt gehört klar zu `Deep Learning`, weil ich ein CNN mit mehreren gelernten Schichten trainiert habe.

Warum mein Problem supervised learning ist:

- Ich habe gelabelte Beispiele mit `0` und `1`.
- Das Modell lernt also aus Eingabe-Ausgabe-Paaren.

Warum es eine Klassifikation und keine Regression ist:

- Die Ausgabe ist eine Klasse: Fussgängerstreifen vorhanden oder nicht vorhanden.
- Ich schätze keinen kontinuierlichen Wert wie Preis, Höhe oder Distanz.

Was Overfitting und Underfitting in meinem Kontext bedeuten:

- `Overfitting`: Das Modell lernt Trainingsmuster zu speziell und generalisiert schlecht auf neue Regionen.
- `Underfitting`: Das Modell ist zu schwach oder zu wenig trainiert und erkennt selbst im Training die relevanten Muster nicht gut.

Warum das zu meinem Projekt passt:

- Fussgängerstreifen haben zwar erkennbare Muster, aber die visuelle Erscheinung variiert stark durch Beleuchtung, Schatten, Auflösung, Farbe, Perspektive und Umgebung.
- Klassische harte Regeln wären hier deutlich weniger robust als ein trainiertes neuronales Netz.

## Unterrichtsthema: Tensors and Batches

Was ich umgesetzt habe:

- Klassen, Labels und Samples sauber getrennt
- Bilder werden als Tensoren verarbeitet.
- Das Training läuft in Mini-Batches.
- In `train.py` und `predict.py` werden `DataLoader` verwendet.

Bezug zu den Vorlesungsinhalten:

- Ein `Sample` ist bei mir ein einzelnes Tile.
- Die `Labels` sind `0` oder `1`.
- Die Daten liegen als mehrdimensionale Tensoren vor, also Bildhöhe, Bildbreite und Kanäle bzw. im Modell als Batch-Tensoren.
- Für Deep Learning ist dabei wichtig, dass Tensoren Achsen und Form haben und in Batches effizient weiterverarbeitet werden.

Warum das sinnvoll ist:

- Batches machen das Training auf GPU effizient.
- Sie stabilisieren die Optimierung gegenüber reinem Single-Sample-Training.
- Das ist Standard in Deep Learning und bei meinem Datensatz auch praktisch notwendig.

## Unterrichtsthema: Neural Networks Basics / Gears of Neural Networks

Was ich umgesetzt habe:

- Aufbau eines Modells aus Schichten
- binäre Klassifikation mit einem einzelnen Logit
- `sigmoid` zur Umwandlung in Wahrscheinlichkeiten
- `BCEWithLogitsLoss` als Loss-Funktion
- Training über Vorwärtsdurchlauf, Loss, Backpropagation und Optimizer-Schritt

Architektur-Idee aus der Vorlesung:

- Schichten transformieren Rohdaten schrittweise in nützlichere Repräsentationen.
- Genau das passiert in meinem Projekt auch: frühe Schichten lernen einfache Bildmuster, spätere Schichten kombinieren sie zu komplexeren Merkmalen.

Aktivierungsfunktionen im Projektkontext:

- Im selbst definierten `simple_cnn` verwende ich `ReLU`.
- Beim Hauptmodell `EfficientNet-B0` sind die Aktivierungs- und Blockstrukturen bereits in der Architektur integriert.

Warum der Threshold-Bezug aus der Vorlesung wichtig ist:

- Binäre Klassifikation bedeutet nicht nur "Wahrscheinlichkeit ausgeben", sondern auch eine Entscheidungsgrenze festlegen.
- Genau deshalb war mein Threshold-Tuning methodisch wichtig.

Warum ich das so gewählt habe:

- Für ein binäres Klassifikationsproblem ist ein einzelner Output mit `BCEWithLogitsLoss` methodisch passend und einfach.
- `BCEWithLogitsLoss` ist numerisch stabiler als erst `sigmoid` und danach BCE zu rechnen.

## Unterrichtsthema: Feed Forward Neural Networks

Was ich dazu in meinem Projekt sagen würde:

- Ein klassisches Feedforward-Netz war die konzeptionelle Basis.
- Für Bilder ist ein reines dicht verbundenes Netz aber unpraktisch, weil es räumliche Struktur schlecht ausnutzt und sehr viele Parameter braucht.
- Deshalb habe ich für die eigentliche Aufgabe ein CNN gewählt.

Warum CNN besser als reines Feedforward-Netz ist:

- lokale Muster werden effizienter erkannt
- Parameter werden geteilt
- räumliche Struktur des Bildes bleibt länger erhalten

## Unterrichtsthema: Optimization

Was ich umgesetzt habe:

- Optimizer: `AdamW`
- Learning Rate als zentrale Hyperparameter-Entscheidung
- Weight Decay für Regularisierung
- mehrere Trainingsläufe mit unterschiedlichen Strategien
- numerische Stabilitätsprüfungen auf nicht-finite Gewichte, Logits und Loss

Warum ich `AdamW` gewählt habe:

- `AdamW` funktioniert in der Praxis sehr gut für Bildklassifikation.
- Er ist robust und schneller nutzbar als sehr aufwendig getunte Verfahren.
- Für ein Projekt mit begrenzter Zeit war das ein pragmatischer und solider Entscheid.

Warum Hyperparameter wichtig waren:

- Schon kleine Änderungen bei Sampling, Loss-Gewichtung oder Threshold hatten grossen Einfluss.
- Das Projekt zeigt gut, dass Optimierung nicht nur heisst "mehr Epochen trainieren", sondern saubere Entscheidungen entlang der ganzen Trainingskonfiguration zu treffen.

## Unterrichtsthema: Convolutional Neural Networks

Das ist das zentrale Vorlesungsthema für mein Projekt.

Was ich umgesetzt habe:

- CNN-basierte Bildklassifikation
- unterstützt werden `simple_cnn`, `resnet18`, `efficientnet_b0`
- Hauptmodell war `efficientnet_b0`

Warum ein CNN geeignet ist:

- Fussgängerstreifen sind visuelle Muster mit lokalen Strukturen.
- CNNs sind stark darin, lokale Merkmale wie Linien, Kontraste, Kanten und Texturen zu erkennen.
- Genau solche Muster sind bei Fussgängerstreifen entscheidend.

Direkter Bezug zu den Vorlesungsbegriffen:

- `Feature Maps`: entstehen durch die Faltungen und enthalten aktivierte Bildmerkmale.
- `Max-Pooling`: reduziert räumliche Auflösung und verdichtet relevante Signale.
- `Flattening` bzw. ein Klassifikationskopf verbindet die gelernten Merkmale mit der Endentscheidung.

Was davon ich konkret umgesetzt habe:

- Im `simple_cnn` sieht man klassische CNN-Bausteine direkt: `Conv2d`, `ReLU`, `MaxPool2d`, danach ein Klassifikationskopf.
- Bei `EfficientNet-B0` sind diese Ideen in einer moderneren Architektur umgesetzt.

Warum `EfficientNet-B0` mein Hauptmodell war:

- sehr guter Kompromiss zwischen Modellstärke und Rechenaufwand
- stärker als ein kleines selbst gebautes CNN
- für Bildklassifikation sehr bewährt
- mit GPU gut trainierbar

Wie ich das Modell angepasst habe:

- Vortrainiertes Modell geladen
- Klassifikationskopf durch eine einzelne lineare Ausgabe ersetzt
- damit auf binäre Klassifikation angepasst

## Unterrichtsthema: Data Augmentation and Pretrained ConvNets

Dieses Thema habe ich sehr direkt umgesetzt.

### Pretraining

Ich habe vortrainierte ImageNet-Gewichte verwendet.

Warum:

- Mein Datensatz ist zwar gross, aber die positive Klasse ist selten.
- Vortraining liefert bereits allgemeine visuelle Merkmale.
- Dadurch wird das Training stabiler und oft daten-effizienter.

Bezug zur Vorlesung: Feature Extraction vs Fine Tuning

- `Feature Extraction` bedeutet, ein vortrainiertes Netz als Merkmalsextraktor zu nutzen.
- `Fine Tuning` bedeutet, ein vortrainiertes Modell weiter auf die eigene Aufgabe anzupassen.

Was ich praktisch gemacht habe:

- Ich habe vortrainierte Gewichte geladen und das Modell auf meine binäre Aufgabe angepasst.
- Damit nutze ich Transfer Learning.
- In meinem Setup steht nicht das Einfrieren einzelner Schichten im Vordergrund, sondern das praktische Weitertrainieren des angepassten Modells auf meinem Datensatz.

Warum dieser Ansatz sinnvoll war:

- Mein Problem ist visuell verwandt mit allgemeiner Bildverarbeitung, aber nicht identisch mit ImageNet.
- Deshalb ist Vorwissen aus dem Pretraining nützlich, das Modell muss sich aber trotzdem auf Luftbilder von Fussgängerstreifen spezialisieren.

### Standard-Augmentation

Im Standardmodus verwende ich:

- Resize
- Horizontal Flip
- Vertical Flip
- Rotationen
- Color Jitter
- Gaussian Blur

Warum Augmentation hier sinnvoll ist:

- Luftbilder variieren in Helligkeit, Farbe, Orientierung und Detailwirkung.
- Augmentation hilft dem Modell, robustere Merkmale zu lernen.
- Besonders bei der kleinen positiven Klasse erhöht das die effektive Datenvielfalt.

### Class-aware Augmentation

Ich habe zusätzlich einen Modus `class_aware` umgesetzt:

- positive Bilder werden stärker augmentiert als negative

Warum ich das getestet habe:

- Die seltene positive Klasse soll mehr visuelle Vielfalt sehen.
- Das ist ein gezielter Versuch, die Imbalance nicht nur über den Loss, sondern auch über die Datenrepräsentation anzugehen.

Reflexion:

- Der class-aware-Ansatz war fachlich plausibel, war aber nicht mein bester Lauf.
- Das ist in der Prüfung ein guter Punkt: Nicht jede gute Idee verbessert automatisch die Testleistung.

## Unterrichtsthema: Regularization

Reguläre Deep-Learning-Regularisierung habe ich über mehrere Bausteine umgesetzt:

- Data Augmentation
- Weight Decay
- räumlich fairer Split
- Threshold-Tuning auf Validation statt blind `0.5`
- Vergleich mehrerer Runs statt Vertrauen auf einen einzigen Zufallslauf

Wie `Dropout` in mein Projekt einzuordnen ist:

- `Dropout` ist laut Vorlesung eine klassische Methode gegen Overfitting, bei der während des Trainings zufällig Teile des Netzes deaktiviert werden.
- In meinem Projekt habe ich `Dropout` aber nicht als eigene zentrale Optimierungsstrategie eingesetzt oder systematisch verglichen.
- Im selbst definierten `simple_cnn` habe ich keine explizite `Dropout`-Schicht eingebaut.
- Mein Hauptfokus zur Generalisierung lag stattdessen auf Data Augmentation, Weight Decay, räumlichem Split und fairer Evaluation.

Warum ich Dropout nicht ins Zentrum gestellt habe:

- Das Hauptproblem meines Projekts war weniger ein zu grosses Modell allein, sondern vor allem Klassenimbalance und das Risiko von räumlichem Data Leakage.
- Deshalb waren Methoden, die direkt Datenverteilung und Generalisierung adressieren, für mich wichtiger.

Wie `Early Stopping` einzuordnen ist:

- Laut Vorlesung stoppt `Early Stopping` das Training, wenn sich die Validierungsleistung nicht weiter verbessert.
- In meinem Projekt habe ich nicht mit einem expliziten Early-Stopping-Callback gearbeitet.
- Stattdessen speichere ich das beste Modell anhand der Validierungsleistung und bewerte am Ende genau diesen besten Checkpoint.

Wie `Bagging` einzuordnen ist:

- `Bagging` bedeutet, mehrere Modelle zu trainieren und ihre Vorhersagen zu kombinieren.
- Das habe ich in meinem Projekt nicht als Ensemble-Lösung umgesetzt.
- Ich habe zwar mehrere Runs und Seeds verglichen, aber keine gemittelten Ensemble-Vorhersagen gebaut.

Wie `Batch Normalization` einzuordnen ist:

- `Batch Normalization` normalisiert Aktivierungen, um das Training stabiler zu machen.
- Ich habe keine eigene Batch-Normalization-Strategie als Experiment eingeführt.
- Bei vortrainierten Architekturen wie `EfficientNet-B0` sind gewisse Stabilitätsmechanismen bereits Teil der Architektur, aber ich habe Batch Normalization nicht als eigenen Untersuchungsfaktor begründet oder variiert.

Wie `Label Noise` bzw. fehlerhafte Labels einzuordnen ist:

- In den Folien kommt auch die Idee vor, mit fehlerhaften Labels oder verrauschten Targets umzugehen.
- Das ist für mein Projekt relevant, weil manuelles Labeling nie perfekt ist.
- Ich habe dafür aber keine eigene mathematische Label-Smoothing- oder Noise-Injection-Strategie implementiert.
- Stattdessen reflektiere ich das Thema über Datenqualität, manuelle Sichtung und Fehleranalyse.

Warum das wichtig ist:

- Regularisierung bedeutet nicht nur Dropout.
- In meinem Projekt war vor allem die Generalisierung auf neue Regionen entscheidend.
- Der räumliche Split ist inhaltlich fast der wichtigste Regularisierungs-/Fairnessmechanismus des ganzen Projekts.

## Unterrichtsthema: PyTorch Introduction

Dieses Thema habe ich vollständig praktisch umgesetzt.

Konkret:

- eigener `Dataset`
- `DataLoader`
- Training Loop
- Modellwahl über Parameter
- Device-Handling: `cpu`, `mps`, `cuda`
- Checkpointing
- separate Skripte für Training, Inferenz und Analyse

Warum PyTorch gut gepasst hat:

- Es gibt viel Kontrolle über die Pipeline.
- Für Experimente mit Imbalance, Threshold-Tuning, Grad-CAM und Fehleranalyse ist diese Flexibilität sehr hilfreich.

Expliziter Bezug zur Vorlesung:

- Im Gegensatz zu Keras habe ich Loss, Optimizer, Training Loop und Device-Handling explizit selbst definiert.
- Genau diese Feinsteuerung war für mein Projekt ein Vorteil.

## Unterrichtsthema: Keras Programming

Das habe ich in meinem Projekt nicht direkt verwendet, weil ich vollständig mit PyTorch gearbeitet habe.

Was ich dazu in der Prüfung sagen würde:

- Die Grundideen sind ähnlich: Modell definieren, trainieren, evaluieren.
- Ich habe mich für PyTorch entschieden, weil ich damit die Trainingslogik und Analysewerkzeuge flexibler anpassen konnte.

Bezug zu den Vorlesungsinhalten:

- Das Vorlesungsthema unterscheidet `Sequential Model` und `Functional API`.
- Mein Projekt braucht zwar kein Keras, aber konzeptionell ist es eher näher an einer flexiblen, expliziten API als an einem einfachen sequentiellen Standardmodell.
- Ich habe nur einen Input und einen Output, aber ich wollte volle Kontrolle über Loader, Loss, Threshold-Suche, Checkpointing und Analyse.

## Unterrichtsthema: Umgang mit Klassenimbalance

Das ist fachlich einer der wichtigsten Teile meines Projekts.

### Problem

- Positive Klasse ist sehr selten.
- Accuracy allein wäre dadurch irreführend.

### Getestete Strategien

1. `pos_weight`
2. `sampler`
3. `both`

### 1. `pos_weight`

Was es macht:

- Fehler auf positiven Beispielen werden im Loss stärker gewichtet.

Warum ich es getestet habe:

- Die seltene Klasse soll im Optimierungsziel mehr Gewicht bekommen.

### 2. `sampler`

Was es macht:

- Mit `WeightedRandomSampler` werden positive und negative Beispiele im Training ausgewogener gezogen.

Warum ich es getestet habe:

- Das Modell soll positives Material häufiger sehen und nicht durch die Mehrheit der Negativklasse dominiert werden.

### 3. `both`

Was es macht:

- kombiniert gewichteten Loss und balanciertes Sampling

Warum ich das getestet habe:

- Ich wollte prüfen, ob sich beide Mechanismen sinnvoll ergänzen.

### Meine Erkenntnis

- Die Imbalance-Behandlung hatte einen grossen Einfluss auf die Resultate.
- Das bestätigt, dass Datenverteilung oft wichtiger ist als kleine Architekturänderungen.

## Unterrichtsthema: Threshold-Tuning und Evaluation

Auch das ist ein sehr wichtiger Punkt für die Prüfung.

Ich habe den Threshold nicht einfach auf `0.5` fixiert, sondern auf dem Validation-Set optimiert.

Warum:

- Bei unausgeglichenen Daten ist `0.5` oft nicht der beste operative Schwellenwert.
- Je nach Ziel will man Precision und Recall anders gewichten.

Umsetzung:

- In `train.py` wird auf dem Validation-Set nach dem besten Threshold gesucht.
- Optimiert wird primär nach `F1`.
- optional konnte ich zusätzlich einen Mindest-Recall vorgeben.

Warum dieser Ansatz gut ist:

- Er verbindet Modelltraining mit einem realistischen Entscheidungsziel.
- Das ist methodisch sauberer als blind einen Standardthreshold zu übernehmen.

## Welche Metriken ich verwendet habe und warum

Ich habe nicht nur Accuracy betrachtet, sondern vor allem:

- Precision
- Recall
- F1
- PR-AUC
- zusätzlich ROC-Kurve und Confusion Matrix

Warum Accuracy nicht reicht:

- Bei nur rund `2.93 %` positiven Beispielen könnte ein Modell fast immer `0` sagen und trotzdem sehr hohe Accuracy haben.

Warum diese Metriken besser sind:

- `Precision`: Wie viele als positiv erkannte Tiles sind wirklich positiv?
- `Recall`: Wie viele echte Fussgängerstreifen finde ich?
- `F1`: guter Kompromiss zwischen Precision und Recall
- `PR-AUC`: besonders sinnvoll bei starker Klassenimbalance

## Meine wichtigsten Experimente

Ich habe nicht nur einen Lauf trainiert, sondern mehrere Varianten verglichen.

### Baseline: `a100_baseline_effb0`

- Modell: `efficientnet_b0`
- Pretrained: ja
- Imbalance: `pos_weight`
- Augmentation: `standard`

Testresultate:

- Accuracy: `0.9966`
- Precision: `0.9343`
- Recall: `0.9499`
- F1: `0.9420`
- PR-AUC: `0.9808`
- Threshold: `0.95`

Interpretation:

- sehr starker Ausgangslauf
- beste PR-AUC unter den verglichenen Hauptläufen

### Balanced Sampling: `a100_balanced_sampling`

- Imbalance: `both`
- Augmentation: `standard`

Testresultate:

- Accuracy: `0.9969`
- Precision: `0.9280`
- Recall: `0.9687`
- F1: `0.9479`
- PR-AUC: `0.9759`
- Threshold: `0.95`

Confusion Matrix:

- TP: `464`
- TN: `15898`
- FP: `36`
- FN: `15`

Interpretation:

- bester Hauptlauf nach `F1`
- sehr hoher Recall
- nur wenige übersehene positive Fälle

Warum dieser Lauf für mich wichtig ist:

- Er zeigt, dass die bewusste Imbalance-Behandlung die Praxisleistung verbessert.
- Für mein Problem war das wertvoller als nur maximale Precision.

### Sampler only: `sampler_only_effb0`

Analyse-Summary:

- Accuracy: `0.9970`
- Precision: `0.9352`
- Recall: `0.9645`
- F1: `0.9496`
- Threshold: `0.85`

Wichtige Beobachtung:

- Dieser Lauf ist im aktuellen Analyse-Summary sogar extrem stark.
- Er ist auch als Default-Modell hinterlegt:
  - Checkpoint: `artifacts/runs/server/sampler_only_effb0/best_model.pt`
  - Threshold: `0.85`

Das ist in der Prüfung gut erklärbar:

- Ich habe nicht einfach einen einzelnen "offiziellen Sieger", sondern mehrere starke Konfigurationen verglichen.
- Je nach ausgewählter Metrik oder finaler Analyse kann die bevorzugte Variante leicht variieren.

### Class-aware + both: `a100_classaware_both`

Testresultate:

- Accuracy: `0.9965`
- Precision: `0.9290`
- Recall: `0.9562`
- F1: `0.9424`
- Threshold: `0.95`

Interpretation:

- ebenfalls stark
- aber nicht besser als die besten Standard-Augmentation-Läufe

### Final image-size Variante: `final_sampler_only_effb0_img320`

Testresultate:

- Accuracy: `0.9961`
- Precision: `0.9462`
- Recall: `0.9186`
- F1: `0.9322`
- Threshold: `0.9`

Interpretation:

- höhere Precision
- aber mehr False Negatives
- für mein Ziel weniger attraktiv als ein recall-stärkerer Lauf

## Was ich aus den Seed-Runs gelernt habe

Ich habe zusätzliche Seed-Runs gerechnet, um die Stabilität der Resultate zu prüfen.

Beispiele:

- `both_effb0_seed123`: `F1 = 0.9409`
- `both_effb0_seed42`: `F1 = 0.9344`
- `both_effb0_seed777`: `F1 = 0.9378`
- `sampler_only_effb0_seed123`: `F1 = 0.9322`
- `sampler_only_effb0_seed42`: `F1 = 0.9396`
- `sampler_only_effb0_seed777`: `F1 = 0.9435`

Warum diese Runs wichtig sind:

- Sie zeigen, dass ich nicht nur einen glücklichen Einzelrun betrachtet habe.
- Ich habe überprüft, ob die Tendenzen auch über verschiedene Zufallsinitialisierungen sichtbar bleiben.

## Fehleranalyse und Interpretierbarkeit

Ich habe nicht bei den Metriken aufgehört, sondern das Modellverhalten genauer untersucht.

### Fehleranalyse

Mit `analyze_errors.py` habe ich False Positives und False Negatives systematisch ausgewertet.

Warum das wichtig ist:

- Gute Metriken allein sagen noch nicht, welche Bildtypen schwierig sind.
- Für die Prüfung zeigt das methodische Tiefe.

### Export schwieriger Fälle

Mit `export_error_tiles.py` kann ich FP/FN-Fälle für eine manuelle Sichtung exportieren.

Warum sinnvoll:

- qualitative Analyse ergänzt die quantitativen Metriken
- hilft, Datensatzprobleme und Modellschwächen besser zu verstehen

### Grad-CAM

Mit `grad_cam.py` und `generate_grad_cam_cases.py` habe ich Heatmaps erzeugt.

Warum ich das gemacht habe:

- Ich wollte sehen, auf welche Bildbereiche das Modell schaut.
- Das hilft zu prüfen, ob das Modell wirklich Fussgängerstreifen nutzt oder auf irrelevante Muster reagiert.

### Feature Maps

Mit `visualize_feature_maps.py` habe ich Aktivierungen einzelner Faltungsfilter visualisiert.

Warum das nützlich ist:

- Es zeigt, dass ich mich nicht nur für den Output, sondern auch für die internen Repräsentationen interessiere.

## Unterrichtsthema: RNN / LSTM

Dieses Thema habe ich in meinem Projekt nicht verwendet.

Begründung:

- Mein Problem ist statische Bildklassifikation.
- Es gibt keine zeitliche Sequenz, für die rekurrente Modelle nötig wären.

Wichtige Vorlesungsbegriffe und Einordnung:

- `RNNs` arbeiten mit Sequenzen und besitzen einen Zustand bzw. eine Art Gedächtnis.
- `LSTMs` lösen das Problem einfacher RNNs besser, weil sie mit Speicherzustand und Gates längere Abhängigkeiten verarbeiten können.
- Themen wie `vanishing gradients`, `long-term dependencies`, `recurrent dropout` oder `bidirectional` RNNs sind für Zeitreihen oder Text wichtig, aber nicht für einzelne Luftbild-Tiles.

Gute Prüfungsantwort:

- RNNs und LSTMs wären eher relevant für Text, Zeitreihen oder Bildfolgen, nicht für einzelne Luftbild-Tiles.

## Unterrichtsthema: 1D Convolutional Layers

Auch dieses Thema habe ich nicht direkt verwendet.

Begründung:

- 1D-Convolutions sind typisch für Sequenzen oder Signale.
- Mein Datentyp ist ein 2D-Bild, daher sind 2D-CNNs die passende Wahl.

Wichtige Vorlesungsbegriffe und Einordnung:

- 1D-CNNs erkennen lokale Muster in Sequenzen.
- Sie arbeiten mit 1D-Pooling und können für Zeitreihen oder Text günstiger als RNNs sein.
- Die Kombination aus 1D-CNN und RNN ist vor allem dann sinnvoll, wenn lange Sequenzen zuerst verdichtet und danach mit Ordnungssensitivität weiterverarbeitet werden sollen.
- Für mein Projekt war das nicht nötig, weil mein Input kein Zeitverlauf, sondern ein einzelnes 2D-Bild ist.

## Unterrichtsthema: Transformer

Transformer habe ich im Projekt nicht eingesetzt.

Begründung:

- Für meine konkrete Aufgabe war ein CNN-basierter Ansatz einfacher, effizienter und für die vorhandene Problemgrösse ausreichend stark.
- Ein Vision Transformer wäre ein möglicher Vergleich gewesen, war aber für den Projektumfang nicht notwendig.

Wichtige Vorlesungsbegriffe und Einordnung:

- `Transformer` wurden ursprünglich für `Sequence-to-Sequence`-Aufgaben wie maschinelle Übersetzung entwickelt.
- Zentrale Idee ist `Self-Attention`: Das Modell gewichtet, welche Teile einer Eingabe für ein bestimmtes Element besonders relevant sind.
- Weil Self-Attention von sich aus reihenfolgeblind ist, braucht man `Positional Encoding`, um Ordnungsinformation einzuspeisen.
- Klassische Transformer bestehen aus `Encoder`- und `Decoder`-Komponenten.

Warum ich das nicht verwendet habe:

- Mein Projekt hat keine Quell- und Zielsequenz wie bei Übersetzung.
- Ich brauchte keine textuelle Kontextmodellierung, sondern robuste 2D-Bildklassifikation.
- Ein Transformer-Ansatz wäre möglich gewesen, hätte aber die Komplexität erhöht, ohne für dieses Projekt zwingend nötig zu sein.

Gute Reflexion dazu:

- Ich habe mich bewusst für einen robusten und gut verstandenen CNN-Ansatz entschieden, statt unnötig komplex zu werden.

## Warum mein Gesamtansatz sinnvoll war

Wenn ich meinen Ansatz in der Prüfung verteidigen muss, würde ich die Wahl so begründen:

1. Die Daten sind Bilddaten, also ist ein CNN-Ansatz fachlich passend.
2. Die positive Klasse ist selten, deshalb brauchte ich gezielte Imbalance-Strategien.
3. Die Daten sind räumlich strukturiert, deshalb war ein räumlicher Split methodisch notwendig.
4. Pretraining und Augmentation waren sinnvoll, um robustere visuelle Merkmale zu lernen.
5. Die Evaluation musste über Precision, Recall, F1 und PR-AUC laufen, nicht nur über Accuracy.
6. Mit Fehleranalyse und Grad-CAM habe ich das Modell nicht nur trainiert, sondern auch kritisch untersucht.

## Stärken meines Projekts

- vollständige End-to-End-Pipeline
- eigener Datenerfassungs- und Labeling-Workflow
- methodisch sauberer räumlicher Split
- mehrere Imbalance-Strategien verglichen
- sinnvolle Metriken für unbalancierte Daten
- qualitative Fehleranalyse und Interpretierbarkeit
- reproduzierbare Trainings- und Server-Skripte

## Grenzen meines Projekts

- Labels können Fehler enthalten
- positive Klasse bleibt selten
- regionale Unterschiede können zu Bias führen
- binäre Klassifikation erkennt nur "vorhanden / nicht vorhanden"
- keine exakte Lokalisierung des Fussgängerstreifens im Bild

## Was ich in der Prüfung als wichtigste Erkenntnis sagen würde

Die wichtigste Erkenntnis war, dass gute Deep-Learning-Resultate nicht nur von der Architektur abhängen. In meinem Projekt waren vor allem vier Dinge entscheidend: ein sauber aufgebauter Datensatz, ein räumlich fairer Split, ein bewusster Umgang mit Klassenimbalance und eine passende Evaluation mit F1, Recall und PR-AUC. Genau diese Kombination hat die robusten Resultate ermöglicht.

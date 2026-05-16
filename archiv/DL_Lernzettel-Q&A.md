# DeepLearningKurs Beurteilung 2026: Lernzettel Frage -> Antwort

## Datensatz

### Was ist euer Datensatz?

Unser Datensatz besteht aus Swisstopo-Tiles, die binär darauf klassifiziert werden, ob ein Fussgängerstreifen vorhanden ist oder nicht.

### Wie sind die Labels definiert?

`1` bedeutet Fussgängerstreifen vorhanden, `0` bedeutet kein Fussgängerstreifen vorhanden.

### Wie läuft das Labeling ab?

Die Bilder werden zuerst nach `data/unlabeled/` geladen, im Web-UI manuell gelabelt und danach nach `data/y` oder `data/n` verschoben.

### Wie gross ist der Datensatz?

Die Hauptauswertung basiert auf `107922` Bildern.

### Wie verteilt sich der Datensatz auf Train, Validation und Test?

- Train: `75097`
- Validation: `16412`
- Test: `16413`

### Wie ist die Klassenverteilung?

- Positiv: `3159`
- Negativ: `104763`

Die positive Klasse macht nur rund `2.93 %` aus und ist damit selten.

### Was ist die wichtigste Eigenschaft des Datensatzes?

Der Datensatz ist stark unausgeglichen. Genau diese Klassenimbalance ist eines der Hauptprobleme des Projekts.

### Wie wurden die Daten aufgeteilt?

Nicht zufällig pro Bild, sondern räumlich über Gruppen auf Basis der Koordinaten in `EPSG:2056`.

### Warum habt ihr einen räumlichen Split verwendet?

Weil benachbarte Tiles sich sehr ähnlich sind. Ein normaler Zufallssplit würde leicht zu Data Leakage führen und die Leistung zu optimistisch erscheinen lassen.

### Welche Split-Ratios wurden verwendet?

- Train: `70 %`
- Validation: `15 %`
- Test: `15 %`

### Wie beurteilst du die Datenqualität?

Die Datenqualität ist insgesamt brauchbar und fachlich passend, aber nicht perfekt. Die grössten Risiken sind Klassenimbalance, mögliche Label-Fehler und regionale Unterschiede.

### Welche Biases oder Risiken gibt es?

- Klassenimbalance
- regionale Verzerrungen
- visuelle Unterschiede durch Licht, Schatten oder Perspektive
- mögliche Inkonsistenzen beim manuellen Labeling

## Modell

### Welches Modell war euer Hauptmodell?

`EfficientNet-B0` mit vortrainierten ImageNet-Gewichten.

### Welche Modelle unterstützt der Code zusätzlich?

- `simple_cnn`
- `resnet18`
- `efficientnet_b0`

### Warum habt ihr EfficientNet-B0 gewählt?

Weil es ein sehr guter Kompromiss aus Genauigkeit, Effizienz und Rechenaufwand ist und sich für Bildklassifikation sehr gut eignet.

### Warum habt ihr vortrainierte Gewichte verwendet?

Weil das Modell dadurch bereits allgemeine visuelle Merkmale kennt und mit begrenzten Daten stabiler und schneller trainiert werden kann.

### Wie wurde das Modell für eure Aufgabe angepasst?

Der Klassifikationskopf wurde auf eine binäre Ausgabe mit einem einzelnen Logit umgebaut.

### Warum ist ein CNN für diese Aufgabe geeignet?

Weil Fussgängerstreifen charakteristische visuelle Muster haben, die ein CNN über lokale Merkmale gut erkennen kann.

### Warum nicht einfach ein sehr kleines Modell?

Ein einfaches CNN ist zwar leichter, aber schwächer in der Merkmalsextraktion. EfficientNet-B0 liefert bei vertretbarem Aufwand deutlich stärkere Resultate.

## Optimierungen

### Was war die wichtigste technische Herausforderung?

Die starke Klassenimbalance, weil nur ein kleiner Teil der Bilder positiv ist.

### Welche Strategien gegen Klassenimbalance habt ihr getestet?

- `pos_weight`
- `sampler`
- `both`

### Was bedeutet `pos_weight`?

Positive Beispiele werden im Loss stärker gewichtet, damit Fehler auf der seltenen Klasse stärker bestraft werden.

### Was bedeutet `sampler`?

Mit `WeightedRandomSampler` werden positive und negative Beispiele im Training ausgewogener gezogen.

### Was bedeutet `both`?

`both` kombiniert gewichteten Loss und balanciertes Sampling.

### Welche Variante war am besten?

Nach `F1` war die Variante `both` am besten.

### Welche Augmentierungen wurden verwendet?

Im Standardmodus wurden unter anderem Resize, Flips, Rotationen, Color Jitter und Gaussian Blur verwendet.

### Was ist `class_aware` Augmentation?

Dabei werden positive Bilder stärker augmentiert als negative, um die seltene Klasse robuster zu machen.

### Warum ist Augmentierung hier sinnvoll?

Weil positive Beispiele selten sind und das Modell dadurch mehr Variabilität der positiven Klasse sieht.

### Wie wurde der Threshold gewählt?

Nicht fix auf `0.5`, sondern über die Validation nach dem besten `F1` optimiert.

### Warum ist Threshold-Tuning wichtig?

Weil bei unausgeglichenen Daten der beste operative Schwellenwert oft nicht bei `0.5` liegt.

### Welche weiteren methodischen Entscheidungen waren wichtig?

- räumlicher Split
- Validierung fehlerhafter Bilder
- Vergleich mehrerer Trainingsstrategien
- Seed-Runs zur Stabilitätsprüfung

## Resultate

### Welcher Lauf war der beste?

`a100_balanced_sampling`

### Welche Kennzahlen hatte der beste Lauf?

- Accuracy: `0.9969`
- Precision: `0.9280`
- Recall: `0.9687`
- F1: `0.9479`
- PR-AUC: `0.9759`
- Threshold: `0.95`

### Wie sieht die Konfusionsmatrix des besten Laufs aus?

- TP: `464`
- TN: `15898`
- FP: `36`
- FN: `15`

### Was bedeutet das fachlich?

Das Modell erkennt Fussgängerstreifen sehr zuverlässig und verpasst nur wenige positive Fälle, produziert aber noch einige False Positives.

### Welche anderen starken Läufe gab es?

`sampler_only_effb0` und `a100_baseline_effb0`.

### Was war beim Baseline-Lauf besonders gut?

Er hatte die beste `PR-AUC` mit `0.9808`.

### Warum ist Accuracy bei eurem Problem nicht ausreichend?

Weil der Datensatz stark unausgeglichen ist. Man kann eine hohe Accuracy erreichen, auch wenn die positive Klasse nicht gut erkannt wird.

### Welche Metriken sind wichtiger als Accuracy?

- Precision
- Recall
- F1
- PR-AUC

### Warum ist Recall bei euch wichtig?

Weil verpasste Fussgängerstreifen inhaltlich problematisch sind. Ein hoher Recall bedeutet, dass nur wenige positive Fälle übersehen werden.

### Warum ist PR-AUC hier sinnvoll?

Weil PR-AUC bei stark unausgeglichenen Datensätzen aussagekräftiger ist als viele klassische Metriken.

### Wie würdest du die Resultate zusammenfassen?

Die Resultate sind stark und überzeugend, vor allem weil sie auf einem räumlich getrennten Testsplit erzielt wurden. Das macht die Bewertung realistischer.

## Persönliche Erkenntnisse

### Was war eure wichtigste Erkenntnis?

Dass gute Resultate nicht nur von der Modellarchitektur abhängen, sondern stark von Datenaufbereitung, Split-Strategie, Imbalance-Behandlung und Evaluation.

### Was war überraschend?

Dass kleine methodische Änderungen wie Sampling, gewichteter Loss oder Threshold-Tuning einen sehr grossen Effekt auf die Qualität haben können.

### Was habt ihr über Evaluation gelernt?

Dass Accuracy bei unausgeglichenen Daten zu optimistisch sein kann und F1, Recall und PR-AUC deutlich aussagekräftiger sind.

### Was habt ihr über den Datensatz gelernt?

Dass räumlich benachbarte Bilder ein echtes Risiko für Data Leakage sind und ein räumlicher Split deshalb methodisch sauberer ist.

### Was würdet ihr als Nächstes verbessern?

- mehr positive Beispiele sammeln
- Fehleranalyse systematischer auswerten
- regionale Vielfalt erhöhen
- modernere Modelle oder Ensembles testen
- Threshold noch gezielter an das Anwendungsziel anpassen

### Wo liegen die Grenzen eures Ansatzes?

Das Modell ist von den Labels abhängig, es bleibt fehleranfällig bei Grenzfällen und es macht nur binäre Klassifikation, aber keine genaue Lokalisierung.

### Was ist eine gute Abschlussantwort in der Prüfung?

Die wichtigste Erkenntnis war, dass robuste Resultate aus dem Zusammenspiel von sauberem Datensatz, räumlich fairem Split, sinnvoller Imbalance-Behandlung und passender Evaluation entstehen, nicht nur aus einem starken Modell.

## 30-Sekunden-Version

Wenn du nur sehr kurz antworten kannst:

Wir haben einen stark unausgeglichenen Swisstopo-Bilddatensatz für die binäre Erkennung von Fussgängerstreifen verwendet. Das Hauptmodell war EfficientNet-B0 mit vortrainierten Gewichten. Die wichtigsten Optimierungen waren räumlicher Split, balanciertes Sampling, gewichteter Loss und Threshold-Tuning. Der beste Lauf erreichte `F1 = 0.9479` bei hohem Recall auf einem räumlich getrennten Testsplit. Die wichtigste Erkenntnis war, dass Datenaufbereitung und Evaluation genauso wichtig sind wie die Modellarchitektur.

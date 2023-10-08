## Regressionsmodell bewerten / Bestimmtheitsmaß R<sup>2</sup>

[Merkblatt_Bestimmtheitsmaß](pdfs/Merkblatt_Bestimmtheitsmaß.pdf)

Dabei geht es darum wie man die verschiedenen Modelle bewerten kann.
Im eindimensionalen konnte man das Ergebnis plotten, aber bei mehreren Parametern (Features) braucht man ein eigenes Bewertungsschema (ein mathematisches Maß wie man die Güte oder Qualität bewertet)

Diese Güte wird **Bestimmtheitsmaß** oder **R<sup>2</sup>** genannt.

Idee für R<sup>2</sup>-Score
- Wir berechnen den "Fehler" für ein richtig schlechtes Modell
- Wie gut ist unser Modell besser, als das richtig schlechte Modell?

Aber: Was ist ein richtig schlechtes Modell?
- Ein Modell, welches unsere Eingabedaten gar nicht wirklich betrachtet
- Aber es darf auch nicht zu sehr daneben liegen
- => Wir schätzen immer den Durchschnitt von den Daten, die wir gelernt haben

Beispiel:<br>
![Bestimmtheitsmaß](pictures/Bestimmtheitsmaß01.jpg)

Man bestimmt unabhängig vom der x-Achse (Carat) einen Mittelwert der y-Achse (Preis). Dieses schlechte Modell ist vollkommen unabhängig von den Eingabeparametern.

Zudem soll gelten:
- Man hat dann 2 Modelle (z.B. Lineares Regressionsmodell und Durchschnittsmodell)
- Wert 1, wenn unser Modell die Daten perfekt beschreibt
- Wert 0, wenn unser Modell die Daten genauso gut wie im Durchschnittsmodell beschreibt
- Wert von unter 0, wenn es noch schlechter als der Durchschnitt ist

Man kann dies auch in eine mathematische Formel gießen:<br>
![Bestimmtheitsmaßformel](pictures/Bestimmtheitsmaß02.jpg)

```math
R^2 = \sum_{i=1}^{n} \left(\hat{y_i} - \bar{y}\right)^2/\sum_{i=1}^{n} \left(y_i - \bar{y}\right)^2
```

Man muss dieses Bestimmtheitsmaß nicht selbst berechnen - bereits implementiert.

## Bestimmtheitsmaß R<sup>2</sup> in Python

```python
# Matplotlib config
%matplotlib inline
%config InlineBackend.figure_formats = ['svg']
%config InlineBackend.rc = {'figure.figsize': (5.0, 3.0)}

import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("../data/Diamonds/diamonds.csv.bz2")

df.head()
```

```python
from sklearn.linear_model import LinearRegression

xs = df["carat"].to_numpy().reshape(-1, 1)
ys = df["price"]

model = LinearRegression()
model.fit(xs, ys)
```

- ys sind die echten Preisdaten
- y_pred sind die mittels Vorhersagen (model.predict) gewonnenen Daten

```python
from sklearn.metrics import r2_score

#r2_score?
y_pred = model.predict(xs)
r2_score(ys, y_pred)

0.8493305264354858
```

Da es sich um eine so essetielle Metrik handelt gibt es dafür auch eine fertige Funktion (**model.score**). Achtung: es müssen jedoch andere Werte übergeben werden.
```python
model.score?
print(model.score(xs, ys))

0.8493305264354858
```

### Modelle vergleichen

```python
from sklearn.linear_model import LinearRegression

xs = df["carat"].to_numpy().reshape(-1, 1)
ys = df["price"]

model = LinearRegression()
model.fit(xs, ys)

print(model.score(xs, ys))

0.8493305264354857
```

Ein Modell mit 2 Features:

```python
from sklearn.linear_model import LinearRegression

#print(df[["carat", "x"]].to_numpy())
xs = df[["carat", "x"]]
ys = df["price"]

model = LinearRegression()
model.fit(xs, ys)

print(model.score(xs, ys))

0.8534314749939833
```

Wir gewinnen ein etwas höheres Bestimmtheitsmaß

Nachteil: für die Vorhersagen muss ich beim zweiten Modell auch 2 Parameter angeben

Problem: Wir berechnen das Bestimmtheitsmaß nicht auf irgendwelche zukünftigen Daten, sondern auf Daten mit denen wir bereits das Modell trainiert haben.

### Daten mit test/train aufteilen

Uns interessiert nicht das Bestimmtheitsmaß auf Daten die das Modell bereits gesehen hat, sondern auf Daten die das Modell noch nicht gesehen hat.

Die Idee dahinter ist, dass man das Modell nur mit einem Teil der Daten trainiert und mit dem Rest der Daten testet

Man kann dies "zu Fuss" realisieren
```python
# Test / Train
print(len(df))
53940

df_train = df.iloc[:40000] # die ersten 40000-Zeilen sind Trainingsdaten
df_test = df.iloc[40000:]  # die Testdaten sind ab der 40000sten Zeile
```
```python
from sklearn.linear_model import LinearRegression

X_train = df_train["carat"].to_numpy().reshape(-1, 1)
y_train = df_train["price"]

model = LinearRegression()
model.fit(X_train, y_train)

X_test = df_test["carat"].to_numpy().reshape(-1, 1)
y_test = df_test["price"]

print(model.score(X_test, y_test))

-1.0517239092305672
```

Warum ergibt dies so einen schlechten Score (Bestimmtheitsmaß)?
- die Daten sind nach aufsteigendem Preis sortiert
- damit sind die günstigen Diamanten am Anfang und die teuersten am Ende
- wir haben somit nur mit den günstigeren Diamanten (die ersten 40000) trainiert
- aber mit den teuren Diamanten getestet

Lösung: Daten neu sortieren
```python
from sklearn.linear_model import LinearRegression

df = df.sample(frac = 1) # mischt die kompletten Daten
df_train = df.iloc[:40000] # iloc verwenden, da loc den falschen Index liefert
df_test = df.iloc[40000:]

X_train = df_train["carat"].to_numpy().reshape(-1, 1)
y_train = df_train["price"]

model = LinearRegression()
model.fit(X_train, y_train)

X_test = df_test["carat"].to_numpy().reshape(-1, 1)
y_test = df_test["price"]

print(model.score(X_test, y_test))

0.8452169386942134
```

### Daten mit test_train_split aufteilen

Aufteilung der Trainings- und Testdaten mithilfe einer Funktion (**test_train_split()**)

```python
from sklearn.model_selection import train_test_split

#train_test_split?
train, test = train_test_split([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], train_size = 0.75)
print(train)
print(test)

[4, 10, 9, 6, 7, 5, 1]
[8, 3, 2]
```

Man kann der Funktion auch gleich mehrere Arrays übergeben. Die Aufteilung über beide Listen erfolgt identisch.
```python
train_test_split([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [101, 102, 103, 104, 105, 106, 107, 108, 109, 1010])

[[6, 1, 4, 7, 5, 8, 2],
 [9, 3, 10],
 [106, 101, 104, 107, 105, 108, 102],
 [109, 103, 1010]]
```

```python
X = df[["carat"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state = 42)
```
train_test_split()
- train_size bestimmt die Aufteilung in Train- und Testdaten
- random_state mit irgendeiner Zahl initialisieren
- liefert eine verschachtelte Liste zurück, die man aber normalerweise gleich den entsprechenden Variablen zuweist

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

0.8452169386942134
```

Man kann die Aufteilung leicht auf mehrere Spalten erweitern und bekommt leicht ein anderes Bestimmtheitsmaß
```python
X = df[["carat", "x"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state = 42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

0.8532274320378628
```

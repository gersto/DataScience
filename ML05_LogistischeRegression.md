## Einführung: Klassifizierung

Bei der logistischen Regression geht es um Klassifizierungsmöglichkeiten

![Klassifizierung](pictures/LogistischeRegression01.jpg)

Klassifizierung ist wenn wir nicht eine Zahl vorhersagen wollen, sondern eine Kategorie

Ein gutes Beispiel ist die Bilderkennung. Wir wollen im oberen Bild zwar Zahlen aus Bildern
erkennen, diese Zahlenbilder sollen aber jeweils der richtigen Kategorie zugeordnet werden
(hier 10 verschieden Klassen - Zwischenwerte haben keine Bedeutung)

Wo liegt der Unterschied zur Regression?

![Regression](pictures/LogistischeRegression02.jpg)

Bei der Regression haben wir versucht Zahlenwerte vorherzusagen

Bei der Klassifizierung sind immer nur bestimmte Klassen vorherzusagen<br>
Beispiele:
- Ist Pilz essbar? Ja / Nein
- Wird der Patient zum Arzttermin erscheinen? Ja / Nein
- Brustkrebs: Ja / Nein
- Tumor: Gutartig / Bösartig

Aber auch mehrere Klassen:
- Welche Ziffer (0-9) ist auf einem Bild zu sehen?
- Welche Frucht ist auf einem Bild zu sehen?

Die logistische Regression baut zwar auf der linearen Regression auf, hat aber noch
Zusätze, sodass wir damit z.B. binäre Klassifizierungen durchführen können.

## Einführung: Logistische Regression

**Idee:**
- Wir betrachten eine Ja/Nein-Frage
- Jetzt können wir definieren: (Wertebereich von möglichen Ergebnissen auf den Zahlenberich 0 oder 1 reduzieren)
  - 0 = Nein
  - 1 = Ja
- Mathematisch ausgedrückt
  - Wir nehmen die lineare Regression
  - und wandeln die Ausgabe in den Wertebereich 0-1 um!
  - dann kann unser Modell ausschließlich Werte zwischen 0 und 1 ausgeben!

![Logistische Regression](pictures/LogistischeRegression03.jpg)

![Logistische Regression](pictures/LogistischeRegression04.jpg)

Man wendet auf die Werte die sogenannte Sigmoid-Funktion an:

![Sigmoid-Funktion](pictures/LogistischeRegression05.jpg)

## Die Sigmoid-Funktion

```python
# Matplotlib config
%matplotlib inline
%config InlineBackend.figure_formats = ['svg']
%config InlineBackend.rc = {'figure.figsize': (5.0, 4.0)}
import numpy as np
import seaborn as sns

sns.set()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

a = 1
b = 1

xs = np.arange(-10, 10, 0.1)
ys = sigmoid(a * xs + b)

sns.lineplot(x = xs, y = ys);
```
![Sigmoid-Funktion](pictures/LogistischeRegression06.jpg)

Die Idee der Sigmoid-Funktion ist den reellen Zahlenbereich auf die Werte zwischen 0 und 1 abzubilden.

Die Geradengleichung (a * xs + b) der linearen Regression auf die Werte zwischen 0 und 1 abzubilden - logistische Regression

Wie wirken sich die Parameter a und b aus:
- a = 10, b = 0 (a bestimmt die Steilheit)
![Sigmoid-Funktion](pictures/LogistischeRegression07.jpg)
- a = 2, b = 5 (b verschiebt die Kurve)
![Sigmoid-Funktion](pictures/LogistischeRegression08.jpg)

Man kann die Sigmoid-Kurve auch als **Wahrscheinlichkeitskurve** auffassen.

Es werden die Parameter a und b gelernt.


## Logistische Regression

```python
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("data/Diabetes/diabetes.csv")
df.head()
```

Zur Ausgabe der Anzahl der Einträge
```python
print(len(df))

768
```

Die Daten sind Echtdaten, kommen aber nur aus Indien und auch da nur von einem bestimmten Teil und nur von weiblichen Patienten (Hintergrundinformation)

Wie trainiert man jetzt die logistische Regression (bereits in sklearn implementiert)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = df[["BMI", "Age"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)

model = LogisticRegression()
model.fit(X_train, y_train)

# print(model.predict(X_test))
#[0 0 0 0 0 0 0 0 0 1 1 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 1 0 0 0 1 0 1 0 0 0 0 1 1 0 0 1 0 1 0 0 0 1 0 1 0 0 0 0 0 0 1 0 1 0 0 0
# 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 1 1 0 1 0 0 1 0 1 0 0 0 0 0 0 1
# 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 1 0
# 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 1 0 0]

print(model.score(X_test, y_test))

0.6666666666666666
```

Wie wird nun dieser Score berechnet? (anders als bei der linearen Regression)

```python
# in y_test_pred steht die Vorhersage ob der Patient Diabetes hat oder nicht
# in X_test steht ob der Patient wirklich Diabetes hat oder nicht (Testdaten)
y_test_pred = model.predict(X_test)

print(np.mean(y_test_pred == y_test))

0.6666666666666666
```

```python
y_test_pred == y_test

143    False
675    False
44      True
243    False
332     True
       ...  
111     True
455    False
586     True
465     True
477     True
Name: Outcome, Length: 192, dtype: bool
```

## Logistische Regression (Features auswählen)

Mit der Funktion model.predict_proba() kann man Vorhersagen auf Wahrscheinlichkeitsbasis machen
```python
# print(model.classes_)
model.predict_proba(X_test)

array([[0.45404599, 0.54595401],
       [0.55037434, 0.44962566],
       [0.75003697, 0.24996303],
       [0.65112378, 0.34887622],
       [0.23100323, 0.76899677],
       [0.85357436, 0.14642564],
       [0.57807064, 0.42192936],
       [0.54202819, 0.45797181],
       [0.73904466, 0.26095534],
       [0.87800612, 0.12199388],
       ...
```
Man bekommt für jeden Testdatensatz die Wahrscheinlichkeiten für die jeweiligen Klassen (0 oder 1).

Dies Wahrscheinlichkeiten haben manchmal eine höhere Aussagekraft.

Um jetzt nur die Wahrscheinlichkeit ob ein Patient Diabetes hat oder nicht - nur die zweite Spalte ausgeben.
```python
model.predict_proba(X_test)[:, 1]

array([0.54595401, 0.44962566, 0.24996303, 0.34887622, 0.76899677,
       0.14642564, 0.42192936, 0.45797181, 0.26095534, 0.12199388,
       0.0798396 , 0.47763782, 0.72010356, 0.40119139, 0.56838914,
       0.17839958, 0.43217499, 0.19626865, 0.21919905, 0.45422353,
       0.1638898 , 0.44392305, 0.79292102, 0.77241705, 0.15521775,
       ...
```

Eigene Daten testen (sollen natürlich aus der gleichen Population - Indien, Frauen, ... - kommen)<br>
Wir bereiten die entsprechenden Daten auf:
```python
X_pred = pd.DataFrame([
    [28, 40]
], columns = ["BMI", "Age"])

X_pred.head()

   BMI 	Age
0 	28 	40
```

und berechnen danach die Vorhersage und die Wahrscheinlichkeiten
```python
print(model.predict(X_pred))
print(model.predict_proba(X_pred))

[0]
[[0.70325763 0.29674237]]
```

## Logistische Regression - Aufgabe

Das bisherige Modell, welches nur auf dem BMI und dem Age beruht, zu verbessern (andere Parameter zu berücksichtigen).

Zuerst erweitern wir um den Glucose-Parameter

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = df[["BMI", "Age", "Glucose"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)

model = LogisticRegression()
model.fit(X_train, y_train)

# print(model.predict(X_test))
print(model.score(X_test, y_test))

0.7708333333333334
```

Ist schon eine deutliche Verbesserung.

Nun auch noch mit dem Blutdruck

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = df[["BMI", "Age", "Glucose", "BloodPressure"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)

model = LogisticRegression()
model.fit(X_train, y_train)

# print(model.predict(X_test))
print(model.score(X_test, y_test))

0.796875
```

Jetzt auch noch die Anzahl der Schwangerschaften berücksichtigt (bringt meist aber keine Verbesserung)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = df[["BMI", "Age", "Glucose", "BloodPressure", "Pregnancies"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)

model = LogisticRegression()
model.fit(X_train, y_train)

# print(model.predict(X_test))
print(model.score(X_test, y_test))

0.8125
```


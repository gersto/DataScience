# Neuronale Netze (ein einzelnes Neuron)

## Einführung: Neuronale Netze

- Deep Lerning beschreibt Machine Learning mit neuronalen Netzen
- Es ist aber weiterhin Machine Learning
  - Daten => Modell => Vorhersagen! (meist mit komplexeren Daten und Modellen)
  - Train / Test
  - Klassifizierung / Regression (meist handelt es sich aber um Klassifizierungsprobleme)
  - Overfitting / Underfitting ist auch hier ein Problem

Welches Tool?
- Sklearn **nicht**
  - Bietet eine einfache Implementierung für Neuronale Netze
  - Diese unterstützt aber viele Features nicht (z.B. CNN - Convolutional neuronal networks für die Bilderkennung sind in sklearn nicht implementiert)
  - keine GPU-Unterstützung (für komplexe Netzwerke)
- Tensorflow / Keras
  - Quasi der Standard für Neuronale Netze & Deep Learning
  - Unterstützt alle wesentlichen Features


Was sind Neuronale Netze?

Analogie mit Nervenzellen:<br>
![NeuronaleNetze01](pictures/NeuronaleNetze01.jpg)

Analogie ist wahrscheinlich nicht ganz richtig. NN ist viel angewandte Mathematik.

Mathematik:<br>
![NeuronaleNetze02](pictures/NeuronaleNetze02.jpg)

entspricht der linearen Regression, aber in einer etwas anderen Schreibweise

Eingabeknoten (X1, X2, X3) mit Gewichten (w1, w2, w3) versehen, zusätzlich mit einem Bias-Knoten (b) versehen.
Das Neuron macht nichts anderes als die Summe (entspricht der Ausgabe y) zu berechnen.<br>
Man versucht die Gewichte und das Bias zu optimieren, damit das Neuron die entsprechende Vorhersage macht

## Wie lernt ein einzelnes Neuron

- Die idealen Gewichte vom Neuron können in einem Neuronalen Netz nicht direkt durch eine mathematische Formel ermittelt werden
- Wir müssen uns ihnen daher Schritt für Schritt annähern
- Idee:
  - Wir initialisieren die Gewichte zufällig
  - Wir schauen uns dann die Datensätze einzeln an
  - Und dann verändern wir die Gewichte ein wenig in die "richtige Richtung"

![NeuronaleNetze03](pictures/NeuronaleNetze03.jpg)

z.B. ein Neuron, welches gerade als Vorhersage gerade die 2 liefert, in Wirklichkeit hätten wir bei den Daten aber eine 6
stehen. Man kann jetzt an den Gewichten (w1, w2, w3 und b) drehen um die Vorhersage näher an die 6 zu bringen. Dieses drehen
muss ntürlich relativ oft geschehen. Man nähert sich also relativ langsam an den wirklichen Wert an. Es kann aber auch sein,
dass man die Gewichte in die andere Richtung drehen muss. Die Gewichte hängen auch von den Eingabewerten ab.

## Ein einzelnes Neuron

```python

```

```python
# Matplotlib config
%matplotlib inline
%config InlineBackend.figure_formats = ['svg']
%config InlineBackend.rc = {'figure.figsize': (5.0, 3.0)}

import numpy as np
import pandas as pd
import seaborn as sns

# Tensorflow laden
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

Wir verwenden wieder die Daten von Diamanten
```python
df = pd.read_csv("../data/Diamonds/diamonds.csv.bz2")

df.head()
```

Ziel ist wieder auf Basis des Gewichts den Preis der Diamanten vorherzusagen.

Basis ist ein einfaches sequentielles Modell mit einer einzigen Ebene mit einem einzigen Knoten (ein Neuron)

```python
model = keras.Sequential([
    layers.Dense(1, name = "neuron")
])

# Testen mit 2 Datensätzen
v = tf.Variable([
    [1],
    [3]
])

#print(v)
#print(v.numpy())
print(model(v))

tf.Tensor(
[[1.3962232]
 [4.1886697]], shape=(2, 1), dtype=float32)
```

Jedesmal wenn diese Berechnung angestoßen wird kommen andere Werte heraus -> liegt an der zufälligen Initialisierung
der Startwerte.

Bei einem Aufruf mit mehreren Eingabespalten würden wir einen InvalidArgumentError bekommen
```python
v2 = tf.Variable([
    [1, 3],
    [3, 7]
])

print(model(v2))

InvalidArgumentError                      Traceback (most recent call last)
<ipython-input-38-00ff387867e8> in <module>
      4 ])
      5 
----> 6 print(model(v2))
...
```

Das Modell merkt sich beim ersten mal Starten wie viele Eingangsparameter verwendet wurden.<br>
Falls man das Modell jetzt nochmals neu initialisiert und mit 2 Eingangsparamatern testet funktioniert dies.

```python
model = keras.Sequential([
    layers.Dense(1, name = "neuron")
])

# Testen mit 2 Datensätzen mit jeweils 2 Paramatern
v2 = tf.Variable([
    [1, 3],
    [3, 7]
])

print(model(v2))

tf.Tensor(
[[3.6744065]
 [8.290762]], shape=(2, 1), dtype=float32)
```

Zusammengefasst:<br>
Das einzelne Neuron wird durch das Model erzeugt und die Anzahl der Eingabedaten wird beim ersten Starten mit Eingabewerten bestimmt.

## Neuron trainieren

Zuerst die X und y-Daten vorbereiten
```python
X = df[["carat"]]
y = df["price"]
```

Mit model.weights erhält man die Gewichte
```python
print(model.weights)
```

Damit das Model die Gewichte lernt und nicht nur zufällig initialisiert, muss das Model compiliert werden
```python
model.compile(
    optimizer = keras.optimizers.RMSprop(),
    loss = keras.losses.MeanSquaredError()
)
```

Bevor wir das Model fitten können müssen noch die X-Daten auf float32 umgewandelt werden.<br>
Wir sehen uns mal die X-Daten an
```python
#print(X)
#print(X.to_numpy())
print(X.to_numpy().dtype)
```

Neuronales Netz will mit float32 rechnen -> schneller<br>
Das folgende fitten dauert einige Zeit
```python
model.fit(X.astype(np.float32), y, batch_size = 1, epochs = 1)

53940/53940 [==============================] - 56s 1ms/step - loss: 30607938.0000
<tensorflow.psython.kera.callbacks.History at 0x7fd784452590>
```

Danach kann man wieder ein predict durchführen
```python
model.predict(np.array([
    [0.1]
]))

array([[74.41793]], dtype=float32)
```

Das einzelne Neuron hat zwar gelernt, aber noch nicht so richtig gut:
- bis jetzt nur über eine Epoche trainiert (wir legen dem Neuron alle Daten vor und drehen dann
  an den Gewichten - dieses ansehen der Daten und drehen der Gewichte sollte mehrfach erfolgen) --> epochs-Paramter ändern
  ```python
  model.fit(X.astype(np.float32), y, batch_size = 1, epochs = 10)
  ```
  dauert sehr lange, ist also relativ ineffizient
- Änderung der batch_size --> es werden dann viele Ergebnisse auf einmal berechnet und dann nur einmal an den Gewichten gedreht
  (sollte auch von den Prozessoren - Paraellelverarbeitung - unterstützt werden) --> geht dann viel schneller
  ```python
  model.fit(X.astype(np.float32), y, batch_size = 64, epochs = 10)
  ```

Dieses neuronale Netz macht aber noch immer keine guten Vorhersagen - der berechnete **loss** über den MeanSquareError (auch in der Ausgabe)
ändert sich nicht sehr. D.h. wir drehen nur sehr schwach an unseren Geichten, d.h. wir bräuchten sehr, sehr viele Epochen um den Fehler auszugleichen.<br>
Dafür gibt es die Lernrate (gibt an wie stark wir an den Gewichten drehen sollen) - kann über den optimizer als Parameter beeinflusst werden. 
```python
model.compile(
    optimizer = keras.optimizers.RMSprop(learning_rate=0.5),
    loss = keras.losses.MeanSquaredError()
)

model.fit(X.astype(np.float32), y, batch_size = 64, epochs = 10)

model.predict(np.array([
    [0.7]
]))
```
Bei zu großer Learning-Rate (z.B. 100) ergibt sich auch kein gutes loss-Verhalten

Ausgabe des Bestimmtheitsmaß:
```python
from sklearn.metrics import r2_score

y_pred = model.predict(X.astype(np.float32))

print(r2_score(y, y_pred))

0.8493171091838858
```

## Die Aktivierungsfunktion

Idee ist ähnlich der logistischen Regression:
- Damit das Neuron eine Ausgabe zwischen 0 und 1 ausgibt, wird die Ausgabe durch eine Aktivierungsfunktion geleitet
- Dadurch können wir eine Klassifizierung durchführen (ja/nein Anwort)
- Dies wird aber später auch noch wichtig sein, um mehrere Neuronen hintereinander schalten zu können ("Neuronales Netz") - dort zwingend notwendig

![NeuronaleNetze04](pictures/NeuronaleNetze04.jpg)

- Unser Neuron führt jetzt eine Klassifizierung durch (Ja/Nein)
- Aber:
  - Das Trainieren würde so noch recht lange dauern / nicht richtig funktionieren
  - Wir benötigen noch eine andere Loss-Funktion!
  - die sogenannte BinaryCrossentropy!

Aktivierungsfunktion im Code mit Diabetesdaten:

```python
# Matplotlib config
%matplotlib inline
%config InlineBackend.figure_formats = ['svg']
%config InlineBackend.rc = {'figure.figsize': (5.0, 3.0)}

import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("../data/Diabetes/diabetes.csv")
df.head()
```

Auf Basis des BMI und des Alters möchte man die Wahrscheinlichkeit (Outcome) für Diabetes schätzen
```python
X = df[["BMI", "Age"]]
y = df["Outcome"]
```

Man könnte auch noch train_test_split ausführen - hier nicht<br>
Wichtig ist der activation-Parameter in layers.Dense auf sigmoid zu setzen (standardmäßig auf linear)
```python
# Tensorflow laden
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(1, name = "neuron", activation = "sigmoid")
])

model.compile(
    optimizer = keras.optimizers.RMSprop(0.01),
    loss = keras.losses.BinaryCrossentropy()
)

model.fit(X.astype(np.float32), y, batch_size = 64, epochs = 100)
```
```python
model.predict(X)
```
Umwandlung in True/False-Werte (Wahrscheinlichkeiten > 0.5

```python
#(model.predict(X) > 0.5).shape
#y.to_numpy().shape
#(model.predict(X) > 0.5).ravel().shape
# reshape(-1) entspricht dem ravel()
#(model.predict(X) > 0.5).reshape(-1).shape

np.mean((model.predict(X) > 0.5).ravel() == y)

0.6627604166666666  --> entspricht einer Genauigkeit von 66%
```

```python
model.predict(np.array([
    [25, 30]
]))

array([[0.24642956]], dtype=float32)
```

# Neuronale Netze (vom Neuron zm Netz)

## Vom Neuron zum Netz (Backpropagation)

- Idee:
  - Wir "schieben" eine Schicht von Neuronen zwischen die Eingabe und unseren Ausgabe-Knoten
  - Hierbei sind alle Knoten der neuen Schicht mit allen Knoten der vorherigen Schicht verbunden
- Hierbei wir unterschieden:
  - Input-Layer
  - Hidden-Layer
  - Output-Layer
 
![NeuronaleNetze05](pictures/NeuronaleNetze05.jpg)



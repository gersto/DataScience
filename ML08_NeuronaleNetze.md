# Neuronale Netze

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
  

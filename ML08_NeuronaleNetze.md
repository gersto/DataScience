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





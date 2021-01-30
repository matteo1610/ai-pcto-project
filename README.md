# AI PCTO Project

## Descrizione
Questo progetto è stato sviluppato durante un PCTO presso l'Università di Camerino (UNICAM).
Si occupa della creazione e valutazione di reti neurali DNN e CNN utilizzando TensorFlow/Keras
su dataset come IRIS, MNIST e Fashion-MNIST.

## Struttura del progetto
- `dnn/` : Contiene implementazioni di reti DNN per IRIS e MNIST.
- `cnn/` : Contiene implementazioni di reti CNN per MNIST, Fashion-MNIST e VGG16.
- `notebooks/` : Contiene versioni interattive in Jupyter Notebook.
- `docs/` : Contiene la relazione del progetto.
- `requirements.txt` : Contiene le dipendenze necessarie.

## Installazione
```bash
pip install -r requirements.txt
```

## Esecuzione dei modelli
Esempio per eseguire la DNN su MNIST:
```bash
python dnn/dnn_mnist.py
```
Esempio per eseguire la CNN su Fashion-MNIST:
```bash
python cnn/cnn_fashion_mnist.py
```

## Dipendenze principali
- `tensorflow`
- `keras`
- `numpy`
- `scikit-learn`

"""

# requirements.txt
"""
tensorflow
keras
numpy
scikit-learn
"""

# Creazione dei file di codice

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def dnn_iris():
    """Implementazione della DNN per il dataset IRIS"""
    from sklearn import datasets
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=(X.shape[1],)),
        layers.Dense(10, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

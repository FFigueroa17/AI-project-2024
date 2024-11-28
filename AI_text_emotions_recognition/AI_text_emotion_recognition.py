import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import load_dataset
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras import regularizers
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import pickle

# 1. Cargamos el dataset
ds = load_dataset("dair-ai/emotion", "unsplit")

# Acceder al split de entrenamiento
ds_train = ds['train']

# 2. Tomar una muestra aleatoria de 100,000 filas
ds_sampled = ds_train.shuffle(seed=42).select(range(100000))  # Seleccionar las primeras 100,000 filas aleatorias

# Convertir el dataset a un DataFrame para facilitar el manejo
df = pd.DataFrame(ds_sampled)

# 3. Preprocesar el texto
def preprocess_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = ''.join([char if char.isalnum() or char.isspace() else ' ' for char in text])  # Eliminar puntuación
    return text

df['text'] = df['text'].map(preprocess_text)  # Aplicar el preprocesamiento

# Tokenización del texto
texts = [word_tokenize(sentence) for sentence in df['text']]
labels = np.array(df['label'])

# 4. Guardar los vectores de GloVe en un archivo binario (solo se necesita hacer una vez)
def load_glove_vectors(file_path):
    glove = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove[word] = vector
    return glove

# Cargar y guardar GloVe 50d
glove = load_glove_vectors('glove.6B.50d.txt')  # Descargar desde https://nlp.stanford.edu/projects/glove/
with open('glove_vectors4.pkl', 'wb') as f:
    pickle.dump(glove, f)

# 5. Cargar los vectores de GloVe desde nuestro archivo .pkl
with open('glove_vectors4.pkl', 'rb') as f:
    glove = pickle.load(f)

# 6. Promediamos vectores de las palabras de una oración
def sentence_to_vector(sentence, glove):
    vectors = [glove[word] for word in sentence if word in glove]
    if len(vectors) == 0:
        return np.zeros(100)  # Vector vacío si no hay palabras en GloVe
    return np.mean(vectors, axis=0)

# 7. Creamos los vectores para cada oración
X = np.array([sentence_to_vector(sentence, glove) for sentence in texts])

# 8. Balnceamos las clases usando RandomOverSampler
ros = RandomOverSampler(sampling_strategy='auto', random_state=42)  # Sobremuestreo de clases minoritarias

X_resampled, y_resampled = ros.fit_resample(X, labels)

# 9. Ya con nuestro dataset balanceado, dividimos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

class_counts = pd.Series(y_resampled).value_counts()  # Usar y_resampled para la distribución balanceada

# Crear la gráfica de barras para poder ver la distribución de las clases
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color='skyblue')
plt.title('Distribución de las clases después de balancear')
plt.xlabel('Clase')
plt.ylabel('Número de ejemplos')
plt.xticks(ticks=np.arange(len(class_counts)), labels=['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'], rotation=45)
plt.show()

# 10. Crear el modelo
model = tf.keras.models.Sequential([
    # Primera capa densa con regularización L2 y Dropout
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.1)),
    tf.keras.layers.Dropout(0.6),

    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.6),

    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.6),

    # Capa de salida para clasificación multiclase
    tf.keras.layers.Dense(6, activation='softmax')  # 6 clases: sadness, joy, love, anger, fear, surprise
])

# Compilación con optimizador Adam y tasa de aprendizaje ajustada
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss='sparse_categorical_crossentropy',  # Usamos sparse porque las etiquetas son enteros
              metrics=['accuracy'])

# 11. Entrenar el modelo
history_model = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test))

# 12. Evaluar el modelo
loss, acc = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Acc: {acc}")

# 13. Graficar la pérdida y la precisión durante el entrenamiento
plt.figure(figsize=(12, 4))
# Graficar la precisión
plt.subplot(1, 2, 1)
plt.plot(history_model.history['accuracy'], label='Train Accuracy')
plt.plot(history_model.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Graficar la pérdida
plt.subplot(1, 2, 2)
plt.plot(history_model.history['loss'], label='Train Loss')
plt.plot(history_model.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 14. Realizar predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)  # Obtenemos las predicciones de las clases (índices)

# 15. Classification Report y Matriz de Confusión
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión con valores numéricos
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(conf_matrix, cmap='Blues')
fig.colorbar(cax)
ax.set_xticks(np.arange(6))
ax.set_yticks(np.arange(6))
ax.set_xticklabels(['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'])
ax.set_yticklabels(['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'])

# Agregar los valores en las celdas de la matriz
for i in range(6):
    for j in range(6):
        ax.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# 16. Guardar el modelo
model.save('modelo_sentiment_emotion_11.keras')
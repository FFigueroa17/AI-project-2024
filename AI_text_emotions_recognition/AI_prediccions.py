import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from utiles import load_glove_vectors_from_pickle, preprocess_text, sentence_to_vector  # Importamos las funciones para procesar el texto

glove = load_glove_vectors_from_pickle('glove_vectors4.pkl')
model = tf.keras.models.load_model('modelo_sentiment_emotion_11.keras')

# Función para hacer predicciones de emociones, retorna el índice
def predict_emotion(sentence, glove, model):
    preprocessed_sentence = preprocess_text(sentence)  # Preprocesar la oración
    tokenized_sentence = word_tokenize(preprocessed_sentence)  # Tokenizar la oración
    sentence_vector = sentence_to_vector(tokenized_sentence, glove)  # Convertir a vector

    # Realizar la predicción
    prediction = model.predict(np.array([sentence_vector]))
    predicted_label = np.argmax(prediction, axis=1)[0]

    return predicted_label

# Con esta funcion podemos convertir el indice a la label
def emotion_index_to_label(index):
    emotion_labels = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
    return emotion_labels[index]

# Ejemplo de la entrada que tendra el usuario
user_input = "Im devastated"

# Obtener la emoción predicha
predicted_index = predict_emotion(user_input, glove, model)
predicted_emotion = emotion_index_to_label(predicted_index)

print(f"Predicted index: {predicted_index}")
print(f"Predicted Emotion: {predicted_emotion}")
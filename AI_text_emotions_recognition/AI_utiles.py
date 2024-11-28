import numpy as np
import pickle

def load_glove_vectors(file_path):
    glove = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove[word] = vector
    return glove

def save_glove_vectors(glove, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(glove, f)

def load_glove_vectors_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def preprocess_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = ''.join([char if char.isalnum() or char.isspace() else ' ' for char in text])  # Eliminar puntuación
    return text

def sentence_to_vector(sentence, glove):
    vectors = [glove[word] for word in sentence if word in glove]
    if len(vectors) == 0:
        return np.zeros(300)  # Vector vacío si no hay palabras en GloVe
    return np.mean(vectors, axis=0)
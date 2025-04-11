import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import soundfile as sf
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------- PARTE 1: EXTRAIR FEATURES --------------------------------- #

# Carregar o modelo e o scaler
MODEL_PATH = r"modelo_scaler\audio_emotion_model.keras"
SCALER_PATH = r"modelo_scaler\scaler.joblib"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Lista de emoções
EMOTIONS = ["angry", "calm", "disgust", "fear",
            "happy", "neutral", "sad", "surprise"]

# Função para extrair features
def extract_features(audio_path):
    data, sr = librosa.load(audio_path, sr=16000, mono=True)
    features = []

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y= data), axis= 1)
    features.extend(zcr)

    # Chroma STFT
    chroma_stft = np.mean(librosa.feature.chroma_stft(y= data, sr= sr), axis= 1)
    features.extend(chroma_stft)

    # MFCCs
    mfcc = np.mean(librosa.feature.mfcc(y= data, sr= sr), axis= 1)
    features.extend(mfcc)

    # RMS
    rms = np.mean(librosa.feature.rms(y= data), axis= 1)
    features.extend(rms)

    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y= data, sr= sr), axis= 1)
    features.extend(mel)

    # Garantir que tenha exatamente 162 features (ou truncar/zerar)
    target_length = 162
    if len(features) < target_length:
        features.extend([0] * (target_length - len(features)))
    elif len(features) > target_length:
        features = features[:target_length]

    return np.array(features).reshape(1, -1)


# --------------------------------- PARTE 2: STREAMLIT --------------------------------- #

# Configuração do app Streamlit (Título e descrição)

st.title('Detector de Emoções em Áudio🤔')
st.write('#### Envie um áudio para detectarmos a emoção presente!')

# Upload de arquivo de áudio (wav, mp3, ogg)
uploaded_file = st.file_uploader(
    "Escolha um arquivo de áudio...", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Salvar temporariamente o áudio

    arquivo_temporario = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    arquivo_temporario.write(uploaded_file.getvalue())
    audio_path = arquivo_temporario.name
    arquivo_temporario.close()

    # Reproduzir o áudio enviado

    st.audio(uploaded_file)

    # Extrair features

    features = extract_features(audio_path)

    # Normalizar os dados com o scaler treinado

    features_scaled = scaler.transform(features)

    # Ajustar formato para o modelo

    features_format = np.expand_dims(features_scaled, axis=2)

    # Fazer a predição

    predictions = model.predict(features_format)
    emotion = EMOTIONS[np.argmax(predictions[0])]

    # Exibir o resultado

    st.success("#### 🕴️Emoção Detectada!!!")
    st.write(f'##### 🔊{emotion}')

    # Exibir probabilidades (gráfico de barras)
   
    classes = EMOTIONS
    plt.figure(figsize=(8, 6))
    sns.barplot(x= classes, y= predictions[0], color= 'skyblue')
    st.pyplot(plt.gcf())

    # Remover o arquivo temporário
   
    os.remove(audio_path)
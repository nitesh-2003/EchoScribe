import os
import wave
import json
import pyaudio
from flask import Flask, render_template, jsonify, request
from deepmultilingualpunctuation import PunctuationModel
from vosk import Model, KaldiRecognizer
from transformers import MarianMTModel, MarianTokenizer
import threading
import datetime

# Initialize Flask app
app = Flask(__name__)

# Path to save audio files
AUDIO_FOLDER = 'audio/'
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

# Initialize punctuation model
punctuation_model = PunctuationModel()

# Language models for Vosk
LANGUAGE_MODELS = {
    "english": "C:/Users/nitee/OneDrive/Desktop/EchoScribe/models/vosk_model/english/vosk-model-small-en-us-0.15",
    "hindi": "C:/Users/nitee/OneDrive/Desktop/EchoScribe/models/vosk_model/hindi/vosk-model-small-hi-0.22",
}

# Translation models
TRANSLATION_MODELS = {
    "hindi": "Helsinki-NLP/opus-mt-hi-en",
    "english": None  # No translation needed for English
}

# Global variables for recording
is_recording = False
frames = []

# Function to record audio and save to file
def record_audio(file_name):
    global is_recording, frames
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)
    frames = []  # Clear any previous audio frames

    print("Recording...")

    while is_recording:
        data = stream.read(1024)
        frames.append(data)

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save audio to a .wav file
    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

# Function to transcribe audio to text
def transcribe_audio(file_name, model):
    recognizer = KaldiRecognizer(model, 16000)
    with wave.open(file_name, 'rb') as wf:
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                result_data = json.loads(result)
                text = result_data.get('text', '')
                if text:
                    return text
    return ""  # Return empty string if no text is transcribed

# Function to add punctuation to the transcribed text
def add_punctuation(text):
    return punctuation_model.restore_punctuation(text)

# Function to translate text to English
def translate_to_english(text, language):
    if language not in TRANSLATION_MODELS or TRANSLATION_MODELS[language] is None:
        return text  # Return original text if no translation model is specified

    model_name = TRANSLATION_MODELS[language]
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    translated = model.generate(**tokenizer(text, return_tensors="pt", max_length=512, truncation=True))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Route to start recording
@app.route('/start_recording', methods=['GET'])
def start_recording():
    global is_recording
    if not is_recording:
        is_recording = True
        file_name = os.path.join(AUDIO_FOLDER, 'recorded_audio.wav')
        threading.Thread(target=record_audio, args=(file_name,)).start()
        return jsonify({"message": "Recording started"})
    else:
        return jsonify({"message": "Recording is already in progress"})

# Function to stop recording and process the audio
@app.route('/stop_recording', methods=['GET'])
def stop_recording():
    global is_recording
    if is_recording:
        is_recording = False
        file_name = os.path.join(AUDIO_FOLDER, 'recorded_audio.wav')

        # Transcribe the recorded audio
        language = request.args.get('language', 'english')
        model_path = LANGUAGE_MODELS.get(language, LANGUAGE_MODELS['english'])
        model = Model(model_path)
        transcribed_text = transcribe_audio(file_name, model)

        if not transcribed_text:
            return jsonify({"message": "No speech detected."}), 400

        original_text = transcribed_text

        # Translate to English if the language is not English
        if language != 'english':
            translated_text = translate_to_english(transcribed_text, language)
        else:
            translated_text = transcribed_text

        # Add punctuation to the (translated) transcribed text
        punctuated_text = add_punctuation(translated_text)

        return jsonify({"original_text": original_text, "translated_text": translated_text, "punctuated_text": punctuated_text})
    else:
        return jsonify({"message": "No active recording to stop."})

# Route to render the frontend HTML
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
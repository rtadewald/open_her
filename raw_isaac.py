import speech_recognition as sr
import time
import re
from faster_whisper import WhisperModel
import os 
from RealtimeTTS import TextToAudioStream, OpenAIEngine
from langchain_groq import ChatGroq
from openai import OpenAI
import pyaudio
from pynput import keyboard

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

wake_word = "siri"
client = OpenAI()
r = sr.Recognizer()
source = sr.Microphone()
whisper_model = WhisperModel("small",
                             compute_type="int8",
                             cpu_threads=os.cpu_count(),
                             num_workers=os.cpu_count())
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")   
stop_streaming = False


# Assistant Controller
def callback(recognizer, audio):
    prompt_audio_path = "prompt.wav"
    with open(prompt_audio_path, "wb") as f:
        f.write(audio.get_wav_data())
    
    clean_prompt = wav_to_clean_text(prompt_audio_path, wake_word)
    if clean_prompt:
        print(clean_prompt)
        llm_response = llm.invoke(clean_prompt)
        print(llm_response.content)
        
        speak(llm_response.content)

def wav_to_clean_text(audio_path, wake_word):
    segments, _ = whisper_model.transcribe(audio_path, language="pt")
    text = "".join(segment.text for segment in segments)

    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return

   
def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print("Say", wake_word)
    r.listen_in_background(source, callback)

    while True:
        time.sleep(.5)

def speak(text):
    global stop_streaming
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,
                                           channels=1,
                                           rate=24000,
                                           output=True)
    stream_start = False
    with client.audio.speech.with_streaming_response.create(model="tts-1",
                voice="onyx", 
                response_format="pcm",
                 input=text
                ) as response:
        
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stop_streaming:
                stop_streaming = False
                break  # Interrompe o streaming se stop_streaming for True

            if stream_start:
                player_stream.write(chunk)
            else: 
                if max(chunk) > silence_threshold:
                    player_stream.write(chunk)
                    stream_start = True

def stop_tts(key):
    try:
        if key == keyboard.Key.cmd:
            print("Stop Jarvis")
            global stop_streaming
            stop_streaming = True
    except AttributeError:
        pass



if __name__ == "__main__":
    listener = keyboard.Listener(on_press=stop_tts)
    listener.start()
    start_listening()
import speech_recognition as sr
import time
import re
from faster_whisper import WhisperModel
import os
from RealtimeTTS import TextToAudioStream, OpenAIEngine
from openai import OpenAI
import pyaudio
from pynput import keyboard
import threading
from dotenv import load_dotenv, find_dotenv
import flet as ft

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

import cv2
from PIL import Image
import io
import base64
from collections import deque

_ = load_dotenv(find_dotenv())

wake_word = "Ana"
client = OpenAI()
r = sr.Recognizer()
source = sr.Microphone()
whisper_model = WhisperModel("small", compute_type="int8", cpu_threads=os.cpu_count(), num_workers=os.cpu_count())
stop_streaming = False
last_frames = deque(maxlen=10)

# LLM Setup
template = open('templates/vision_assistant.md', 'r').read()
prompt = PromptTemplate(input_variables=["input"], 
                        template=template)
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")
memory = ConversationBufferMemory(memory_key="chat_history",
                                input_key='input')
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)


# Vision Model
client_vlm = OpenAI(api_key='YOUR_API_KEY', base_url='http://192.168.1.4:23333/v1')
model_name = client_vlm.models.list().data[0].id
vision_chat = ChatOpenAI(
    temperature=0.0, 
    model=model_name,
    base_url="http://192.168.1.4:23333/v1", 
    api_key="YOUR_API_KEY",
    )


# Voice management
def start_listening(chat_list):
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print("Say", wake_word)
    r.listen_in_background(source, lambda recognizer, audio: callback(recognizer, audio, chat_list))
    # while True:
        # time.sleep(0.5)

def callback(recognizer, audio, chat_list):
    prompt_audio_path = "prompt.wav"
    with open(prompt_audio_path, "wb") as f:
        f.write(audio.get_wav_data())

    clean_prompt = wav_to_clean_text(prompt_audio_path, wake_word)
    if clean_prompt:
        # print(clean_prompt)
        write_chat(chat_list, clean_prompt, "👦🏽 User")

        vision_prompt()
        llm_response = llm_chain.invoke(clean_prompt)
        print(llm_response)
        # print(llm_response["text"])
        write_chat(chat_list, llm_response["text"], "🤖 Assistant")
        speak(llm_response["text"])

def vision_prompt():
    global last_frames

    inputs = [
        [HumanMessage(content=[
            {"type": "text", "text": "Transcribe briefly what you see."},
            {"type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{last_frames[-1]}"}},
            ])],
    ]

    response = vision_chat.invoke(inputs[0])
    print(response.content)

def wav_to_clean_text(audio_path, wake_word):
    segments, _ = whisper_model.transcribe(audio_path, language="pt")
    text = "".join(segment.text for segment in segments)
    print(text)

    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def write_chat(chat_list, message, user):
    user_message = ft.Markdown(f"**{user}**: {message}", 
                               extension_set="github-web", 
                               code_theme="github")
    chat_list.controls.append(user_message)
    chat_list.update()



def speak(text):
    global stop_streaming

    stop_streaming = False
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False
    with client.audio.speech.with_streaming_response.create(model="tts-1", voice="onyx", response_format="pcm", input=text) as response:
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stop_streaming:
                stop_streaming = False
                break

            if stream_start:
                player_stream.write(chunk)
            else:
                if max(chunk) > silence_threshold:
                    player_stream.write(chunk)
                    stream_start = True

def stop_tts(key):
    try:
        if key == keyboard.Key.cmd:
            print("Stop Assistant")
            global stop_streaming
            stop_streaming = True
    except AttributeError:
        pass

def webcam_stream(page, img):
    global last_frames

    fps = 30
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, fps)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame)
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        img.src_base64 = img_str
        
        if i % fps == 0:
            last_frames.append(img_str)
        page.update()
        i += 1

def main(page):
    page.window_width = 1280
    page.window_height = 720
    page.title = "Voice Assistant"
    page.vertical_alignment = ft.MainAxisAlignment.START

    chat_list = ft.ListView(expand=True, spacing=10, padding=20, auto_scroll=True)
    vision_list = ft.ListView(expand=True, spacing=10, padding=20, auto_scroll=True)
    webcam_view = ft.Image(width=640, height=360)

    chat_column = ft.Column([ft.Text("Voice Assistant is listening for the wake word..."), chat_list], expand=True)
    webcam_column = ft.Column([webcam_view], width=640)
    main_row = ft.Row([chat_column, webcam_column], expand=True)
    page.add(main_row)

    # Start the listening thread automatically
    threading.Thread(target=start_listening, args=(chat_list,)).start()
    listener = keyboard.Listener(on_press=stop_tts)
    listener.start()
    
    threading.Thread(target=webcam_stream, args=(page, webcam_view)).start()

ft.app(target=main)

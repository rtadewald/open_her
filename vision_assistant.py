from faster_whisper import WhisperModel
import os
import pyaudio
import wave
from pynput import keyboard
import threading
import flet as ft

import google.generativeai as genai

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_groq import ChatGroq
from langchain import hub
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from openai import OpenAI

from wake_word_detector import *

from PIL import Image
import pyautogui
import markdown
from dotenv import load_dotenv, find_dotenv
import cv2
import io
import base64
from collections import deque
import queue
from time import time, sleep


# App Control Variables
image_analysis = True
local_whisper_inferece=False
audio_device = 0
cam_device = 0
llm_image_prompt = False
# base_image_prompt = """
# You are in a kitchen. Describe all the foods you see in the image.
# """
base_image_prompt = """
Você está em uma cozinha. Descreva os alimentos que você vê. 
Responda em portugues apenas.
# """

base_image_prompt = """
Describe what you see in this image.
"""

# ===========================
# Recording Setup
stop_streaming = False
recording_thread = None
recording = False
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Threading Variables
wake_event = threading.Event()

last_frames = deque(maxlen=1)
processed_frames = deque()
image_prompt = queue.Queue(maxsize=1)

last_ts = time()
process_counter=0
global_ts = 0


# LLM Setup
if True:
    _ = load_dotenv(override=True)
    client = OpenAI()
    local_client = OpenAI(api_key="cant-be-empty", 
                        base_url="http://192.168.1.5:8000/v1/")

    whisper_model = WhisperModel("small", 
                                compute_type="int8", 
                                cpu_threads=os.cpu_count(), 
                                num_workers=os.cpu_count())

    template = open('templates/vision_assistant.md', 'r').read()
    prompt = PromptTemplate(input_variables=["input", "video_description"], 
                            template=template)
    llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")
    memory = ConversationBufferMemory(memory_key="chat_history",
                                    input_key='input')
    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    # LLM Prompter
    template2 = open('templates/vision_prompter.md', 'r').read()
    prompt2 = PromptTemplate(input_variables=["input"], 
                            template=template2)
    llm2 = ChatGroq(temperature=0, model_name="llama3-8b-8192")
    memory2 = ConversationBufferMemory(memory_key="chat_history",
                                    input_key='input')
    llm_chain2 = LLMChain(llm=llm2, prompt=prompt2, memory=memory2)

    # Vision Model
    client_vlm = OpenAI(api_key='YOUR_API_KEY', base_url='http://192.168.1.5:23333/v1')
    model_name = client_vlm.models.list().data[0].id
    vision_chat = ChatOpenAI(
        temperature=0.0, 
        model=model_name,
        base_url="http://192.168.1.5:23333/v1", 
        api_key="YOUR_API_KEY",
        )


# Audio Functions
def toggle_recording(e, page, chat_list):
    global recording, recording_thread, global_ts, wake_event

    if not recording:
        recording = True
        recording_thread = threading.Thread(target=record_audio, args=('prompt.wav', 
                                                                       chat_list))
        recording_thread.start()
        
        wake_event.set()
        wake_event.clear()
        e.control.icon = ft.icons.STOP
        e.control.tooltip = "Stop Recording"
    else:
        global_ts = time()
        recording = False
        if recording_thread:
            recording_thread.join()
        e.control.icon = ft.icons.MIC
        e.control.tooltip = "Start Recording"
    page.update()

def record_audio(filename, chat_list):
    global recording
    recording = True
    audio = pyaudio.PyAudio()

    # Inicia o stream de gravação
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        input_device_index=audio_device,
                        frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []
    while recording:
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording stopped")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Salva o arquivo de áudio
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    threading.Thread(target=callback, 
                        args=(filename, 
                        chat_list)).start()

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
 

# UI Functions
def write_chat(chat_list, prompt, user=True):
    sender = "You" if user else "Assistant"
    message = ft.Markdown(f"**{sender}**: {prompt}", 
                          extension_set="github-web", code_theme="github")
    chat_list.controls.append(message)
    chat_list.update()

def on_text_input(e, page):
    user_input = e.control.value
    e.control.value = ""
    page.update()
    callback(user_input, transcribe=False)

def webcam_stream(page, img):
    global last_frames, recording

    fps = 30
    cap = cv2.VideoCapture(cam_device)
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
            last_frames.append((time(), img_str))
        page.update()
        i += 1


# MAIN CALLBACK
def vlm_call():
    global last_frames, recording, process_counter, base_image_prompt 
    global wake_event

    def vision_ai(ts, question, frame):
        global last_frames, process_counter
        input = [HumanMessage(content=[
                {"type": "text", "text": question},
                {"type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"}},
                ])]
        response = vision_chat.invoke(input, max_tokens=512)
        processed_frames.append((ts, response.content))
        process_counter -= 1

    while True:
        wake_event.wait()
        process_counter += 1

        # if len(last_frames) > 0:
        #     ts, frame = last_frames.popleft()
        # else:
        #     sleep(0.1)

        #     continue
        ts, frame = last_frames.popleft()
        
        if llm_image_prompt:
            question = image_prompt.get()
        else:
            question = base_image_prompt

        print(f"Adding new thread {process_counter}")
        t1 = threading.Thread(target=vision_ai, args=(ts, question, frame))
        t1.start()

            

def callback(audio_or_input, chat_list, transcribe=True):
    if transcribe:
        if local_whisper_inferece:
            segments, _ = whisper_model.transcribe(audio_or_input, language="pt")
            clean_prompt = "".join(segment.text for segment in segments).strip()
        else:
            audio_file = open(audio_or_input, "rb")
            clean_prompt = local_client.audio.transcriptions.create(
                # model="Systran/faster-distil-whisper-large-v3", 
                model="Systran/faster-whisper-small", 
                file=audio_file
            ).text
            # print(transcript.text)
    else:
        clean_prompt = audio_or_input

    if clean_prompt:
        print(clean_prompt)
        write_chat(chat_list, clean_prompt)

        if llm_image_prompt:
            vision_prompt = llm_chain2.invoke(clean_prompt)["text"]
            image_prompt.put(vision_prompt)
            print(vision_prompt)
        
        print(f"Process {process_counter}")
        while process_counter != 0:
            sleep(0.1)
        
        last_view = ""
        while len(processed_frames) > 0:
            ts, answer = processed_frames.popleft()
            last_view += f"{ts}: {answer}\n"
        print("Last view:", last_view)
        llm_response = llm_chain.invoke({"input": clean_prompt, 
                                         "video_description": last_view})
        
        global global_ts
        timediff = round((time() - global_ts) * 10) / 10
        # response = f"{timediff} : {llm_response['text']}"
        response = llm_response['text']
        print(f"LLM Response: {response}")
        write_chat(chat_list, response, user=False)
        threading.Thread(target=speak, args=(response, )).start()


def main(page):
    # UI Setup
    page.title = "Voice Assistant"
    page.vertical_alignment = ft.MainAxisAlignment.START

    chat_list = ft.ListView(expand=True, spacing=10, padding=20, auto_scroll=True)
    webcam_view = ft.Image()
    input_field = ft.TextField(label="You:",
                               width=90, 
                               expand=True, 
                               on_submit=lambda e: on_text_input(e, page))
    record_button = ft.IconButton(
        icon=ft.icons.MIC,
        tooltip="Start Recording",
        on_click=lambda e: toggle_recording(e, page, chat_list),
    )

    chat_column = ft.Column([webcam_view, chat_list,
        ft.Row(controls=[input_field, record_button,
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ], expand=True)
    page.add(chat_column)


    # Background Controller Functions
    def on_keyboard_event(e):
        global stop_streaming, recording
        if e.meta and e.key == "S":
            print("Stop Assistant")
            record_button.icon = ft.icons.MIC
            record_button.tooltip = "Start Recording"
            stop_streaming = True
        
        elif e.meta and e.key == "R":
            record_button.icon = ft.icons.MIC if recording else ft.icons.STOP
            record_button.tooltip = "Start Recording" if recording else "Stop Recording"
            toggle_recording(e, page, chat_list)


    def detect_wake_work_in_background(chat_list):
        global wake_event
        wake_word_detector = WakeWordDetector(wake_event)
        wake_word_detector.start()

        while True:
            wake_event.wait()
            callback(wake_word_detector.clean_prompt, 
                     chat_list,
                     transcribe=False)

    # Threadings
    page.on_keyboard_event = on_keyboard_event
    detect_thread = threading.Thread(target=detect_wake_work_in_background,
                                     args=(chat_list, ) )
    detect_thread.start()

    t_vlm = threading.Thread(target=vlm_call)
    t_vlm.start()
   
    threading.Thread(target=webcam_stream, args=(page, webcam_view)).start()



if __name__ == "__main__":
    ft.app(target=main)
    # oi = 1
    
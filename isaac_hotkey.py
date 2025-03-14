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

from PIL import Image
import pyautogui
import markdown
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(override=True)

client = OpenAI()
whisper_model = WhisperModel("small", 
                             compute_type="int8", 
                             cpu_threads=os.cpu_count(), 
                             num_workers=os.cpu_count())

# Configurações de gravação
stop_streaming = False
recording_thread = None
recording = False
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# LLM Setup
template = open('templates/vision_assistant.md', 'r').read()
prompt = PromptTemplate(input_variables=["input", "video_description"], 
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


# Funções
def record_audio(filename='prompt.wav', page=None):
    global recording
    recording = True
    audio = pyaudio.PyAudio()

    # Inicia o stream de gravação
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        input_device_index=1,
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
    
    if page:
        threading.Thread(target=callback, args=(filename, page)).start()
        # callback(filename, page)

def on_press(key, page):
    global recording_thread, recording, stop_streaming
    if key == keyboard.Key.alt_r:
        if not recording:
            recording = True
            recording_thread = threading.Thread(target=record_audio, args=('prompt.wav',))
            recording_thread.start()
        else:
            recording = False
            recording_thread.join()
            callback('prompt.wav', page)
    
    if key == keyboard.Key.alt_l:
        user_input = input("You:")
        callback(user_input, page, transcribe=False)

    elif key == keyboard.Key.cmd:
        print("Stop Assistant")
        stop_streaming = True

def start_listening(page):
    print("Press Alt to start and stop recording")
    print("Press Cmd to stop TTS")
    
    def on_press_wrapper(key):
        on_press(key, page)

    listener = keyboard.Listener(on_press=on_press_wrapper)
    listener.start()
    listener.join()

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
 
def vision_prompt(vision_list):
    global last_frames

    inputs = []
    last_view = ""
    for frame in last_frames:
        inputs+= [HumanMessage(content=[
            {"type": "text", "text": "Describe the body of the guy"},
            {"type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{frame}"}},
            ])],
    responses = vision_chat.batch(inputs, max_tokens=512)
    for i, response in enumerate(responses):
        last_view += f"{i}: {response.content}"
        video_description = ft.Markdown(f"{i}: {response.content}")
        vision_list.controls.append(video_description)
    vision_list.update()
    return last_view




# UI Functions
def toggle_recording(e, page):
    global recording, recording_thread
    if not recording:
        recording = True
        recording_thread = threading.Thread(target=record_audio, args=('prompt.wav', page))
        recording_thread.start()
        e.control.icon = ft.icons.STOP
        e.control.tooltip = "Stop Recording"
    else:
        recording = False
        if recording_thread:
            recording_thread.join()
        e.control.icon = ft.icons.MIC
        e.control.tooltip = "Start Recording"
    page.update()

def write_chat(page, prompt, user=True):
    sender = "You" if user else "Assistant"
    message = ft.Markdown(f"**{sender}**: {prompt}", extension_set="github-web", code_theme="github")
    page.controls[1].controls.append(message)
    page.update()

def on_text_input(e, page):
    user_input = e.control.value
    e.control.value = ""
    page.update()
    callback(user_input, page, transcribe=False)




# MAIN CALLBACK
def callback(audio_or_input, page, transcribe=True):
    if transcribe:
        segments, _ = whisper_model.transcribe(audio_or_input, language="pt")
        clean_prompt = "".join(segment.text for segment in segments).strip()
    else:
        clean_prompt = audio_or_input

    if clean_prompt:
        print(clean_prompt)
        write_chat(page, clean_prompt)

        response = groq_prompt(clean_prompt)
        print(response)
        write_chat(page, response, user=False)


        last_view = vision_prompt(vision_list)
        llm_response = llm_chain.invoke({"input": clean_prompt, 
                                         "video_description": last_view})
        # print(llm_response)
        # print(llm_response["text"])
        write_chat(chat_list, llm_response["text"], "🤖 Assistant")
        speak(llm_response["text"])

        threading.Thread(target=speak, args=(response, )).start()

def main(page):
    def on_keyboard_event(e):
        global stop_streaming
        if e.meta and e.key == "S":
            print("Stop Assistant")
            record_button.icon = ft.icons.MIC
            record_button.tooltip = "Start Recording"
            stop_streaming = True
        
        elif e.meta and e.key == "R":
            record_button.icon = ft.icons.STOP
            record_button.tooltip = "Stop Recording"
            toggle_recording(e, page)
        page.update()

    page.on_keyboard_event = on_keyboard_event

    page.title = "Voice Assistant"
    page.vertical_alignment = ft.MainAxisAlignment.START

    chat_list = ft.ListView(expand=True, spacing=10, padding=20, auto_scroll=True)
    input_field = ft.TextField(label="You:", on_submit=lambda e: on_text_input(e, page))
    record_button = ft.IconButton(
        icon=ft.icons.MIC,
        tooltip="Start Recording",
        on_click=lambda e: toggle_recording(e, page),
    )

    page.add(
        ft.Row(
            controls=[
                ft.Text("Voice Assistant is listening for the hotkey..."),
                record_button
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN  # Alinha os itens nas extremidades
        )
    )
    page.add(chat_list)
    page.add(input_field)

    # Adicionar o KeyboardListener para capturar hotkeys
    



if __name__ == "__main__":
    ft.app(target=main)
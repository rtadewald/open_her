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

agent_executor = None

# Configurações de gravação
stop_streaming = False
recording_thread = None
recording = False
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024

# Funções
def record_audio(filename='prompt.wav'):
    global recording
    recording = True
    audio = pyaudio.PyAudio()

    # Inicia o stream de gravação
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
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
    return filename

def write_chat(page, prompt, llm_response):
    user_message = ft.Markdown(f"**You**: {prompt}", extension_set="github-web", code_theme="github")
    assistant_message = ft.Markdown(f"**Assistant**: {llm_response}", extension_set="github-web", code_theme="github")
    
    page.controls[1].controls.append(user_message)
    page.controls[1].controls.append(assistant_message)
    page.update()

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
 

# Tools
@tool
def vision_prompt(prompt: str = None):
    """
    Take a screenshot and uses the 'prompt' variable to
    ask for an artificial intelligence (AI) to describe precisely
    what's in the image.

    prompt(str): The prompt to the AI to guide it on how should it describe in the image.
    """
    
    img_path = "screenshot.png"
    screenshot = pyautogui.screenshot()
    screenshot.save(img_path)
    img = Image.open(img_path)

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    generation_config = {"temperature": 0.7, 
                        "top_p": 1, 
                        "top_k": 1,
                        "max_output_tokens": 2048}

    gemini = genai.GenerativeModel("gemini-1.5-flash-latest",
                                generation_config=generation_config)
    
    if prompt is None:
        prompt = open('templates/vision.md', 'r').read()
    response = gemini.generate_content([prompt, img])
    return response.text



# LLMs 
def groq_prompt(user_input):
    global agent_executor
    if agent_executor is None:
        template = open('templates/react.md', 'r').read()
        prompt = PromptTemplate.from_template(template)
        
        llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
        # llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")
        # llm = ChatOpenAI(temperature=0, model="gpt-4")
        memory = ConversationBufferMemory(memory_key="chat_history",
                                        input_key='input')
        
        tools = [vision_prompt]
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, 
                                       memory=memory,
                                       tools=tools, 
                                       verbose=True)
    # print(user_input)
    response = agent_executor.invoke({"input": user_input})
    print(response)
    return response["output"]


# MAIN CALLBACK
def callback(audio_or_input, page, transcribe=True):
    if transcribe:
        segments, _ = whisper_model.transcribe(audio_or_input, language="pt")
        clean_prompt = "".join(segment.text for segment in segments).strip()
    else:
        clean_prompt = audio_or_input

    if clean_prompt:
        print(clean_prompt)
        response = groq_prompt(clean_prompt)
        print(response)

        write_chat(page, clean_prompt, response)
        threading.Thread(target=speak, args=(response, )).start()
        # speak(response)

def main(page):
    page.title = "Voice Assistant"
    page.vertical_alignment = ft.MainAxisAlignment.START

    chat_list = ft.ListView(expand=True, spacing=10, padding=20, auto_scroll=True)
    page.add(ft.Text("Voice Assistant is listening for the hotkey..."))
    page.add(chat_list)

    listener_thread = threading.Thread(target=start_listening, args=(page,))
    listener_thread.daemon = True
    listener_thread.start()


if __name__ == "__main__":
    ft.app(target=main)
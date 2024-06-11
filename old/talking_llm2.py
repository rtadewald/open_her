import sounddevice as sd
import numpy as np
# import keyboard
import subprocess
import tempfile
import wave
from pynput import keyboard
import whisper
from faster_whisper import WhisperModel
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from playsound import playsound
from queue import Queue
import threading
import io
import soundfile as sf
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents.agent_types import AgentType
import openai
from dotenv import load_dotenv, find_dotenv
import asyncio
import speech_recognition as sr
import pyaudio
import time
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

_ = load_dotenv(find_dotenv())
client = openai.Client()


class TalkingLLM():
    def __init__(self, model="gpt-3.5-turbo-0613", 
                 whisper_size="small", 
                 tts_voice="echo", 
                 faster_whisper=True):
        self.is_recording = False
        self.audio_data = []
        self.llm_queue = Queue()
        self.samplerate=44100
        self.channels=1
        self.dtype='int16'

        self.wake_word = "GPT"
        self.listening_for_wake_word = True
        self.r = sr.Recognizer()
        self.source = sr.Microphone()
        self.tts_voice = tts_voice #echo, alloy, echo, fable, onyx, nova 
        self.fw = faster_whisper

        # Run on GPU with FP16
        if self.fw:
            num_cores = os.cpu_count()
            self.whisper = WhisperModel(whisper_size,
                                        compute_type="int8",
                                        cpu_threads=num_cores,
                                        num_workers=num_cores)
        else:
            self.whisper = whisper.load_model(whisper_size) # tiny, base, small, medium
        
        # self.llm = ChatOpenAI(temperature=0, model=model) #gpt-3.5-turbo-0613
        self.llm = ChatOllama(model="llama3:8b")
        # self.llm = Ollama(model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
        
    def speak(self, text):
        player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
        stream_start = False 
        with client.audio.speech.with_streaming_response.create(
            model="tts-1", 
            voice="alloy", 
            response_format="pcm", 
            input=text) as response: 
        
            silence_threshold = 0.01 
            for chunk in response.iter_bytes(chunk_size=1024): 
                if stream_start: 
                    player_stream.write(chunk) 
                elif max(chunk) > silence_threshold: 
                    player_stream.write(chunk) 
                    stream_start = True 

    def wav_to_text(self, audio_path):
        print("iniciando")
        if self.fw:
            segments, _ = self.whisper.transcribe(audio_path)
            text = ''.join(segment.text for segment in segments)
        else:
            text = self.whisper.transcribe(audio_path, fp16=False)["text"]
        print("transcrito")
        return text
    

    def listen_for_wake_word(self, audio):
        wake_audio_path = 'wake_detect.wav'
        with open(wake_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())
    
        text_input = self.wav_to_text(wake_audio_path)
        print("You said: ", text_input)

        if self.wake_word.lower() in text_input.lower().strip():
            print('Wake word detected. Please speak your prompt to Gemini.')
            self.listening_for_wake_word = False


    def prompt_gpt(self, audio):
        # try:
        prompt_audio_path = 'prompt.wav'

        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())

        prompt_text = self.wav_to_text(prompt_audio_path)

        if len(prompt_text.strip()) == 0:
            print('Empty prompt. Please speak again.')
            self.listening_for_wake_word = True
        else:
            print('User: ' + prompt_text)
            response = self.llm.invoke(prompt_text)
            output = response.content

            print('Gemini: ', output)
            self.speak(output)
        
        # except Exception as e:
            # print("Prompt error: ", e)


    def callback(self, recognizer, audio):
        if self.listening_for_wake_word:
            self.listen_for_wake_word(audio)
        else:
            self.prompt_gpt(audio)


    def start_listening(self):
        with self.source as s:
            self.r.adjust_for_ambient_noise(s, duration=2)

        print('\nSay', self.wake_word, 'to wake me up.\n')
        self.r.listen_in_background(self.source, self.callback)

        while True:
            time.sleep(0.5)

if __name__ == "__main__":
    # tts_voice = echo, alloy, echo, fable, onyx, nova 
    talking_llm = TalkingLLM(model="gpt-4", 
                             tts_voice="alloy",
                             faster_whisper=True)
    talking_llm.start_listening()

    
    
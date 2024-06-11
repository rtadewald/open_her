import numpy as np
# import keyboard
import subprocess
import tempfile
import wave
from pynput import keyboard
# import whisper
from faster_whisper import WhisperModel
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
# from playsound import playsound
from queue import Queue
import threading
import io
import soundfile as sf
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents.agent_types import AgentType
import openai
import asyncio

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_groq import ChatGroq


from RealtimeTTS import TextToAudioStream, SystemEngine, ElevenlabsEngine, GTTSEngine, OpenAIEngine
from RealtimeSTT import AudioToTextRecorder

from stt.google_stt import GoogleSTT

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

client = openai.Client()


class TalkingLLM():
    def __init__(self, 
                 model="gpt-3.5-turbo-0613", 
                 whisper_size="small", 
                 tts_voice="echo",
                 faster_whisper=True):

        self.fw = faster_whisper
        
        self.start = False  
        # self.recorder = GoogleSTT()     
        self.recorder = AudioToTextRecorder(
                            spinner=False, 
                            model="small", 
                            input_device_index=2, 
                            compute_type='int8',
                            enable_realtime_transcription=True, 
                            silero_use_onnx=True, 
                            realtime_model_type="small", 
                            language="pt")
        
        if self.fw:
            self.whisper = WhisperModel(whisper_size,
                                        compute_type="int8",
                                        cpu_threads=os.cpu_count(),
                                        num_workers=os.cpu_count())
        else:
            self.whisper = whisper.load_model(whisper_size) # tiny, base, small, medium
        # self.llm = ChatOpenAI(temperature=0, model=model) #gpt-3.5-turbo-0613
        # self.llm = ChatOllama(model="llama3:instruct")
        self.llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")       


    def _stt(self):
        recorder = AudioToTextRecorder(
                        spinner=False, 
                        model="small", 
                        input_device_index=2, 
                        compute_type='int8',
                        # enable_realtime_transcription=True, 
                        # silero_use_onnx=True, 
                        # realtime_model_type="small", 
                        language="pt")

        while self.start:
            print("here")
            print(recorder.text(), end=" ", flush=True)
        print("done")


    def _tts(self, text):
        # engine = GTTSEngine() # replace with your TTS engine
        def dummy_generator():
            for chunk in self.llm.stream(text):
                print(chunk.content, end="", flush=True)
                yield chunk.content
            
        engine = OpenAIEngine(model="tts-1", voice='onyx')
        # engine = ElevenlabsEngine(os.environ.get("ELEVENLABS_API_KEY"))

        stream = TextToAudioStream(engine)
        stream.feed(dummy_generator())
        stream.play_async()
        # stream.feed("Hello world! How are you today?")
        # self.tts_engine.play(fast_sentence_fragment=True)
    

    def run(self):
        def on_activate():
            self.start = not self.start
            if self.start:
                # print("Loading Recorder...")
                self.recorder.start()
                print("Recording...")
            else:
                self.recorder.stop()
                # self.transcription = self.recorder.stop()
                self.transcription = self.recorder.text()
                print(self.transcription)
                self._tts(self.transcription)

            # print(self.recorder.text(), end=" ", flush=True)
            # print("done")


        def on_press(key):
            try:
                # Verifica se a tecla Command foi pressionada
                if key == keyboard.Key.cmd:
                    on_activate()
            except AttributeError:
                pass

        with keyboard.Listener(
                on_press=on_press) as l:
            l.join()



if __name__ == "__main__":
    # tts_voice = echo, alloy, echo, fable, onyx, nova 
    talking_llm = TalkingLLM(model="gpt-4", tts_voice="alloy")
    print("ready")
    talking_llm.run()

    
    
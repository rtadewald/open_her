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
from dotenv import load_dotenv
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
import asyncio

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from RealtimeTTS import TextToAudioStream, OpenAIEngine

from dotenv import load_dotenv, find_dotenv
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
        self.fw = faster_whisper
        self.tts_engine = TextToAudioStream(OpenAIEngine(model="tts-1", voice="nova"))
        self.tts_voice = tts_voice #echo, alloy, echo, fable, onyx, nova 

        if self.fw:
            self.whisper = WhisperModel(whisper_size,
                                        compute_type="int8",
                                        cpu_threads=os.cpu_count(),
                                        num_workers=os.cpu_count())
        else:
            self.whisper = whisper.load_model(whisper_size) # tiny, base, small, medium

        # self.llm = ChatOpenAI(temperature=0, model=model) #gpt-3.5-turbo-0613
        self.llm = ChatOllama(model="llama3:8b")       
        # self.create_agent()


    def create_agent(self):
        pass


    def start_or_stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            # self.save_recording()
            self.save_and_transcribe()
            self.audio_data = []
        else:
            print("Starting record")
            self.audio_data = []
            self.is_recording = True
    

    def save_and_transcribe(self):
        print("Saving the recording...")
        if "temp.wav" in os.listdir(): os.remove("temp.wav")
        wav_file = wave.open("test.wav", 'wb')
        wav_file.setnchannels(self.channels)
        wav_file.setsampwidth(2)  # Corrigido para usar a largura de amostra para int16 diretamente
        wav_file.setframerate(self.samplerate)
        wav_file.writeframes(np.array(self.audio_data, dtype=self.dtype))
        wav_file.close()
        
        sequential = False
        # result = self.whisper.transcribe("test.wav", fp16=False)
        if self.fw:
            segments, _ = self.whisper.transcribe("test.wav")
            result = ''.join(segment.text for segment in segments)
        else:
            result = self.whisper.transcribe("test.wav", fp16=False)["text"]
        # result = self.whisper.transcribe("test.wav")
        print("Usuário:", result)

        if sequential:  
            # result = {"text": "Olá, tudo bem?"}
            # response = self.agent.invoke(result["text"])
            # self.llm_queue.put(response["output"]) 
            # print(response)
            response_ = self.llm.invoke(result)
            self.llm_queue.put(response_.content)

        else:
            self.result = result
            # def play_audio
            t2 = threading.Thread(target=self.convert_and_play)
            t2.start()

            # # self.play = True
            # # print(result)
            # def dummy_generator():
            #     for chunk in self.llm.stream(self.result):
            #         print(chunk.content, end="", flush=True)
            #         yield chunk.content
                
            # self.tts_engine.feed(dummy_generator())
            # self.tts_engine.play()
            # for chunk in self.llm.stream(result):
            #     # print(chunk)
            #     self.llm_queue.put(chunk.content)


    def convert_and_play(self):
        def dummy_generator():
            for chunk in self.llm.stream(self.result):
                print(chunk.content, end="", flush=True)
                yield chunk.content
            
        self.tts_engine.feed(dummy_generator())
        self.tts_engine.play(fast_sentence_fragment=True)
        # tts_text = ''
        # while True:
        #     tts_text += self.llm_queue.get()

        #     if '.' in tts_text or '?' in tts_text or '!' in tts_text:
        #         print(tts_text)

        #         spoken_response = client.audio.speech.create(model="tts-1",
        #         voice=self.tts_voice, 
        #         response_format="opus",
        #          input=tts_text
        #         )

        #         buffer = io.BytesIO()
        #         for chunk in spoken_response.iter_bytes(chunk_size=4096):
        #             buffer.write(chunk)
        #         buffer.seek(0)

        #         with sf.SoundFile(buffer, 'r') as sound_file:
        #             data = sound_file.read(dtype='int16')
        #             sd.play(data, sound_file.samplerate)
        #             sd.wait()
        #         tts_text = ''


    def run(self):
        # t2 = threading.Thread(target=self.convert_and_play)
        # t2.start()

        def callback(indata, frame_count, time_info, status):
            if self.is_recording:
                self.audio_data.extend(indata.copy())

        with sd.InputStream(samplerate=self.samplerate, 
                            channels=self.channels, 
                            dtype=self.dtype , 
                            callback=callback):
            def on_activate():
                self.start_or_stop_recording()

            def for_canonical(f):
                return lambda k: f(l.canonical(k))

            hotkey = keyboard.HotKey(
                keyboard.HotKey.parse('<cmd>'), on_activate)

            with keyboard.Listener(
                    on_press=for_canonical(hotkey.press),
                    on_release=for_canonical(hotkey.release)) as l:
                l.join()



if __name__ == "__main__":
    # tts_voice = echo, alloy, echo, fable, onyx, nova 
    talking_llm = TalkingLLM(model="gpt-4", tts_voice="alloy")
    talking_llm.run()

    
    
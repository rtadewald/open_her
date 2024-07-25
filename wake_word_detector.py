import threading
import speech_recognition as sr
from openai import OpenAI
from pydub import AudioSegment
import numpy as np
import librosa
import io
import webrtcvad
import threading
import speech_recognition as sr
from openai import OpenAI

local_client = OpenAI(api_key="cant-be-empty", 
                        base_url="http://192.168.1.5:8000/v1/")


class WakeWordDetector:
    def __init__(self, 
                 wake_event,
                 wake_word="siri",
                 min_energy=0.8,
                 model = "Systran/faster-whisper-small",
                 ):
        
        self.min_energy = min_energy
        self.model = model
        self.wake_word = wake_word
        self.wake_event = wake_event

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
    def detect_voice(self, audio_data):
        # self.detect_voice
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
        samples = np.array(audio_segment.get_array_of_samples())
        samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
        
        # Use librosa to get the zero-crossing rate
        energy = np.sum(samples ** 2) / len(samples) * 10000
        # print(f"Zero Crossing Rate: {zcr.mean()}, Energy: {energy}")
        print(f"Energy: {energy}")

        # if np.mean(zcr) > self.min_zcr and energy > self.min_energy:
        if energy > self.min_energy:
            print("Voice detected based on ZCR and energy")
            return True
        else:
            print("No voice detected")
            return False


    # Função para processar o áudio e procurar pela wake word
    def listen_for_wake_word(self):
        with self.microphone as source:            
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            while True:
                print("Fale")
                
                while True:
                    try:
                        audio = self.recognizer.listen(source, timeout=2)
                        break
                    except:
                        continue
                
                print(f"Processando... {type(audio)}")
                audio_data = audio.get_wav_data()
                voice_detected = self.detect_voice(audio_data)
                # detect_voice = self.detect_voice(audio_data)

                # try:
                if voice_detected:
                    clean_prompt = local_client.audio.transcriptions.create(
                        model=self.model, 
                        file=audio_data
                    ).text
                    print(f"Você disse: {clean_prompt}")

                if voice_detected and self.wake_word.lower() in clean_prompt.lower():
                    self.clean_prompt = clean_prompt
                    self.wake_event.set()
                    print("Wake word detectada!")
                self.wake_event.clear()


    # Função principal para iniciar o listener em uma thread separada
    def start(self):
        self.listener_thread = threading.Thread(target=self.listen_for_wake_word)
        self.listener_thread.start()


if __name__ == "__main__":
    wake_event = threading.Event()
    self = WakeWordDetector(wake_event)
    self.start()

    # self.start()


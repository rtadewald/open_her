from openai import OpenAI

client = OpenAI(api_key="cant-be-empty", base_url="http://192.168.1.4:8000/v1/")

ts = time()
audio_file = open("oi.wav", "rb")
transcript = client.audio.transcriptions.create(
    # model="Systran/faster-distil-whisper-large-v3", 
    model="Systran/faster-whisper-small", 
    file=audio_file
)
transcript
print(transcript.text)
print(f"Total time: {time() - ts}")


# Local
import os
from faster_whisper import WhisperModel
whisper_model = WhisperModel(
                            # "small", 
                            "Systran/faster-whisper-small",
                            compute_type="int8", 
                            cpu_threads=os.cpu_count(), 
                            num_workers=os.cpu_count())
ts = time()
segments, _ = whisper_model.transcribe("oi.wav", language="pt")
clean_prompt = "".join(segment.text for segment in segments).strip()
print(clean_prompt)
print(f"Total time: {time() - ts}")



import pyaudio
import wave 
recording = False
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
recording = True
audio = pyaudio.PyAudio()

# Inicia o stream de gravação
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    input_device_index=0,
                    frames_per_buffer=CHUNK)
print("Recording...")
frames = []
from time import time
ts = time()
while recording:
    data = stream.read(CHUNK)
    frames.append(data)
    if time() - ts > 10:
        break


print("Recording stopped")
stream.stop_stream()
stream.close()
audio.terminate()

with wave.open("oi.wav", 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
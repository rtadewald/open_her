import pyaudio
import numpy as np
import faster_whisper
import time

# Configurações de áudio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 * 5
CHUNK = int(RATE / 10)  # 100ms de áudio por chunk
THRESHOLD = 500  # Limiar para detecção de som

# Inicializar o modelo Faster-Whisper
whisper_model = faster_whisper.WhisperModel('small')

audio_buffer = []

def detect_sound(data):
    """Detecta se há som baseado no nível de energia."""
    audio_data = np.frombuffer(data, dtype=np.int16)
    energy = np.sum(audio_data**2) / len(audio_data)
    return energy > THRESHOLD

def callback(in_data, frame_count, time_info, status):
    global audio_buffer
    if detect_sound(in_data):
        audio_buffer.extend(np.frombuffer(in_data, dtype=np.int16))
    return (in_data, pyaudio.paContinue)

# Inicializar PyAudio
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

def transcribe_audio_buffer():
    global audio_buffer
    if len(audio_buffer) >= RATE:  # 1 segundo de áudio
        audio_data = np.array(audio_buffer[:RATE], dtype=np.float32) / 32768.0  # Normalizar o áudio
        audio_buffer = audio_buffer[RATE:]

        segments, _ = whisper_model.transcribe(audio_data, language='pt')
        transcribed_text = " ".join([segment.text for segment in segments])
        if transcribed_text.strip():
            print(f"Transcrição: {transcribed_text}")

def main():
    print("Iniciando a captura de áudio...")
    stream.start_stream()

    try:
        while True:
            transcribe_audio_buffer()
            # time.sleep(1)  # Aguarda 1 segundo para a próxima transcrição
    except KeyboardInterrupt:
        print("Interrompido pelo usuário, finalizando...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()

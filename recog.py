import threading
import speech_recognition as sr
from openai import OpenAI

local_client = OpenAI(api_key="cant-be-empty", 
                        base_url="http://192.168.1.4:8000/v1/")

# Função para processar o áudio e procurar pela wake word
def listen_for_wake_word(recognizer, microphone, wake_word, stop_event):
    with microphone as source:            
        recognizer.adjust_for_ambient_noise(source, duration=2)
        while not stop_event.is_set():
            print("Fale")
            audio = recognizer.listen(source)
            print(f"Processando... {type(audio)}")


            from pydub import AudioSegment
            import numpy as np
            import librosa
            import io
            import webrtcvad


            audio_data = audio.get_wav_data()

            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")

            samples = np.array(audio_segment.get_array_of_samples())
            samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
            
            
            # Use librosa to get the zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(samples, frame_length=2048, hop_length=512)
            energy = np.sum(samples ** 2) / len(samples) * 10000

            print(f"Zero Crossing Rate: {zcr.mean()}, Energy: {energy}")

            # Usando WebRTC VAD
            vad = webrtcvad.Vad(1)
            frame_duration = 30  # ms
            frame = audio_data[:int(0.03 * 16000 * 2)]  # 30 ms de áudio em 16kHz, 2 bytes por amostra

            # if vad.is_speech(frame, 16000):
                # print("WebRTC VAD detectou voz")
            if np.mean(zcr) > 0.8 and energy > 0.8:
                print("Voice detected based on ZCR and energy")
            else:
                print("No voice detected")
                continue

            # try:
            clean_prompt = local_client.audio.transcriptions.create(
                # model="Systran/faster-distil-whisper-large-v3", 
                model="Systran/faster-whisper-small", 
                file=audio.get_wav_data()
            ).text
            print(f"Você disse: {clean_prompt}")

            if wake_word.lower() in clean_prompt.lower():
                print("Wake word detectada!")
                
            # except Exception as e:
                # print(f"Errinho: {e}")

# Função principal para iniciar o listener em uma thread separada
def main():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    wake_word = "siri"  # Defina sua wake word aqui
    stop_event = threading.Event()

    listener_thread = threading.Thread(target=listen_for_wake_word, args=(recognizer, microphone, wake_word, stop_event))
    listener_thread.start()

    print("Pressione Enter para parar...")
    input()
    stop_event.set()
    listener_thread.join()

if __name__ == "__main__":
    main()
import queue
import re
import sys
import time
import threading
from google.cloud import speech
import pyaudio

# Audio recording parameters
STREAMING_LIMIT = 240000  # 4 minutes
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"

class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk_size):
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = self.get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._fill_buffer,
        )

    def get_current_time(self):
        """Return Current Time in MS."""
        return int(round(time.time() * 1000))

    def __enter__(self):
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, *args, **kwargs):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            data = []
            if self.new_stream and self.last_audio_input:
                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)
                if chunk_time != 0:
                    if self.bridging_offset < 0:
                        self.bridging_offset = 0
                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time
                    chunks_from_ms = round(
                        (self.final_request_end_time - self.bridging_offset) / chunk_time
                    )
                    self.bridging_offset = round(
                        (len(self.last_audio_input) - chunks_from_ms) * chunk_time
                    )
                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])
                self.new_stream = False
            chunk = self._buff.get()
            self.audio_input.append(chunk)
            if chunk is None:
                return
            data.append(chunk)
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

class GoogleSTT:
    def __init__(self):
        self.transcription = ""
        self.client = speech.SpeechClient()
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE,
                language_code="pt-BR",
                enable_automatic_punctuation=True,
            ),
            interim_results=True,
        )
        self.mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
        self.closed = False
        # self.start()

    def start(self):
        self.done = False
        def listen_print_loop(responses):
            for response in responses:
                if self.closed:
                    break
                if not response.results:
                    continue
                result = response.results[0]
                if not result.alternatives:
                    continue
                transcript = result.alternatives[0].transcript
                result_seconds = 0
                result_micros = 0
                if result.result_end_time.seconds:
                    result_seconds = result.result_end_time.seconds
                if result.result_end_time.microseconds:
                    result_micros = result.result_end_time.microseconds
                corrected_time = int((result_seconds * 1000) + (result_micros / 1000))
                
                if result.is_final:
                    self.done = True
                    self.transcription += transcript + " "
                    sys.stdout.write(GREEN)
                    sys.stdout.write("\033[K")
                    sys.stdout.write(str(corrected_time) + ": " + transcript + "\n")
                    if re.search(r"\b(exit|quit)\b", transcript, re.I):
                        sys.stdout.write(YELLOW)
                        sys.stdout.write("Exiting...\n")
                        self.closed = True
                        break
                else:
                    sys.stdout.write(RED)
                    sys.stdout.write("\033[K")
                    sys.stdout.write(str(corrected_time) + ": " + transcript + "\r")

        def stream_audio():
            with self.mic_manager as stream:
                while not self.closed:
                    stream.audio_input = []
                    audio_generator = stream.generator()
                    requests = (
                        speech.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator
                    )
                    responses = self.client.streaming_recognize(self.streaming_config, requests)
                    listen_print_loop(responses)
                    if stream.result_end_time > 0:
                        stream.final_request_end_time = stream.is_final_end_time
                    stream.result_end_time = 0
                    stream.last_audio_input = []
                    stream.last_audio_input = stream.audio_input
                    stream.audio_input = []
                    stream.restart_counter += 1
                    if not stream.last_transcript_was_final:
                        sys.stdout.write("\n")
                    stream.new_stream = True

        self._audio_thread = threading.Thread(target=stream_audio)
        self._audio_thread.start()

    def stop(self):
        while not self.done:
            time.sleep(0.1)
        self.closed = True
        self._audio_thread.join()
        
    
    def text(self): 
        return self.transcription


if __name__ == "__main__":
    stt = GoogleSTT()
    stt.start()
    print("Gravando... Diga 'exit' ou 'quit' para parar.")
    time.sleep(10)  # Aguarde 10 segundos de gravação
    stt.stop()
    print("Transcrição:", stt.transcription)


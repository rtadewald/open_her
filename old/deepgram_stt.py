import os
import pyaudio
import threading
from dotenv import load_dotenv
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

load_dotenv()

API_KEY = os.getenv("DG_API_KEY")


def main():
    try:
        # STEP 1: Create a Deepgram client using the API key
        deepgram = DeepgramClient(API_KEY)

        # STEP 2: Create a websocket connection to Deepgram
        dg_connection = deepgram.listen.live.v("1")

        # STEP 3: Define the event handlers for the connection
        def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            print(f"speaker: {sentence}")

        def on_metadata(self, metadata, **kwargs):
            print(f"\n\n{metadata}\n\n")

        def on_error(self, error, **kwargs):
            print(f"\n\n{error}\n\n")

        # STEP 4: Register the event handlers
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        # STEP 5: Configure Deepgram options for live transcription
        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            )

        # STEP 6: Start the connection
        dg_connection.start(options)

        # STEP 7: Create a lock and a flag for thread synchronization
        lock_exit = threading.Lock()
        exit_flag = False

        # STEP 8: Configure PyAudio to capture audio from the microphone
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

        # STEP 9: Define a thread that streams the audio and sends it to Deepgram
        def stream_audio():
            while True:
                data = stream.read(1024)
                lock_exit.acquire()
                if exit_flag:
                    break
                lock_exit.release()
                dg_connection.send(data)

        # STEP 10: Start the thread
        audio_thread = threading.Thread(target=stream_audio)
        audio_thread.start()

        # STEP 11: Wait for user input to stop recording
        input("Press Enter to stop recording...\n\n")

        # STEP 12: Set the exit flag to True to stop the thread
        lock_exit.acquire()
        exit_flag = True
        lock_exit.release()

        # STEP 13: Wait for the thread to finish
        audio_thread.join()

        # STEP 14: Close the connection to Deepgram and the audio stream
        dg_connection.finish()
        stream.stop_stream()
        stream.close()
        audio.terminate()

        print("Finished")

    except Exception as e:
        print(f"Could not open socket: {e}")
        return


if __name__ == "__main__":
    main()

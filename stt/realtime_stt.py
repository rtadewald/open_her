from RealtimeSTT import AudioToTextRecorder
import time


class RtSTT:
    def __init__(self) -> None:
        self.recorder = AudioToTextRecorder(
                        spinner=False, 
                        model="small", 
                        input_device_index=0, 
                        compute_type='int8',
                        enable_realtime_transcription=True, 
                        silero_use_onnx=True, 
                        realtime_model_type="small", 
                        language="pt")
        self.start()

    def start(self):
        self.recorder.start()
        print("Recording...")

    def stop(self):
        self.recorder.stop()
        print("Transcribing...")
        transcription = self.recorder.text()
        return transcription
        # print(self.transcription)
        # while self.start:
        #     print("here")
        #     print(self.recorder.text(), end=" ", flush=True)
        # print("done")


if __name__ == "__main__":
    stt = RtSTT()

    transcription = stt.stop()
    print(transcription)
    
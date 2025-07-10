import os
import time
import wave
import threading
import pyaudio
import keyboard
import requests
import pyperclip

from dataclasses import dataclass
from typing import Optional


@dataclass
class RecordingConfig:
    output_file: str = "output.wav"
    min_duration: float = 0.3
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 48000
    chunk: int = 1024
    device_name: str = "USB Condenser Microphone"
    server_url: str = "http://localhost:8000/transcribe/"


class AudioRecorder:
    def __init__(self, config: RecordingConfig):
        self.config = config
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.lock = threading.Lock()
        self.is_recording = False
        self.is_sending = False
        self.start_time = 0
        self.device_index = self._find_device_index()

    def _find_device_index(self) -> int:
        for i in range(self.audio.get_device_count()):
            dev_info = self.audio.get_device_info_by_index(i)
            if self.config.device_name in dev_info["name"]:
                return i
        raise ValueError(f"Device '{self.config.device_name}' not found")

    def start_recording(self) -> None:
        with self.lock:
            if self.is_recording or self.is_sending:
                return
            try:
                self.stream = self.audio.open(
                    format=self.config.format,
                    channels=self.config.channels,
                    rate=self.config.rate,
                    input=True,
                    frames_per_buffer=self.config.chunk,
                    input_device_index=self.device_index,
                )
                self.frames.clear()
                self.is_recording = True
                self.start_time = time.time()
                print(f"Recording started with {self.config.device_name}...")
            except Exception as e:
                print(f"Failed to start recording: {e}")
                self.is_recording = False

    def stop_recording(self) -> None:
        with self.lock:
            if not self.is_recording or not self.stream:
                return
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                self.is_recording = False

                duration = time.time() - self.start_time
                print(f"Recording stopped. Duration: {duration:.2f}s")

                if duration >= self.config.min_duration:
                    self._save_audio()
                    self._send_audio()
                else:
                    print(
                        f"Recording too short (<{self.config.min_duration}s). Discarded."
                    )
                    self._cleanup()
            except Exception as e:
                print(f"Error stopping recording: {e}")
                self._cleanup()

    def _save_audio(self) -> None:
        try:
            with wave.open(self.config.output_file, "wb") as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.config.format))
                wf.setframerate(self.config.rate)
                wf.writeframes(b"".join(self.frames))
        except Exception as e:
            print(f"Failed to save audio: {e}")

    def _send_audio(self) -> None:
        self.is_sending = True
        try:
            with open(self.config.output_file, "rb") as f:
                response = requests.post(
                    self.config.server_url, files={"file": f}
                )
                response.raise_for_status()

            text = response.json().get("text", "No text returned")
            print(f"Transcription: {text}\n")
            self._copy_to_clipboard_and_paste(text)
        except requests.RequestException as e:
            print(f"Request failed: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            self._cleanup()
            self.is_sending = False

    def _copy_to_clipboard_and_paste(self, text: str) -> None:
        original = None
        try:
            try:
                original = pyperclip.paste()
            except Exception:
                # Clipboard is empty or inaccessible - we'll proceed anyway
                pass

            pyperclip.copy(text)
            keyboard.send("ctrl+shift+v")
            time.sleep(0.1)  # Small delay to ensure paste completes

            if original is not None:
                pyperclip.copy(original)

        except Exception as e:
            print(f"Clipboard error: {e}")
            if original is not None:
                try:
                    pyperclip.copy(original)
                except Exception:
                    pass

    def _cleanup(self) -> None:
        try:
            if os.path.exists(self.config.output_file):
                os.remove(self.config.output_file)
        except Exception as e:
            print(f"Failed to clean up output file: {e}")

    def _record_loop(self):
        while True:
            with self.lock:
                if self.is_recording and self.stream:
                    try:
                        data = self.stream.read(
                            self.config.chunk, exception_on_overflow=False
                        )
                        self.frames.append(data)
                    except Exception as e:
                        print(f"Stream read error: {e}")
                        self.stop_recording()
            time.sleep(0.01)

    def run(self) -> None:
        print(
            "Press and hold Num Lock to record, release to stop. Ctrl+Shift+Esc to exit."
        )

        keyboard.on_press_key("num lock", lambda _: self.start_recording())
        keyboard.on_release_key("num lock", lambda _: self.stop_recording())

        try:
            self._record_loop()
        except KeyboardInterrupt:
            print("\nExiting...")
            self.stop_recording()
        finally:
            self._cleanup()
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.audio.terminate()
            keyboard.unhook_all()


def main():
    config = RecordingConfig()
    AudioRecorder(config).run()


if __name__ == "__main__":
    main()

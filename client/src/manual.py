import os
import time
import threading
import requests
import pyperclip
import keyboard
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from dataclasses import dataclass

@dataclass
class RecordingConfig:
    output_file: str = "output.wav"
    min_duration: float = 0.3
    rate: int = 48000          # 保持原参数
    channels: int = 1
    chunk: int = 1024
    device_name: str = "USB Condenser Microphone"
    server_url: str = "http://localhost:8000/transcribe/"

class AudioRecorder:
    def __init__(self, config: RecordingConfig):
        self.config = config
        self.is_recording = False
        self.is_sending = False
        self.frames = []
        self.start_time = 0
        self.lock = threading.Lock()
        self.device_index = self._find_device_index()

    def _find_device_index(self):
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if self.config.device_name in dev['name']:
                return i
        print(f"警告：未找到指定设备 '{self.config.device_name}'，将使用默认输入设备")
        return None

    def start_recording(self):
        with self.lock:
            if self.is_recording or self.is_sending:
                return
            self.frames.clear()
            self.is_recording = True
            self.start_time = time.time()
            print("录音开始...")

    def stop_recording(self):
        with self.lock:
            if not self.is_recording:
                return
            self.is_recording = False
            duration = time.time() - self.start_time
            print(f"录音结束，时长：{duration:.2f}s")

            if duration >= self.config.min_duration and self.frames:
                self._save_audio()
                threading.Thread(target=self._send_audio, daemon=True).start()
            else:
                print(f"录音太短 (< {self.config.min_duration}s)，已丢弃。")
                self._cleanup()

    def _record_callback(self, indata, frames, time_, status):
        if status:
            print(status)
        if self.is_recording:
            self.frames.append(indata.copy())

    def _save_audio(self):
        try:
            audio_data = np.concatenate(self.frames, axis=0)
            write(self.config.output_file, self.config.rate, audio_data)
            print(f"音频已保存为 {self.config.output_file}")
        except Exception as e:
            print(f"保存音频失败：{e}")

    def _send_audio(self):
        self.is_sending = True
        try:
            with open(self.config.output_file, "rb") as f:
                response = requests.post(self.config.server_url, files={"file": f})
                response.raise_for_status()
            text = response.json().get("text", "未返回文字")
            print(f"识别结果：{text}")
            self._copy_to_clipboard_and_paste(text)
        except Exception as e:
            print(f"发送失败：{e}")
        finally:
            self._cleanup()
            self.is_sending = False

    def _copy_to_clipboard_and_paste(self, text):
        original = None
        try:
            try:
                original = pyperclip.paste()
            except:
                pass
            pyperclip.copy(text)
            keyboard.send("ctrl+shift+v")
            time.sleep(0.1)
            if original:
                pyperclip.copy(original)
        except Exception as e:
            print(f"剪贴板操作失败：{e}")

    def _cleanup(self):
        try:
            if os.path.exists(self.config.output_file):
                os.remove(self.config.output_file)
        except Exception as e:
            print(f"清理文件失败：{e}")

    def run(self):
        print("按住 Num Lock 开始录音，松开停止录音。按 Ctrl+C 退出。")
        keyboard.on_press_key("num lock", lambda _: self.start_recording())
        keyboard.on_release_key("num lock", lambda _: self.stop_recording())

        try:
            with sd.InputStream(samplerate=self.config.rate,
                                channels=self.config.channels,
                                device=self.device_index,
                                blocksize=self.config.chunk,
                                callback=self._record_callback):
                while True:
                    time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n退出程序...")
            self.stop_recording()
        finally:
            self._cleanup()

def main():
    config = RecordingConfig()
    AudioRecorder(config).run()

if __name__ == "__main__":
    main()

import pyaudio
import wave
import webrtcvad
import os
import shutil
import time
import numpy as np
import requests
import pyperclip
import keyboard
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path
from io import BytesIO


@dataclass
class AudioConfig:
    """Audio configuration settings."""

    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 48000
    chunk: int = 1440
    vad_frame_samples: int = 1440
    vad_mode: int = 3
    volume_threshold: int = 100
    silence_threshold_seconds: float = 0.5
    min_recording_duration: float = 0
    device_name: str = "USB Condenser Microphone"


@dataclass
class RecordingState:
    """State management for recording process."""

    segment_number: int = 1
    buffer: List[bytes] = None
    is_speaking: bool = False
    silence_duration: float = 0
    last_voice_time: float = 0
    has_detected_voice: bool = False
    recording_start_time: float = 0


class VoiceRecorder:
    """Manages voice recording and transcription process."""

    def __init__(
        self,
        config: AudioConfig,
        server_url: str = "http://localhost:8000/transcribe/",
    ):
        self.config = config
        self.server_url = server_url
        self.audio = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(config.vad_mode)
        self.state = RecordingState(buffer=[])
        self.device_index = self._get_device_index()

    def _get_device_index(self) -> int:
        """Find the device index for the specified device name."""
        for i in range(self.audio.get_device_count()):
            dev_info = self.audio.get_device_info_by_index(i)
            if self.config.device_name in dev_info["name"]:
                return i
        raise ValueError(f"Device '{self.config.device_name}' not found")

    def calculate_rms(self, audio_data: np.ndarray) -> float:
        """Calculate RMS value of audio data."""
        if len(audio_data) == 0:
            return 0.0
        squared = np.square(audio_data.astype(np.float64))
        mean = np.mean(squared)
        return np.sqrt(mean) if not (np.isnan(mean) or np.isinf(mean)) else 0.0

    def save_segment(self) -> Optional[bytes]:
        """Prepare audio segment for transcription."""
        if not self.state.buffer or not self.state.has_detected_voice:
            return None

        # Create WAV data in memory
        buffer = BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.config.format))
            wf.setframerate(self.config.rate)
            wf.writeframes(b"".join(self.state.buffer))

        print(f"Prepared segment: {self.state.segment_number:03d}.wav")
        return buffer.getvalue()

    def transcribe_segment(self, audio_data: Optional[bytes]) -> None:
        """Send audio data to server for transcription and handle clipboard."""
        if not audio_data:
            return

        try:
            buffer = BytesIO(audio_data)
            buffer.name = f"{self.state.segment_number:03d}.wav"  # Name for compatibility
            response = requests.post(
                self.server_url, files={"file": buffer}, timeout=10
            )
            response.raise_for_status()
            transcribed_text = response.json().get("text", "No text returned")
            if not transcribed_text.strip():
                return

            print(f"Server response: {transcribed_text}")
            self._handle_clipboard(transcribed_text)
        except requests.RequestException as e:
            print(f"Failed to send recording: {e}")
        except Exception as e:
            print(f"Unexpected error during transcription: {e}")

    def _handle_clipboard(self, text: str) -> None:
        try:
            original_clipboard = pyperclip.paste()
        except Exception as e:
            print(f"Error reading clipboard: {e}")
            original_clipboard = ""

        try:
            pyperclip.copy(text)
            keyboard.send("ctrl+shift+v")
            if original_clipboard:
                pyperclip.copy(original_clipboard)
        except Exception as e:
            print(f"Error handling clipboard: {e}")

    def process_audio_chunk(self, data: bytes, stream: pyaudio.Stream) -> None:
        """Process a single chunk of audio data."""
        audio_data = np.frombuffer(data, dtype=np.int16)
        rms = self.calculate_rms(audio_data)
        current_time = time.time()

        # VAD processing
        is_speech_detected = False
        if len(data) == self.config.vad_frame_samples * 2:
            try:
                is_speech_detected = (
                    self.vad.is_speech(data, self.config.rate)
                    and rms > self.config.volume_threshold
                )
            except webrtcvad.Error as e:
                print(f"VAD error: {e}")
                return

        # State management
        if not self.state.has_detected_voice:
            if is_speech_detected:
                self.state.has_detected_voice = True
                self.state.is_speaking = True
                self.state.last_voice_time = current_time
                self.state.recording_start_time = current_time
                self.state.buffer.append(data)
                print(
                    f"Voice detected (RMS: {rms:.2f}), starting to record segment..."
                )
        else:
            self.state.buffer.append(data)
            if is_speech_detected:
                self.state.is_speaking = True
                self.state.last_voice_time = current_time
                self.state.silence_duration = 0
            elif self.state.is_speaking:
                self.state.silence_duration = (
                    current_time - self.state.last_voice_time
                )
                if (
                    self.state.silence_duration
                    >= self.config.silence_threshold_seconds
                ):
                    self._handle_segment_end(current_time)

    def _handle_segment_end(self, current_time: float) -> None:
        """Handle the end of a recording segment."""
        recording_duration = current_time - self.state.recording_start_time
        if recording_duration >= self.config.min_recording_duration:
            audio_data = self.save_segment()
            if audio_data:
                self.transcribe_segment(audio_data)
                self.state.segment_number += 1
                print(f"Recording duration: {recording_duration:.2f}s - Saved")
        else:
            print(
                f"Recording duration: {recording_duration:.2f}s - Too short, discarded"
            )

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset recording state."""
        self.state.buffer = []
        self.state.is_speaking = False
        self.state.has_detected_voice = False
        self.state.recording_start_time = 0
        print("Waiting for next voice detection...\n")

    def run(self) -> None:
        """Main recording loop."""
        stream = self.audio.open(
            format=self.config.format,
            channels=self.config.channels,
            rate=self.config.rate,
            input=True,
            frames_per_buffer=self.config.chunk,
            input_device_index=self.device_index,
        )

        print("Recording started... Waiting for voice detection...")

        try:
            while True:
                data = stream.read(
                    self.config.chunk, exception_on_overflow=False
                )
                self.process_audio_chunk(data, stream)

        except KeyboardInterrupt:
            print("\nRecording stopped")
            self._handle_final_segment()
        finally:
            stream.stop_stream()
            stream.close()
            self.audio.terminate()

    def _handle_final_segment(self) -> None:
        """Process final audio segment on termination."""
        if self.state.buffer and self.state.has_detected_voice:
            recording_duration = time.time() - self.state.recording_start_time
            if recording_duration >= self.config.min_recording_duration:
                audio_data = self.save_segment()
                if audio_data:
                    self.transcribe_segment(audio_data)
                    print(
                        f"Final recording duration: {recording_duration:.2f}s - Saved"
                    )
            else:
                print(
                    f"Final recording duration: {recording_duration:.2f}s - Too short, discarded"
                )


def main():
    """Entry point for the voice recorder application."""
    config = AudioConfig()
    recorder = VoiceRecorder(config)
    recorder._handle_clipboard("initial")
    recorder.run()


if __name__ == "__main__":
    main()

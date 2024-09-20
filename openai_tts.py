import os
import numpy as np
import librosa
# from scipy.io import wavfile
from openai import OpenAI

class OpenAITTSAPI(object):  # Set credential first: export GOOGLE_APPLICATION_CREDENTIALS="KEY_PATH"
    def __init__(self, language_code="en-US", sr=16000) -> None:
        self.client = OpenAI()
        self.sr = sr
    
    def sample_rate(self):
        return self.sr

    def set_speaker(self, speaker: str):
        self.voice_name = speaker

    def wav_normalization(self, wav: np.array) -> np.array:
        denom = max(abs(wav))
        if denom == 0 or np.isnan(denom):
            raise ValueError
        return wav / denom
    
    def text_to_wav(self, text: str, voice_name="alloy", config={}):
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=voice_name,
            input=text,
            response_format="wav",
        )

        audio_bytes = b""
        for data in response.response.iter_bytes():
            audio_bytes += data
        wav = np.frombuffer(audio_bytes, dtype=np.int16)
        # print(wav.shape)
        # wavfile.write("check0.wav", 24000, wav)

        # resample
        wav = wav.astype(np.float32) / 32767
        wav = librosa.resample(wav, orig_sr=24000, target_sr=self.sr)

        wav = self.wav_normalization(wav)
        
        return wav


if __name__ == "__main__":
    api = OpenAITTSAPI(sr=16000)
    wav = api.text_to_wav("Hello world! Deep learning is fun.", voice_name="alloy")
    # wav = wav_normalization(wav)
    # wavfile.write("check.wav", api.sample_rate(), wav)

    # wav = (wav * 32767).astype(np.int16)
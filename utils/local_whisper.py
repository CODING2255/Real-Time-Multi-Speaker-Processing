import whisper
from tempfile import NamedTemporaryFile

model = whisper.load_model("base")  # You can use "small", "medium", or "large" for more accuracy

def transcribe_audio_whisper(audio_bytes):
    with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio.flush()
        result = model.transcribe(temp_audio.name)
        return result["text"]

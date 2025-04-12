import os
import tempfile
import soundfile as sf
import torchaudio

class AudioProcessor:
    def standardize_audio(self, audio_input):
        """
        Accepts either a file path (str) or a file-like object (BytesIO),
        and outputs a standardized wav file path.
        """
        if hasattr(audio_input, "getvalue"):
            # Handle BytesIO or uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
                tmp_in.write(audio_input.getvalue())
                tmp_input_path = tmp_in.name
        elif isinstance(audio_input, str) and os.path.isfile(audio_input):
            tmp_input_path = audio_input
        else:
            raise ValueError("Unsupported audio input type")

        # Convert and standardize using torchaudio
        waveform, sr = torchaudio.load(tmp_input_path)
        standardized_path = tmp_input_path.replace(".wav", "_standardized.wav")
        torchaudio.save(standardized_path, waveform, sr)

        return standardized_path

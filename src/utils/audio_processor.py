"""
Audio Processing Utilities
Handles audio file preprocessing and standardization.
"""

from pydub import AudioSegment
import io
import tempfile
import os

class AudioProcessor:
    @staticmethod
    def standardize_audio(audio_file):
        """Standardize audio file to required format.
        
        Args:
            audio_file: Uploaded audio file
            
        Returns:
            str: Path to processed audio file
        """
        try:
            audio_bytes = io.BytesIO(audio_file.getvalue())
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                if audio_file.name.lower().endswith('.mp3'):
                    audio = AudioSegment.from_mp3(audio_bytes)
                else:
                    audio = AudioSegment.from_wav(audio_bytes)
                
                audio = audio.set_frame_rate(16000)
                audio = audio.set_channels(1)
                audio = audio.set_sample_width(2)
                
                audio.export(
                    tmp.name,
                    format="wav",
                    parameters=["-ac", "1", "-ar", "16000"]
                )
                return tmp.name
                
        except Exception as e:
            raise Exception(f"Error processing audio: {str(e)}")
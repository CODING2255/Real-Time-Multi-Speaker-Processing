"""
Speaker Diarization Model Handler
Manages the pyannote-audio model for speaker diarization tasks.
"""

from pyannote.audio import Pipeline
import streamlit as st

class SpeakerDiarizer:
    def __init__(self, token: str):
        """Initialize the diarization model.
        
        Args:
            token (str): HuggingFace authentication token
        """
        self.token = token
        self.model = None

    def load_model(self):
        """Load the pyannote speaker diarization model."""
        try:
            self.model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=self.token
            )
            return self.model
        except Exception as e:
            st.error(f"Error loading diarization model: {str(e)}")
            return None

    def process(self, audio_path: str):
        """Process audio file for speaker diarization.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Diarization results
        """
        try:
            return self.model(audio_path)
        except Exception as e:
            st.error(f"Error in diarization: {str(e)}")
            return None
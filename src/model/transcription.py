"""
Transcription Model Handler
Manages the Whisper model for speech-to-text transcription.
"""

import whisper
import streamlit as st

class Transcriber:
    def __init__(self):
        """Initialize the transcription model."""
        self.model = None

    def load_model(self):
        """Load the Whisper transcription model."""
        try:
            self.model = whisper.load_model("base")
            return self.model
        except Exception as e:
            st.error(f"Error loading transcription model: {str(e)}")
            return None

    def process(self, audio_path: str):
        """Process audio file for transcription.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Transcription results
        """
        try:
            return self.model.transcribe(audio_path)
        except Exception as e:
            st.error(f"Error in transcription: {str(e)}")
            return None
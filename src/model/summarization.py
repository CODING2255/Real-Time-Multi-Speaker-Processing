"""
Summarization Model Handler
Manages the BART model for text summarization.
"""

from transformers import pipeline
import torch
import streamlit as st

class Summarizer:
    def __init__(self):
        """Initialize the summarization model."""
        self.model = None

    def load_model(self):
        """Load the BART summarization model."""
        try:
            self.model = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
            return self.model
        except Exception as e:
            st.error(f"Error loading summarization model: {str(e)}")
            return None

    def process(self, text: str, max_length: int = 130, min_length: int = 30):
        """Process text for summarization.
        
        Args:
            text (str): Text to summarize
            max_length (int): Maximum length of summary
            min_length (int): Minimum length of summary
            
        Returns:
            list: List containing summary dictionary
        """
        try:
            if not isinstance(text, str):
                raise ValueError("Input must be a string")
            
            if not text.strip():
                raise ValueError("Input text is empty")
                
            summary = self.model(text, max_length=max_length, min_length=min_length)
            return summary
        except Exception as e:
            st.error(f"Error in summarization: {str(e)}")
            return None
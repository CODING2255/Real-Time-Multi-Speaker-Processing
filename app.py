import streamlit as st
import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt
import torch
import torchaudio
from pydub import AudioSegment
import ffmpeg

from src.model.diarization import SpeakerDiarizer
from src.model.transcription import Transcriber
from src.model.summarization import Summarizer
from src.utils.audio_processor import AudioProcessor as MyAudioProcessor
from src.utils.formatter import TimeFormatter

# 🔹 Fix RuntimeError: No Running Event Loop
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(page_title="Multi-Speaker Audio Analyzer", layout="wide")
st.title("Multi-Speaker Audio Analyzer")

st.write("Upload an audio file for speaker diarization, transcription, and summarization.")
uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav"])

@st.cache_resource
def load_models():
    try:
        diarizer = SpeakerDiarizer(st.secrets.get("hf_token", ""))
        transcriber = Transcriber()
        summarizer = Summarizer()

        diarizer_model = diarizer.load_model()
        transcriber_model = transcriber.load_model()
        summarizer_model = summarizer.load_model()

        if not all([diarizer_model, transcriber_model, summarizer_model]):
            raise ValueError("One or more models failed to load")

        return diarizer, transcriber, summarizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def process_audio(audio_file):
    try:
        # Convert audio file using pydub
        audio = AudioSegment.from_file(audio_file)
        audio.export("processed_audio.wav", format="wav")

        audio_processor = MyAudioProcessor()
        standardized_path = audio_processor.standardize_audio("processed_audio.wav")

        diarizer, transcriber, summarizer = load_models()
        if not all([diarizer, transcriber, summarizer]):
            return None

        with st.spinner("🧠 Identifying speakers..."):
            diarization_result = diarizer.process(standardized_path)

        with st.spinner("✍️ Transcribing audio..."):
            transcription = transcriber.process(standardized_path)

        with st.spinner("📝 Generating summary..."):
            summary = summarizer.process(transcription["text"])

        if os.path.exists(standardized_path):
            os.unlink(standardized_path)

        return {
            "diarization": diarization_result,
            "transcription": transcription,
            "summary": summary[0]["summary_text"]
        }

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def display_results(results):
    tab1, tab2, tab3 = st.tabs(["Speakers", "Transcription", "Summary"])

    with tab1:
        st.write("Speaker Timeline:")
        segments = TimeFormatter.format_speaker_segments(results["diarization"], results["transcription"])
        if segments:
            for segment in segments:
                col1, col2, col3 = st.columns([2, 3, 5])
                with col1:
                    speaker_num = int(segment['speaker'].split('_')[1])
                    st.write(f"Speaker {speaker_num}")
                with col2:
                    st.write(f"{TimeFormatter.format_timestamp(segment['start'])} -> {TimeFormatter.format_timestamp(segment['end'])}")
                with col3:
                    st.write(f"\"{segment['text']}\"" if segment['text'] else "(no speech detected)")
                st.markdown("---")
        else:
            st.warning("No speaker segments detected")

    with tab2:
        st.write("Transcription:")
        st.write(results["transcription"].get("text", "No transcription available."))

    with tab3:
        st.write("Summary:")
        st.write(results.get("summary", "No summary available."))

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    if st.button("Analyze Uploaded Audio"):
        results = process_audio(uploaded_file)
        if results:
            display_results(results)
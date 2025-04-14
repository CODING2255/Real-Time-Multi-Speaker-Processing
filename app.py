import streamlit as st
from audiorecorder import audiorecorder
import os

from utils.local_whisper import transcribe_audio_whisper
from utils.summarizer import summarize_text
from utils.diarizer import simulate_speaker_diarization

st.set_page_config(page_title="Audio To Text", layout="centered")
st.title("🎙️ Audio To Transcriber + Summarizer + Speaker Diarization")

# Choose input method
input_method = st.radio("Choose input method:", ["Upload Audio", "Record Live"], horizontal=True)

if input_method == "Upload Audio":
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])

    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Transcribe, Summarize & Diarize"):
            with st.spinner("Transcribing..."):
                audio_bytes = uploaded_file.read()
                transcription = transcribe_audio_whisper(audio_bytes)

            st.subheader("📝 Transcription")
            st.write(transcription)

            with st.spinner("Summarizing..."):
                summary = summarize_text(transcription)

            st.subheader("📌 Summary")
            st.write(summary)

            with st.spinner("Simulating Speaker Diarization..."):
                speakers = simulate_speaker_diarization(transcription)

            st.subheader("🗣️ Speaker Segments")
            for speaker, speech in speakers:
                st.markdown(f"**{speaker}**")
                st.write(speech)

elif input_method == "Record Live":
    st.info("Click 'Start recording' to record your audio using your microphone.")

    # Record audio
    audio = audiorecorder("Start recording", "Stop recording")

    if len(audio) > 0:
        # Save audio to file
        os.makedirs("data", exist_ok=True)
        audio_path = os.path.join("data", "live_audio.wav")
        audio.export(audio_path, format="wav")
        st.audio(audio.export().read())

        if st.button("Transcribe, Summarize & Diarize", key="process_live"):
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            with st.spinner("Transcribing..."):
                transcription = transcribe_audio_whisper(audio_bytes)

            st.subheader("📝 Transcription")
            st.write(transcription)

            with st.spinner("Summarizing..."):
                summary = summarize_text(transcription)

            st.subheader("📌 Summary")
            st.write(summary)

            with st.spinner("Simulating Speaker Diarization..."):
                speakers = simulate_speaker_diarization(transcription)

            st.subheader("🗣️ Speaker Segments")
            for speaker, speech in speakers:
                st.markdown(f"**{speaker}**")
                st.write(speech)

# utils/diarizer.py

def simulate_speaker_diarization(transcript):
    """
    Simulates speaker diarization by splitting sentences and assigning them
    alternately to Speaker 1 and Speaker 2.

    Returns:
        A list of (speaker, sentence) tuples in chronological order.
    """
    # Split by period and filter empty parts
    sentences = [s.strip() for s in transcript.split(". ") if s.strip()]
    
    speaker_chunks = []
    speaker = 1

    for i, sentence in enumerate(sentences):
        speaker_name = f"Speaker {speaker}"
        # Ensure sentence ends with a period
        formatted = sentence if sentence.endswith(".") else sentence + "."
        speaker_chunks.append((speaker_name, formatted))

        # Alternate speaker every 2 sentences
        if (i + 1) % 2 == 0:
            speaker = 2 if speaker == 1 else 1

    return speaker_chunks

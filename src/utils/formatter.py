"""
Formatting utilities for timestamps and speaker segments.
"""

class TimeFormatter:
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Format seconds into MM:SS.ss format.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted time string
        """
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:05.2f}"

    @staticmethod
    def format_speaker_segments(diarization_result, transcription):
        """Format speaker segments with transcribed text.
        
        Args:
            diarization_result: Diarization model output
            transcription: Whisper transcription output
            
        Returns:
            list: Formatted speaker segments
        """
        if diarization_result is None:
            return []
            
        formatted_segments = []
        whisper_segments = transcription.get('segments', [])
        
        try:
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                current_text = ""
                for w_segment in whisper_segments:
                    w_start = float(w_segment['start'])
                    w_end = float(w_segment['end'])
                    
                    if (w_start >= turn.start and w_start < turn.end) or \
                       (w_end > turn.start and w_end <= turn.end):
                        current_text += w_segment['text'].strip() + " "
                
                formatted_segments.append({
                    'speaker': str(speaker),
                    'start': float(turn.start),
                    'end': float(turn.end),
                    'text': current_text.strip()
                })
                
        except Exception as e:
            print(f"Error formatting segments: {str(e)}")
            return []
        
        return formatted_segments
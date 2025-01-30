import numpy as np
import librosa

def load_audio(file_path: str, sample_rate: int = 22050) -> np.ndarray:
    """Load and normalize an audio file.
    
    Args:
        file_path: Path to the audio file
        sample_rate: Target sample rate
        
    Returns:
        Normalized audio array
    """
    audio, _ = librosa.load(file_path, sr=sample_rate)
    
    # Handle short audio files
    if audio.shape[0] < 16000:
        print(f"(Almost) Empty audio file: {file_path}, padding with zeros.")
        audio = np.pad(audio, (0, 16000 - audio.shape[0]))
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    
    return audio 
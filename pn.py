import sounddevice as sd
import scipy.io.wavfile as wavfile
import librosa
import numpy as np
def record_voice(duration, filename):
    """
    Record the user's voice for the given duration and save it to the specified file.
    
    Args:
        duration (float): Duration of the recording in seconds.
        filename (str): Path to the file where the audio will be saved.
    """
    sample_rate = 44100  # Sample rate in Hz
    channels = 1  # Mono

    print(f"Recording voice for {duration} seconds...")

    # Record audio
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()  # Wait until recording is finished

    print("Recording complete. Saving audio file...")
    return audio_data.flatten()

    
 # Define the function to extract features from the audio file
def extract_features(audio_file):
    """
    Extract the specified features from the given audio file.
    
    Args:
        audio_file (str): Path to the audio file from which features will be extracted.
    
    Returns:
        dict: A dictionary containing the extracted features.
    """
    # y, sr = librosa.load(audio_file, sr=None)
    y=audio_file
    sr=44100
    # Define a helper function to handle nan values
    def handle_nan(value):
        return value if not np.isnan(value) else 0
    
    # Calculate features as in the previous code
    fo, voiced_flag, voice_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    mdvp_fo = handle_nan(np.mean(fo))
    mdvp_fhi = handle_nan(np.max(fo))
    mdvp_flo = handle_nan(np.min(fo))
    
    jitter_percentage = handle_nan(np.std(fo) / np.mean(fo) * 100)
    jitter_abs = handle_nan(np.std(fo))
    jitter_ddp = handle_nan(np.mean(np.abs(np.diff(np.diff(fo)))))
    jitter_abs_ms = jitter_abs * 1000
    
    shimmer_db = handle_nan(np.mean(librosa.amplitude_to_db(np.abs(y))))
    shimmer_apq3 = handle_nan(librosa.feature.spectral_flatness(y=y).mean())
    shimmer_apq5 = handle_nan(librosa.feature.spectral_flatness(y=y, n_fft=1024).mean())
    shimmer_dda = handle_nan(librosa.feature.spectral_contrast(y=y).mean())
    
    mdvp_rap = handle_nan(librosa.feature.spectral_flatness(y=y).mean())
    mdvp_ppq = handle_nan(librosa.feature.spectral_bandwidth(y=y).mean())
    mdvp_apq = handle_nan(librosa.feature.spectral_centroid(y=y).mean())
    
    nhr = handle_nan(librosa.feature.spectral_rolloff(y=y).mean())
    hnr = handle_nan(librosa.feature.rms(y=y).mean())
    
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    rpde = handle_nan(tempogram.mean())
    
    dfa = handle_nan(librosa.feature.spectral_rolloff(y=y, roll_percent=0.85).mean())
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    spread1 = handle_nan(np.mean(mfccs[0, :]))
    spread2 = handle_nan(np.mean(mfccs[1, :]))
    
    d2 = handle_nan(librosa.feature.spectral_centroid(y=y).mean())
    
    ppe = handle_nan(librosa.feature.zero_crossing_rate(y=y).mean())
    
    # Return all features in the specified order
    features = {
        "MDVP:Fo(Hz)": mdvp_fo,
        "MDVP:Fhi(Hz)": mdvp_fhi,
        "MDVP:Flo(Hz)": mdvp_flo,
        "MDVP:Jitter(%)": jitter_percentage,
        "MDVP:Jitter(Abs)": jitter_abs_ms,
        "MDVP:RAP": mdvp_rap,
        "MDVP:PPQ": mdvp_ppq,
        "Jitter:DDP": jitter_ddp,
        "MDVP:Shimmer": shimmer_db,
        "MDVP:Shimmer(dB)": shimmer_db,
        "Shimmer:APQ3": shimmer_apq3,
        "Shimmer:APQ5": shimmer_apq5,
        "MDVP:APQ": mdvp_apq,
        "Shimmer:DDA": shimmer_dda,
        "NHR": nhr,
        "HNR": hnr,
        "RPDE": rpde,
        "DFA": dfa,
        "spread1": spread1,
        "spread2": spread2,
        "D2": d2,
        "PPE": ppe
    }
    
    return features
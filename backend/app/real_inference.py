"""
Lightweight Real-time Speech Emotion Recognition and Sarcasm Detection
Optimized to avoid PyTorch import issues
"""
import io
import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

# Lazy loading - models loaded on first use
_whisper_model = None
_emotion_classifier = None
_sentiment_analyzer = None

def get_whisper_model():
    """Lazy load Whisper model"""
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model (tiny) for speech-to-text...")
        import whisper
        _whisper_model = whisper.load_model("tiny")
        print("Whisper model loaded!")
    return _whisper_model

def get_emotion_classifier():
    """Lazy load emotion classifier"""
    global _emotion_classifier
    if _emotion_classifier is None:
        print("Loading emotion classifier...")
        from transformers import pipeline
        _emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=-1  # Force CPU
        )
        print("Emotion classifier loaded!")
    return _emotion_classifier

def get_sentiment_analyzer():
    """Lazy load sentiment analyzer"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        print("Loading sentiment analyzer...")
        from transformers import pipeline
        _sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # Force CPU
        )
        print("Sentiment analyzer loaded!")
    return _sentiment_analyzer

def extract_audio_features(audio_data, sr):
    """Extract acoustic features from audio"""
    features = {}
    
    try:
        # Pitch
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        features['pitch_mean'] = float(pitch_mean)
        
        # Energy
        rms = librosa.feature.rms(y=audio_data)[0]
        features['energy_mean'] = float(np.mean(rms))
        features['energy_std'] = float(np.std(rms))
        
        # Speaking rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features['speaking_rate'] = float(np.mean(zcr))
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = float(np.mean(mfccs))
        features['mfcc_std'] = float(np.std(mfccs))
    except Exception as e:
        print(f"Warning: Error extracting audio features: {e}")
    
    return features

def detect_sarcasm(transcript, audio_features, sentiment_result):
    """Detect sarcasm using multiple signals"""
    sarcasm_score = 0.0
    indicators = []
    
    try:
        from textblob import TextBlob
        text_sentiment = TextBlob(transcript).sentiment.polarity
        
        model_sentiment = 1.0 if sentiment_result[0]['label'] == 'POSITIVE' else -1.0
        
        # Exaggeration detection
        if text_sentiment > 0.3 and model_sentiment > 0:
            exaggeration_words = ['absolutely', 'totally', 'completely', 'perfect', 'amazing', 'fantastic', 'wonderful', 'great']
            if any(word in transcript.lower() for word in exaggeration_words):
                sarcasm_score += 0.3
                indicators.append("exaggeration")
        
        # Prosodic features
        if audio_features.get('pitch_mean', 0) > 200:
            sarcasm_score += 0.2
            indicators.append("high_pitch")
        
        if audio_features.get('energy_mean', 0) < 0.02 and text_sentiment > 0:
            sarcasm_score += 0.2
            indicators.append("low_energy_positive_text")
        
        # Sarcastic phrases
        sarcasm_phrases = [
            'oh great', 'oh wonderful', 'just what i needed', 'yeah right',
            'sure', 'of course', 'obviously', 'clearly'
        ]
        if any(phrase in transcript.lower() for phrase in sarcasm_phrases):
            sarcasm_score += 0.4
            indicators.append("sarcastic_phrase")
        
        sarcasm_score = min(sarcasm_score, 1.0)
        
    except Exception as e:
        print(f"Warning: Error in sarcasm detection: {e}")
        sarcasm_score = 0.0
    
    is_sarcastic = sarcasm_score > 0.5
    return is_sarcastic, sarcasm_score, indicators

def run_inference(audio_bytes: bytes, sample_rate: int = None):
    """
    Real-time inference for Speech Emotion Recognition and Sarcasm Detection.
    
    Args:
        audio_bytes: Raw audio bytes
        sample_rate: Target sample rate (None = auto-detect)
    """
    print(f"\n{'='*60}")
    print(f"Processing audio: {len(audio_bytes)} bytes" + (f" at {sample_rate}Hz" if sample_rate else " (auto-detect SR)"))
    
    try:
        # Load audio - let librosa auto-detect sample rate for uploaded files
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        print(f"Audio loaded: {len(audio_data)} samples, {sr}Hz, duration: {len(audio_data)/sr:.2f}s")
        
        # Ensure minimum length
        min_samples = int(0.5 * sr)
        if len(audio_data) < min_samples:
            print(f"Warning: Audio too short, padding...")
            audio_data = np.pad(audio_data, (0, min_samples - len(audio_data)), mode='constant')
        
        # 1. Speech-to-Text
        print("Step 1/4: Transcribing audio...")
        import tempfile
        import soundfile as sf
        import os
        
        whisper_model = get_whisper_model()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio_data, sr)
        
        try:
            result = whisper_model.transcribe(tmp_path, fp16=False)
            transcript = result['text'].strip()
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        print(f"✓ Transcript: '{transcript}'")
        
        if not transcript or len(transcript) < 3:
            transcript = "Unable to transcribe audio clearly"
            print("Warning: Transcription failed")
            return "neutral", 0.5, False, 0.0, transcript
        
        # 2. Extract audio features
        print("Step 2/4: Extracting audio features...")
        audio_features = extract_audio_features(audio_data, sr)
        print(f"✓ Features: pitch={audio_features.get('pitch_mean', 0):.1f}Hz, "
              f"energy={audio_features.get('energy_mean', 0):.3f}")
        
        # 3. Emotion Recognition
        print("Step 3/4: Analyzing emotion...")
        emotion_classifier = get_emotion_classifier()
        emotion_results = emotion_classifier(transcript)[0]
        
        top_emotion = max(emotion_results, key=lambda x: x['score'])
        emotion = top_emotion['label']
        emotion_confidence = top_emotion['score']
        
        emotion_mapping = {
            'joy': 'happy',
            'sadness': 'sad',
            'anger': 'angry',
            'fear': 'fear',
            'disgust': 'disgust',
            'surprise': 'surprise',
            'neutral': 'neutral'
        }
        emotion = emotion_mapping.get(emotion.lower(), emotion.lower())
        print(f"✓ Emotion: {emotion} (confidence: {emotion_confidence:.2f})")
        
        # 4. Sarcasm Detection
        print("Step 4/4: Detecting sarcasm...")
        sentiment_analyzer = get_sentiment_analyzer()
        sentiment_result = sentiment_analyzer(transcript)
        
        is_sarcastic, sarcasm_score, indicators = detect_sarcasm(
            transcript, audio_features, sentiment_result
        )
        
        print(f"✓ Sarcasm: {'YES' if is_sarcastic else 'NO'} (score: {sarcasm_score:.2f})")
        if indicators:
            print(f"  Indicators: {', '.join(indicators)}")
        
        print(f"{'='*60}\n")
        
        return emotion, round(emotion_confidence, 2), is_sarcastic, round(sarcasm_score, 2), transcript
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR in run_inference: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        return "neutral", 0.5, False, 0.0, "Error processing audio"

print("✓ Real-time inference engine initialized (models load on first use)")

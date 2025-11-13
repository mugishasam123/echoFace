import joblib
from . import utils

def load_artifacts():
    """Loads all trained models, scalers, and encoders."""
    print("Loading all model artifacts...")
    try:
        artifacts = {
            'face_model': joblib.load('models/image/face_recognition_model.pkl'),
            'face_scaler': joblib.load('models/image/face_recognition_scaler.pkl'),
            'face_encoder': joblib.load('models/image/face_recognition_encoder.pkl'),
            'voice_model': joblib.load('models/audio/voiceprint_model.pkl'),
            'voice_scaler': joblib.load('models/audio/voiceprint_scaler.pkl'),
            'voice_encoder': joblib.load('models/audio/voiceprint_encoder.pkl'),
        }
        print("✅ Artifacts loaded successfully.")
        return artifacts
    except FileNotFoundError as e:
        print(f"❌ Error: A required model file was not found. Please train all models first.")
        return None

def run_face_auth(image_path, model, scaler, encoder):
    """
    Performs facial recognition and returns the recognized name.
    Returns the name on success, None on failure.
    """
    print("\n[1/2] Analyzing facial image...")
    image_features = utils.extract_image_features(image_path)
    if image_features is None:
        print("❌ ACCESS DENIED: Could not process the provided image file.")
        return None

    image_features_scaled = scaler.transform(image_features)
    prediction_idx = model.predict(image_features_scaled)[0]
    predicted_name = encoder.inverse_transform([prediction_idx])[0]
    
    print(f"✅ Facial scan complete. User identified as: {predicted_name}")
    return predicted_name

def run_voice_auth(claimed_name, audio_path, model, scaler, encoder):
    """Performs the voice verification check."""
    print("\n[2/2] Analyzing voice sample...")
    audio_features = utils.extract_audio_features(audio_path)
    if audio_features is None: return False

    audio_features_scaled = scaler.transform(audio_features)
    predicted_name = encoder.inverse_transform(model.predict(audio_features_scaled))[0]

    print(f"Voice analysis complete. Detected: {predicted_name}")
    
    if predicted_name.lower() == claimed_name.lower():
        print("✅ Voiceprint MATCHES.")
        return True
    else:
        print("❌ ACCESS DENIED: Voiceprint does not match.")
        return False
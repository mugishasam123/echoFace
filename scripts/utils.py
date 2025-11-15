import librosa
import cv2
import numpy as np
import pandas as pd

def extract_image_features(image_path, bins=(8, 8, 8)):
    """Loads a single image and extracts its histogram features."""
    try:
        image = cv2.imread(image_path)
        if image is None: return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().reshape(1, -1)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def extract_audio_features(audio_path):
    """Loads a single audio file and extracts its features."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        
        features = np.hstack((mfccs_mean, rolloff_mean))
        mfcc_std = np.std(mfccs_mean)
        mfcc_range = np.max(mfccs_mean) - np.min(mfccs_mean)
        
        enhanced_features = np.hstack((features, mfcc_std, mfcc_range))
        return enhanced_features.reshape(1, -1)
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

USER_ID_MAP = {
    'Patrick': 150,
    'Samuel': 162,
    'David': 190,
    'Edine': 105,
    'Anonymous': 111,
}

def get_user_profile_data(user_name, tabular_data_path):
    """Fetches a user's most recent profile data using their name."""
    print(f"\nFetching profile for {user_name} to generate recommendation...")
    try:
        if user_name not in USER_ID_MAP:
            print(f"Error: User '{user_name}' not found in USER_ID_MAP.")
            return None

        customer_id = USER_ID_MAP[user_name]
        df = pd.read_csv(tabular_data_path)
        
        # Find all records for the customer and get the most recent one
        user_df = df[df['customer_id'] == customer_id].copy()
        if user_df.empty:
            print(f"No data found for customer ID {customer_id}.")
            return None
        
        # Sort by date to get the latest profile info
        user_df['purchase_date'] = pd.to_datetime(user_df['purchase_date'])
        latest_profile = user_df.sort_values(by='purchase_date', ascending=False).iloc[0]
        
        # Return the feature row (excluding identifiers and the target variable)
        user_profile = latest_profile.drop(['customer_id', 'transaction_id', 'product_category'])
        return user_profile

    except Exception as e:
        print(f"Could not retrieve profile for user {user_name}. Error: {e}")
        return None

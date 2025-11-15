# scripts/recommendation.py
import joblib
import pandas as pd

def load_product_artifacts():
    """Loads the product recommendation model and its preprocessors."""
    try:
        artifacts = {
            'model': joblib.load('models/product/product_recommendation_model.pkl'),
            'scaler': joblib.load('models/product/product_model_scaler.pkl'),
            'encoder': joblib.load('models/product/product_model_encoder.pkl'),
            'columns': joblib.load('models/product/product_model_columns.pkl'),
            'numeric_cols': joblib.load('models/product/product_model_numeric_cols.pkl') # New
        }
        return artifacts
    except FileNotFoundError as e:
        print(f"❌ Error: A required product model file was not found. ({e.filename})")
        return None

def run_product_recommendation(user_profile, artifacts):
    """Generates a product recommendation for the user."""
    print("Generating product recommendation...")
    
    user_df = pd.DataFrame(user_profile).T
    
    # --- Replicate Preprocessing from Notebook ---
    user_df['purchase_date'] = pd.to_datetime(user_df['purchase_date'])
    user_df['month'] = user_df['purchase_date'].dt.month
    user_df['day'] = user_df['purchase_date'].dt.day
    user_df['weekday'] = user_df['purchase_date'].dt.dayofweek
    user_df['isweekend'] = (user_df['weekday'] >= 5).astype(int)
    user_df = user_df.drop(columns=['purchase_date'])

    sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    user_df['review_sentiment'] = user_df['review_sentiment'].map(sentiment_mapping)

    user_df = pd.get_dummies(user_df)
    
    # Align columns with the training data
    training_columns = artifacts['columns']
    user_df = user_df.reindex(columns=training_columns, fill_value=0)

    # --- Scale and Predict ---
    # Use the exact list of numeric columns the scaler was trained on
    numeric_cols = artifacts['numeric_cols']
    user_df[numeric_cols] = artifacts['scaler'].transform(user_df[numeric_cols])
    
    prediction_idx = artifacts['model'].predict(user_df)[0]
    
    # Decode and return the prediction
    predicted_category = artifacts['encoder'].inverse_transform([prediction_idx])[0]
    return predicted_category
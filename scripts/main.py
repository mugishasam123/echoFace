# scripts/main.py
from . import auth, recommendation, utils
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def get_image_input():
    """Gets the image file path from the user."""
    image_path = input("Please enter the path to your facial image to begin: ")
    return image_path

def main():
    """Main function to orchestrate the full pipeline."""
    print("--- RekomAI: User Authentication & Recommendation System ---\n")

    auth_artifacts = auth.load_artifacts()
    prod_artifacts = recommendation.load_product_artifacts()
    if not auth_artifacts or not prod_artifacts:
        return

    # --- Step 1: Authentication Flow ---
    image_path = get_image_input()
    
    # Corrected: Unpack the dictionary to pass individual arguments
    recognized_name = auth.run_face_auth(
        image_path,
        auth_artifacts['face_model'],
        auth_artifacts['face_scaler'],
        auth_artifacts['face_encoder']
    )

    if recognized_name:
        print(f"\nWelcome, {recognized_name}. Please verify your identity with your voice.")
        audio_path = input("Enter the path to your voice sample: ")
        
        # Corrected: Unpack the dictionary here as well
        voice_auth_passed = auth.run_voice_auth(
            recognized_name,
            audio_path,
            auth_artifacts['voice_model'],
            auth_artifacts['voice_scaler'],
            auth_artifacts['voice_encoder']
        )
        
        if voice_auth_passed:
            print("\n======================================")
            print(f"🔐 AUTHENTICATION SUCCESSFUL for {recognized_name}!")
            print("======================================")

            tabular_data_path = 'data/customer-info/merged_dataset.csv'
            user_profile = utils.get_user_profile_data(recognized_name, tabular_data_path)
            
            if user_profile is not None:
                recommended_product = recommendation.run_product_recommendation(user_profile, prod_artifacts)
                print("\n--------------------------------------")
                print(f"✨ Recommended Product Category for you: {recommended_product}")
                print("--------------------------------------\n")
            else:
                print("\nCould not generate a recommendation for this user.")
        else:
            print("\n--- Authentication Failed at Voice Verification ---")
    else:
        print("\n--- Authentication Failed at Facial Recognition ---")

if __name__ == "__main__":
    main()
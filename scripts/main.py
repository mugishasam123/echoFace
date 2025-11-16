# scripts/main.py
from . import auth
from . import recommendation, utils
import warnings
import os
import time

warnings.filterwarnings("ignore", category=UserWarning)

def print_header():
    """Prints a nice header for the application."""
    print("\n" + "="*60)
    print(" "*15 + "EchoFace App")
    print(" "*10 + "User Authentication & Recommendation System")
    print("="*60 + "\n")

def print_menu():
    """Displays the main menu options."""
    print("\n" + "-"*60)
    print(" " * 20 + "MAIN MENU")
    print("-"*60)
    print("\n  [1] Use Image File Path")
    print("  [0] Exit")
    print("\n" + "-"*60)

def get_image_from_path():
    """Gets the image file path from the user."""
    print("\n" + "="*60)
    print(" " * 15 + "IMAGE FILE PATH MODE")
    print("="*60)
    print("\nPlease enter the full path to your facial image file.")
    print("   Example: data/images/patrick-neutral.jpg\n")
    
    image_path = input("Enter image path: ").strip()
    
    if not image_path:
        print("\nError: No path provided.")
        return None
    
    # Remove quotes if user added them
    image_path = image_path.strip('"').strip("'")
    
    if not os.path.exists(image_path):
        print(f"\nError: File not found at path: {image_path}")
        return None
    
    if not os.path.isfile(image_path):
        print(f"\nError: Path is not a file: {image_path}")
        return None
    
    # Check if it's a valid image file
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
        print(f"\nWarning: File extension may not be a valid image format.")
        print("   Continuing anyway...")
    
    print(f"\nImage file found: {os.path.basename(image_path)}")
    return image_path

def get_user_choice():
    """Gets the user's menu choice."""
    while True:
        try:
            choice = input("\nSelect an option (0-2): ").strip()
            if choice in ['0', '1']:
                return choice
            else:
                print("Invalid choice. Please enter 0 or 1.")
        except KeyboardInterrupt:
            print("\n\nExiting... Goodbye!")
            return '0'
        except Exception as e:
            print(f"Error: {e}")
            return '0'

def get_audio_from_path():
    """Gets the audio file path from the user."""
    print("\n" + "="*60)
    print(" " * 15 + "AUDIO FILE PATH MODE")
    print("="*60)
    print("\nPlease enter the full path to your audio file.")
    print("   Example: data/audio/patrick-approve.wav\n")
    
    audio_path = input("Enter audio path: ").strip()
    
    # Remove quotes if user added them
    audio_path = audio_path.strip('"').strip("'")
    
    if not audio_path:
        print("\nError: No path provided.")
        return None
    
    if not os.path.exists(audio_path):
        print(f"\nError: File not found at path: {audio_path}")
        return None
    
    if not os.path.isfile(audio_path):
        print(f"\nError: Path is not a file: {audio_path}")
        return None
    
    # Check if it's a valid audio file
    valid_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.mp4']
    if not any(audio_path.lower().endswith(ext) for ext in valid_extensions):
        print(f"\nWarning: File extension may not be a valid audio format.")
        print("   Continuing anyway...")
    
    print(f"\nAudio file found: {os.path.basename(audio_path)}")
    return audio_path

def main():
    """Main function to orchestrate the full pipeline."""
    print_header()

    # Load model artifacts
    print("Loading model artifacts...")
    auth_artifacts = auth.load_artifacts()
    prod_artifacts = recommendation.load_product_artifacts()
    
    if not auth_artifacts or not prod_artifacts:
        print("\nFailed to load required models. Exiting...")
        return

    # Main menu loop
    while True:
        print_menu()
        choice = get_user_choice()
        
        if choice == '0':
            print("\nThank you for using EchoFace! Goodbye!\n")
            break
        
        # Get image based on user's choice
        image_path = None
        if choice == '1':
            image_path = get_image_from_path()
        
        if image_path is None:
            print("\nImage input failed or was cancelled.")
            print("   Returning to main menu...\n")
            continue
        
        # --- Face Recognition ---
        print("\n" + "="*60)
        print(" " * 18 + "FACE RECOGNITION")
        print("="*60)
        
        recognized_name = auth.run_face_auth(
            image_path,
            auth_artifacts['face_model'],
            auth_artifacts['face_scaler'],
            auth_artifacts['face_encoder']
        )

        # Initialize variables for cleanup
        audio_path = None
        is_temp_audio = False
        
        if recognized_name:
            print(f"\nFace recognition successful! User identified as: {recognized_name}")
            print("\n" + "-"*60)
            
            # --- Voice Verification ---
            print(f"\nWelcome, {recognized_name}!")
            print("Please verify your identity with your voice sample.")
            
            audio_path = get_audio_from_path()
            
            if audio_path is None:
                print("\nVoice verification cancelled or failed.")
                print("   Authentication failed at voice verification.\n")
            else:
                voice_auth_passed = auth.run_voice_auth(
                    recognized_name,
                    audio_path,
                    auth_artifacts['voice_model'],
                    auth_artifacts['voice_scaler'],
                    auth_artifacts['voice_encoder']
                )
                
                if voice_auth_passed:
                    print("\n" + "="*60)
                    print(" " * 15 + "AUTHENTICATION SUCCESSFUL")
                    print("="*60)
                    print(f"\nUser {recognized_name} has been authenticated!")
                    print("\n" + "-"*60)

                    # --- Product Recommendation ---
                    print("\n" + "="*60)
                    print(" " * 15 + "PRODUCT RECOMMENDATION")
                    print("="*60)
                    
                    tabular_data_path = 'data/customer-info/merged_dataset.csv'
                    user_profile = utils.get_user_profile_data(recognized_name, tabular_data_path)

                    if user_profile is not None:
                        recommended_product = recommendation.run_product_recommendation(user_profile, prod_artifacts)
                        print("\n" + "="*60)
                        print(" " * 12 + "RECOMMENDATION RESULT")
                        print("="*60)
                        print(f"\nRecommended Product Category: {recommended_product}")
                        print("\n" + "="*60 + "\n")
                    else:
                        print("\nCould not generate a recommendation for this user.")
                        print("   User profile data not found.\n")
                else:
                    print("\n" + "="*60)
                    print(" " * 15 + "AUTHENTICATION FAILED")
                    print("="*60)
                    print("\nVoice verification failed.")
                    print("   Voiceprint does not match the claimed identity.\n")
        else:
            print("\n" + "="*60)
            print(" " * 15 + "AUTHENTICATION FAILED")
            print("="*60)
            print("\nFacial recognition could not identify the user.")
            print("   Please try again with a clearer image.\n")
        
        # Ask if user wants to continue
        print("-"*60)
        continue_choice = input("\nWould you like to try again? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("\nThank you for using EchoFace! Goodbye!\n")
            break
        print()

if __name__ == "__main__":
    main()
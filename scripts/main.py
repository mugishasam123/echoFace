# scripts/main.py
from . import auth
from . import recommendation, utils
import warnings
import cv2
import os
import tempfile

warnings.filterwarnings("ignore", category=UserWarning)

def print_header():
    """Prints a nice header for the application."""
    print("\n" + "="*60)
    print(" "*15 + "🎭 EchoFace App")
    print(" "*10 + "User Authentication & Recommendation System")
    print("="*60 + "\n")

def print_menu():
    """Displays the main menu options."""
    print("\n" + "-"*60)
    print(" " * 20 + "📋 MAIN MENU")
    print("-"*60)
    print("\n  [1] 📁 Use Image File Path")
    print("  [2] 📷 Capture from Webcam")
    print("  [0] ❌ Exit")
    print("\n" + "-"*60)

def get_image_from_path():
    """Gets the image file path from the user."""
    print("\n" + "="*60)
    print(" " * 15 + "📁 IMAGE FILE PATH MODE")
    print("="*60)
    print("\n💡 Please enter the full path to your facial image file.")
    print("   Example: data/images/patrick-neutral.jpg\n")
    
    image_path = input("👉 Enter image path: ").strip()
    
    if not image_path:
        print("\n❌ Error: No path provided.")
        return None
    
    # Remove quotes if user added them
    image_path = image_path.strip('"').strip("'")
    
    if not os.path.exists(image_path):
        print(f"\n❌ Error: File not found at path: {image_path}")
        return None
    
    if not os.path.isfile(image_path):
        print(f"\n❌ Error: Path is not a file: {image_path}")
        return None
    
    # Check if it's a valid image file
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
        print(f"\n⚠️  Warning: File extension may not be a valid image format.")
        print("   Continuing anyway...")
    
    print(f"\n✅ Image file found: {os.path.basename(image_path)}")
    return image_path

def get_image_from_webcam():
    """Captures an image from the webcam and returns the file path."""
    print("\n" + "="*60)
    print(" " * 15 + "📷 WEBCAM CAPTURE MODE")
    print("="*60)
    print("\n💡 Instructions:")
    print("   • Position yourself in front of the camera")
    print("   • Press SPACEBAR to capture")
    print("   • Press 'Q' to cancel")
    print("\n📷 Opening webcam...\n")
    
    # Open webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        print("   Please check if your camera is connected and permissions are granted.")
        return None
    
    captured = False
    image_path = None
    
    try:
        while not captured:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Could not read frame from webcam.")
                break
            
            # Add instructions on the frame
            cv2.putText(frame, "Press SPACE to capture, Q to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('EchoFace - Webcam Capture', frame)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space bar to capture
                # Create a temporary file to save the captured image
                temp_dir = tempfile.gettempdir()
                image_path = os.path.join(temp_dir, 'echoface_capture.jpg')
                cv2.imwrite(image_path, frame)
                
                if os.path.exists(image_path):
                    print("✅ Image captured successfully!")
                    captured = True
                else:
                    print("❌ Error: Could not save captured image.")
                    break
                    
            elif key == ord('q') or key == ord('Q'):  # 'q' to quit
                print("❌ Capture cancelled by user.")
                break
                
    except Exception as e:
        print(f"❌ Error during webcam capture: {e}")
    finally:
        # Release the webcam and close windows
        cap.release()
        cv2.destroyAllWindows()
    
    return image_path

def get_user_choice():
    """Gets the user's menu choice."""
    while True:
        try:
            choice = input("\n👉 Select an option (0-2): ").strip()
            if choice in ['0', '1', '2']:
                return choice
            else:
                print("❌ Invalid choice. Please enter 0, 1, or 2.")
        except KeyboardInterrupt:
            print("\n\n👋 Exiting... Goodbye!")
            return '0'
        except Exception as e:
            print(f"❌ Error: {e}")
            return '0'

def main():
    """Main function to orchestrate the full pipeline."""
    print_header()

    # Load model artifacts
    print("🔄 Loading model artifacts...")
    auth_artifacts = auth.load_artifacts()
    prod_artifacts = recommendation.load_product_artifacts()
    
    if not auth_artifacts or not prod_artifacts:
        print("\n❌ Failed to load required models. Exiting...")
        return

    # Main menu loop
    while True:
        print_menu()
        choice = get_user_choice()
        
        if choice == '0':
            print("\n👋 Thank you for using EchoFace! Goodbye!\n")
            break
        
        # Get image based on user's choice
        image_path = None
        if choice == '1':
            image_path = get_image_from_path()
        elif choice == '2':
            image_path = get_image_from_webcam()
        
        if image_path is None:
            print("\n⚠️  Image input failed or was cancelled.")
            print("   Returning to main menu...\n")
            continue
        
        # --- Face Recognition ---
        print("\n" + "="*60)
        print(" " * 18 + "🔍 FACE RECOGNITION")
        print("="*60)
        
        recognized_name = auth.run_face_auth(
            image_path,
            auth_artifacts['face_model'],
            auth_artifacts['face_scaler'],
            auth_artifacts['face_encoder']
        )

        if recognized_name:
            print(f"\n✅ Face recognition successful! User identified as: {recognized_name}")
            print("\n" + "-"*60)
            
            # --- Voice Verification ---
            print("\n" + "="*60)
            print(" " * 18 + "🎤 VOICE VERIFICATION")
            print("="*60)
            print(f"\n👤 Welcome, {recognized_name}!")
            print("💡 Please verify your identity with your voice sample.")
            print("\n💬 Enter the full path to your voice/audio file.")
            print("   Example: data/audio/patrick-approve.wav\n")
            
            audio_path = input("👉 Enter audio path: ").strip()
            
            # Remove quotes if user added them
            audio_path = audio_path.strip('"').strip("'")
            
            if not audio_path:
                print("\n❌ Error: No audio path provided.")
                print("   Authentication failed at voice verification.\n")
            elif not os.path.exists(audio_path):
                print(f"\n❌ Error: Audio file not found at path: {audio_path}")
                print("   Authentication failed at voice verification.\n")
            else:
                print(f"\n✅ Audio file found: {os.path.basename(audio_path)}")
                
                voice_auth_passed = auth.run_voice_auth(
                    recognized_name,
                    audio_path,
                    auth_artifacts['voice_model'],
                    auth_artifacts['voice_scaler'],
                    auth_artifacts['voice_encoder']
                )
                
                if voice_auth_passed:
                    print("\n" + "="*60)
                    print(" " * 15 + "✅ AUTHENTICATION SUCCESSFUL")
                    print("="*60)
                    print(f"\n🔐 User {recognized_name} has been authenticated!")
                    print("\n" + "-"*60)

                    # --- Product Recommendation ---
                    print("\n" + "="*60)
                    print(" " * 15 + "🎯 PRODUCT RECOMMENDATION")
                    print("="*60)
                    
                    tabular_data_path = 'data/customer-info/merged_dataset.csv'
                    user_profile = utils.get_user_profile_data(recognized_name, tabular_data_path)

                    if user_profile is not None:
                        recommended_product = recommendation.run_product_recommendation(user_profile, prod_artifacts)
                        print("\n" + "="*60)
                        print(" " * 12 + "✨ RECOMMENDATION RESULT")
                        print("="*60)
                        print(f"\n🎁 Recommended Product Category: {recommended_product}")
                        print("\n" + "="*60 + "\n")
                    else:
                        print("\n❌ Could not generate a recommendation for this user.")
                        print("   User profile data not found.\n")
                else:
                    print("\n" + "="*60)
                    print(" " * 15 + "❌ AUTHENTICATION FAILED")
                    print("="*60)
                    print("\n⚠️  Voice verification failed.")
                    print("   Voiceprint does not match the claimed identity.\n")
        else:
            print("\n" + "="*60)
            print(" " * 15 + "❌ AUTHENTICATION FAILED")
            print("="*60)
            print("\n⚠️  Facial recognition could not identify the user.")
            print("   Please try again with a clearer image.\n")
        
        # Clean up temporary captured image (only for webcam captures)
        if image_path and 'echoface_capture' in image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                pass  # Silently ignore cleanup errors
        
        # Ask if user wants to continue
        print("-"*60)
        continue_choice = input("\n🔄 Would you like to try again? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("\n👋 Thank you for using EchoFace! Goodbye!\n")
            break
        print()

if __name__ == "__main__":
    main()
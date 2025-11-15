# scripts/main.py
from . import auth
from . import recommendation, utils
import warnings
import cv2
import os
import tempfile
import time
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# Try to import sounddevice for audio recording
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_RECORDING_AVAILABLE = True
except ImportError:
    AUDIO_RECORDING_AVAILABLE = False

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

def get_audio_from_path():
    """Gets the audio file path from the user."""
    print("\n" + "="*60)
    print(" " * 15 + "📁 AUDIO FILE PATH MODE")
    print("="*60)
    print("\n💡 Please enter the full path to your audio file.")
    print("   Example: data/audio/patrick-approve.wav\n")
    
    audio_path = input("👉 Enter audio path: ").strip()
    
    # Remove quotes if user added them
    audio_path = audio_path.strip('"').strip("'")
    
    if not audio_path:
        print("\n❌ Error: No path provided.")
        return None
    
    if not os.path.exists(audio_path):
        print(f"\n❌ Error: File not found at path: {audio_path}")
        return None
    
    if not os.path.isfile(audio_path):
        print(f"\n❌ Error: Path is not a file: {audio_path}")
        return None
    
    # Check if it's a valid audio file
    valid_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.mp4']
    if not any(audio_path.lower().endswith(ext) for ext in valid_extensions):
        print(f"\n⚠️  Warning: File extension may not be a valid audio format.")
        print("   Continuing anyway...")
    
    print(f"\n✅ Audio file found: {os.path.basename(audio_path)}")
    return audio_path

def record_audio_from_microphone(duration=3, sample_rate=22050):
    """Records audio from the microphone and returns the file path."""
    if not AUDIO_RECORDING_AVAILABLE:
        print("\n❌ Error: Audio recording is not available.")
        print("   Please install sounddevice and soundfile:")
        print("   pip install sounddevice soundfile")
        return None
    
    print("\n" + "="*60)
    print(" " * 15 + "🎤 MICROPHONE RECORDING MODE")
    print("="*60)
    print(f"\n💡 Instructions:")
    print(f"   • Recording will start in 3 seconds")
    print(f"   • Speak clearly for {duration} seconds")
    print(f"   • Press Ctrl+C to cancel\n")
    
    try:
        # Countdown
        for i in range(3, 0, -1):
            print(f"   Recording starts in {i}...", end='\r')
            time.sleep(1)
        print("   🎤 Recording now! Speak clearly...        ")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        
        print("   ✅ Recording complete!")
        
        # Save to temporary file
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, 'echoface_voice_recording.wav')
        
        sf.write(audio_path, audio_data, sample_rate)
        
        if os.path.exists(audio_path):
            print(f"   💾 Audio saved: {os.path.basename(audio_path)}")
            return audio_path
        else:
            print("   ❌ Error: Could not save recorded audio.")
            return None
            
    except KeyboardInterrupt:
        print("\n   ❌ Recording cancelled by user.")
        return None
    except Exception as e:
        print(f"\n   ❌ Error during recording: {e}")
        return None

def get_audio_input_menu():
    """Shows menu for audio input options and returns the audio path."""
    print("\n" + "="*60)
    print(" " * 18 + "🎤 VOICE VERIFICATION")
    print("="*60)
    print("\n📋 How would you like to provide your voice sample?")
    print("\n  [1] 📁 Use Audio File Path")
    if AUDIO_RECORDING_AVAILABLE:
        print("  [2] 🎤 Record from Microphone")
    else:
        print("  [2] 🎤 Record from Microphone (⚠️  Not available - install sounddevice)")
    print("  [0] ❌ Cancel")
    print("\n" + "-"*60)
    
    while True:
        try:
            choice = input("\n👉 Select an option: ").strip()
            
            if choice == '0':
                print("\n❌ Voice verification cancelled.")
                return None
            elif choice == '1':
                return get_audio_from_path()
            elif choice == '2':
                if AUDIO_RECORDING_AVAILABLE:
                    duration_input = input("\n⏱️  Recording duration in seconds (default: 3): ").strip()
                    try:
                        duration = float(duration_input) if duration_input else 3.0
                        if duration < 1:
                            print("⚠️  Duration too short, using 1 second minimum.")
                            duration = 1.0
                        elif duration > 10:
                            print("⚠️  Duration too long, using 10 seconds maximum.")
                            duration = 10.0
                    except ValueError:
                        print("⚠️  Invalid input, using default 3 seconds.")
                        duration = 3.0
                    return record_audio_from_microphone(duration=duration)
                else:
                    print("\n❌ Audio recording not available.")
                    print("   Please install: pip install sounddevice soundfile")
                    print("   Or use option 1 to provide an audio file path.")
                    continue
            else:
                print("❌ Invalid choice. Please enter 0, 1, or 2.")
        except KeyboardInterrupt:
            print("\n\n❌ Voice verification cancelled.")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

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

        # Initialize variables for cleanup
        audio_path = None
        is_temp_audio = False
        
        if recognized_name:
            print(f"\n✅ Face recognition successful! User identified as: {recognized_name}")
            print("\n" + "-"*60)
            
            # --- Voice Verification ---
            print(f"\n👤 Welcome, {recognized_name}!")
            print("💡 Please verify your identity with your voice sample.")
            
            audio_path = get_audio_input_menu()
            
            if audio_path is None:
                print("\n❌ Voice verification cancelled or failed.")
                print("   Authentication failed at voice verification.\n")
            else:
                # Check if this is a temporary recorded file
                if 'echoface_voice_recording' in audio_path:
                    is_temp_audio = True
                
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
        
        # Clean up temporary files
        # Clean up temporary captured image (only for webcam captures)
        if image_path and 'echoface_capture' in image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                pass  # Silently ignore cleanup errors
        
        # Clean up temporary recorded audio (only for microphone recordings)
        if is_temp_audio and audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
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
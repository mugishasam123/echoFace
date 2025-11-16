# EchoFace – Authentication & Recommendation System

EchoFace authenticates users with face + voice and then serves a personalized product recommendation.

Important: This project uses file-based inputs only (no webcam or microphone recording).

## How it works
- Face Recognition: classify the person from a facial image file.
- Voice Verification: verify the claimed identity from an audio file.
- Product Recommendation: predict a product category for the authenticated user.

## Run
```bash
python -m scripts.main
```

You will be prompted to provide:
- An image file path (face)
- An audio file path (voice)

Supported extensions:
- Image: .jpg, .jpeg, .png, .bmp, .tiff, .tif
- Audio: .wav, .mp3, .m4a, .flac, .ogg, .aac, .mp4

## Notes
- Webcam capture and microphone recording have been removed for accuracy and consistency.
- Ensure the models are trained and artifacts exist under `models/` directories.
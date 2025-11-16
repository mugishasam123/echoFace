## EchoFace Technical Report

This document details the end-to-end approach for the three core models (Image Face Recognition, Audio Voice Verification, and Product Recommendation) and the scripting layer that orchestrates data flow, inference, and integration.

### 1) Image Face Recognition

- **Goal**: Identify or verify a customer from an image.
- **Inputs**: RGB image (e.g., `.jpg`) from `data/images/` or camera capture.
- **Artifacts**:
  - `models/image/face_recognition_model.pkl` — trained classifier (e.g., SVM/LogReg).
  - `models/image/face_recognition_encoder.pkl` — label encoder mapping class names to indices.
  - `models/image/face_recognition_scaler.pkl` — feature standardizer for embeddings.
- **Preprocessing**:
  - Face detection and alignment (center the face; normalize orientation where possible).
  - Convert to a consistent size and color space (e.g., 160x160, RGB).
  - Normalize pixel values.
- **Feature Extraction**:
  - Generate fixed-length face embeddings using the face encoder/feature extractor (e.g., FaceNet-like or similar embedding model).
  - Apply `face_recognition_scaler.pkl` to standardize embedding distribution.
- **Modeling**:
  - Train a classical classifier on the scaled embeddings (one vector per face) using labels from the encoder.
  - Save model, encoder, and scaler to ensure deterministic inference.
- **Inference**:
  - Detect+align face, compute embedding, scale, run through classifier.
  - Map predicted class index back to name using the label encoder.
- **Quality & Evaluation**:
  - Metrics: accuracy, macro F1, confusion matrix due to class imbalance.
  - Validation via hold-out set or cross-validation; augmentations for robustness (lighting, small rotations).

### 2) Audio Voice Verification (Voiceprint)

- **Goal**: Verify a customer's voice (1:1 speaker verification or 1:N identification).
- **Inputs**: Short utterances (`.m4a`, `.mp4`) from `data/audio/`.
- **Artifacts**:
  - `models/audio/voiceprint_model.pkl` — downstream classifier/verification backend.
  - `models/audio/voiceprint_encoder.pkl` — label encoder for speaker IDs (if identification).
  - `models/audio/voiceprint_scaler.pkl` — feature standardizer for acoustic embeddings/features.
  - `models/audio/voiceprint_metadata.json` — metadata (sample rate, feature params).
- **Preprocessing**:
  - Resample to target sample rate (e.g., 16 kHz), convert to mono.
  - Denoise/normalize loudness; voice activity detection (optional) to trim silence.
- **Feature Extraction**:
  - Compute MFCCs, spectral features, or use a pretrained speaker embedding model (x-vector/ECAPA-like).
  - Aggregate to fixed-size vector (mean/std pooling or segment embeddings + pooling).
  - Standardize with `voiceprint_scaler.pkl`.
- **Modeling**:
  - Identification: classifier trained on speaker embeddings; encoder maps user IDs to indices.
  - Verification: compute similarity (cosine/PLDA) against enrolled templates or use a binary model.
- **Inference**:
  - Extract embedding from incoming audio, standardize, infer identity or compute similarity score.
  - Apply threshold for accept/reject in verification workflows.
- **Quality & Evaluation**:
  - Metrics: EER (Equal Error Rate), ROC AUC, minDCF, and confusion matrix for identification.
  - Robustness: test across channels, background noise, varied phrases.

### 3) Product Recommendation

- **Goal**: Recommend products based on user profile and transaction history.
- **Inputs**:
  - `data/customer-info/customer_transactions - customer_transactions.csv`
  - `data/customer-info/customer_social_profiles - customer_social_profiles.csv`
  - Optional merged view: `data/customer-info/merged_dataset.csv`
- **Artifacts**:
  - `models/product/product_recommendation_model.pkl` — core recommender (e.g., matrix factorization, gradient boosting, or hybrid).
  - `models/product/product_model_encoder.pkl` — encodes categorical features (users/items).
  - `models/product/product_model_scaler.pkl` — normalizes numeric features.
  - `models/product/product_model_columns.pkl` — column ordering/feature schema.
  - `models/product/product_model_numeric_cols.pkl` — list of numeric columns used.
- **Feature Engineering**:
  - Categorical encoding for user/product features (ID encoders, one-hot/target enc).
  - Aggregate behavioral features: frequency, recency, monetary value; product popularity.
  - Social profile signals: interests/categories; topical embeddings (optional).
  - Train-test split by time when possible to prevent leakage.
- **Modeling**:
  - Start with simple baselines (popularity, user-most-frequent) → supervised ranker or MF.
  - Optimize ranking metrics (MAP, NDCG@K) and top-K hit rate.
  - Persist all preprocessing artifacts to reproduce inference.
- **Inference**:
  - Given a user ID (and context), build feature vector(s) for candidate items.
  - Score candidates with the model, sort by predicted relevance, return top-K.
- **Quality & Evaluation**:
  - Offline: Precision@K, Recall@K, MAP@K, NDCG@K, coverage and diversity.
  - Online readiness: A/B test plan and guardrail metrics.

### 4) Scripting Layer and Orchestration (`scripts/`)

Files:
- `scripts/main.py` — CLI or entrypoint orchestrating image/voice verification and recommendations.
- `scripts/auth.py` — authentication/authorization helpers for user/session handling.
- `scripts/recommendation.py` — functions to load recommender artifacts and serve top-K results.
- `scripts/utils.py` — common utilities: I/O, preprocessing shortcuts, logging helpers.

Typical Flow:
1. Initialization
   - Load model artifacts (pickled models, scalers, encoders) from `models/`.
   - Set deterministic seeds, configure logging, parse CLI args or environment.
2. Image Verification
   - If an image is provided, run the face pipeline: detect/align → embed → scale → classify.
   - Return predicted identity and confidence.
3. Voice Verification
   - If an audio sample is provided, run the voice pipeline: resample/denoise → embed → scale → verify or identify.
   - Return identity/confidence or accept/reject decision using a threshold.
4. Recommendation
   - On successful identification, call `scripts/recommendation.py` to generate top-K items for the user.
   - Use `product_model_columns.pkl` and related artifacts to construct the correct feature matrix before scoring.
5. Output
   - Aggregate results: recognized user, verification scores, recommended products.
   - Save to `data/outputs/` or print to console/return via API (depending on integration).

Key Design Choices:
- Strict separation of concerns: preprocessing, feature extraction, and modeling are decoupled via explicit artifacts (scaler/encoder/model).
- Reproducible inference: training-time transforms are serialized and reapplied consistently at inference.
- Graceful fallback: if one biometric fails (e.g., low face confidence), allow the other to proceed or prompt for re-capture.
- Extensibility: drop-in replacement of embedding backbones or classifiers without changing the scripting interface.

### 5) Data Management and Artifacts

- Data directories under `data/` organize raw inputs and generated outputs:
  - `data/images/` — raw face images for each subject.
  - `data/audio/` — voice samples for each subject.
  - `data/outputs/` — derived features and diagnostic exports (e.g., `image_features.csv`, `audio_features.csv`).
- Models persisted under `models/` are versioned by subfolder (image/audio/product) and include all necessary preprocessing artifacts alongside the model.
- Notebooks under `notebooks/` documents experiments, parameters, and results for traceability.

### 6) Operational Concerns

- Robustness & Monitoring
  - Log intermediate scores/confidences for drift monitoring.
  - Store per-user failure rates for image/audio capture to guide UX improvements.
- Security
  - Protect model artifacts and PII; avoid storing raw audio/images beyond necessity.
  - Consider liveness checks for voice and anti-spoofing for faces in production.
- Performance
  - Warm-load models on startup to reduce first-inference latency.
  - Batch feature extraction where possible; use vectorized operations.
- Testing
  - Unit tests for preprocessing functions and encoders.
  - Golden-tests for deterministic inference on fixed inputs.

### 7) How to Extend

- Replace or upgrade embedding models (face/voice) while preserving the scaler/encoder protocol.
- Add additional modalities (e.g., text/chat history) to the recommender as features.
- Expose the scripting logic as a REST API or Streamlit app for live demos.

### 8) Quick Start (High-Level)

1. Ensure Python dependencies from `requirements.txt` are installed.
2. Place input media in `data/images/` and `data/audio/` or provide paths via CLI.
3. Run the main script to perform recognition/verification and get recommendations:
   - Example (illustrative; confirm actual CLI in `scripts/main.py`):
     - `python scripts/main.py --image data/images/patrick-neutral.jpg --audio data/audio/patrick_confirm.m4a --user_id patrick --top_k 5`
4. Inspect outputs and logs; iterate on thresholds and candidate filtering as needed.




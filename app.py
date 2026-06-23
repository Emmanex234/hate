import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
import re
import string
import pickle
import logging
from threading import Thread
import time
import unicodedata
from datetime import datetime
import json
from functools import wraps
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Remove old model file if exists
# if os.path.exists('nigerian_hate_speech_model.pkl'):
#     os.remove('nigerian_hate_speech_model.pkl')

# Kaggle API imports
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("Kaggle API not available. Install with: pip install kaggle")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            logger.error(f"Error in {func.__name__}: {error}")
            
        end_time = time.time()
        
        # Log metrics
        logger.info(f"Function: {func.__name__} - Duration: {end_time - start_time:.2f}s - Success: {success}")
        
        if success:
            return result
        else:
            raise Exception(error)
    return wrapper

class OptimizedNigerianHateSpeechDetector:
    def __init__(self):
        # Environment configuration - OPTIMIZED
        self.environment = os.environ.get('ENVIRONMENT', 'render_free')
        self.config = self.load_optimized_config()
        
        # Initialize Kaggle API
        self.kaggle_api = None
        self.setup_kaggle_api()
        
        # OPTIMIZED: Single fast model instead of ensemble for speed
        self.model = LogisticRegression(
            random_state=42, 
            max_iter=500,  # Reduced iterations
            C=1.0,
            solver='liblinear'  # Faster solver
        )
        
        # OPTIMIZED: Smaller, faster vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.config['max_features'],
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),  # Reduced from (1,3) for speed
            max_df=0.9,  # Less restrictive
            min_df=3,    # Higher min_df for speed
            analyzer='word',
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'
        )
        
        self.is_trained = False
        self.training_in_progress = False
        self.feedback_buffer = []
        self.performance_metrics = {}
        
        # OPTIMIZED: Reduced dataset list for faster training
        self.dataset_info = [
            {
                'name': 'nigerian_multilingual',
                'kaggle_id': 'sharonibejih/nigerian-multilingual-hate-speech',
                'description': 'Nigerian Multilingual Hate Speech Dataset'
            }
        ]
        
        # Enhanced Nigerian-specific hate patterns for better detection
        self.nigerian_hate_patterns = [
            # Common Nigerian hate words/phrases
            r'\b(mumu|oloshi|werey|yeye|ashawo|bastard|fool|idiot|stupid|useless)\b',
            r'\b(thief|ole|barawo|419|fraudster|yahoo\s+boy|scammer)\b',
            r'\b(kill|die|death|destroy|attack|fight)\s+(you|them|him|her)',
            r'\b(bloody|fucking|fuck|shit|damn)\s+\w+',
            r'\b(hate|detest|despise)\s+(you|them|all)',
            r'\b(go\s+to\s+hell|go\s+die|drop\s+dead)',
            r'\b(tribalist|ethnic)\s+(hate|fight|kill)',
            r'\b(animal|dog|goat|monkey|pig)\s+(person|people|you)',
            r'\b(mad|crazy|mental|psycho)\s+(person|man|woman|you)',
            r'\b(worthless|good\s+for\s+nothing|failure|loser)\b',
            # Nigerian pidgin hate expressions
            r'\bwetin\s+(dey\s+)?wrong\s+with\s+you\b',
            r'\byou\s+no\s+get\s+sense\b',
            r'\bcraze\s+(person|man|woman)\b',
            r'\bgo\s+hug\s+transformer\b',
            r'\bdie\s+there\b'
        ]
        
        # Compile patterns for faster matching
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.nigerian_hate_patterns]
        
        # Enhanced hate word dictionary for instant detection
        self.hate_words_set = set([
            'mumu', 'oloshi', 'werey', 'yeye', 'ashawo', 'bastard', 'fool', 'idiot', 
            'stupid', 'useless', 'thief', 'ole', 'barawo', '419', 'fraudster', 
            'scammer', 'bloody', 'fucking', 'fuck', 'shit', 'damn', 'hate', 
            'kill', 'die', 'death', 'destroy', 'attack', 'fight', 'mad', 'crazy', 
            'mental', 'psycho', 'worthless', 'failure', 'loser', 'animal', 'dog', 
            'goat', 'monkey', 'pig'
        ])
        
    def load_optimized_config(self):
        """Load optimized configuration for faster training"""
        return {
            'max_features': 3000,  # Reduced from 5000+ for speed
            'max_samples': 15000,   # Limit dataset size for speed
            'use_ensemble': False,  # Single model for speed
        }
    
    def setup_kaggle_api(self):
        """Setup Kaggle API with authentication"""
        if not KAGGLE_AVAILABLE:
            logger.error("Kaggle API not available")
            return False
            
        try:
            self.kaggle_api = KaggleApi()
            
            # Check for API credentials
            if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
                # Try to create from environment variables
                kaggle_username = os.environ.get('KAGGLE_USERNAME')
                kaggle_key = os.environ.get('KAGGLE_KEY')
                
                if kaggle_username and kaggle_key:
                    # Create .kaggle directory
                    kaggle_dir = os.path.expanduser('~/.kaggle')
                    os.makedirs(kaggle_dir, exist_ok=True)
                    
                    # Create kaggle.json
                    kaggle_config = {
                        "username": kaggle_username,
                        "key": kaggle_key
                    }
                    
                    with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as f:
                        json.dump(kaggle_config, f)
                    
                    # Set correct permissions
                    os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)
                    logger.info("Created Kaggle API credentials from environment variables")
                else:
                    logger.error("No Kaggle API credentials found")
                    return False
            
            # Authenticate
            self.kaggle_api.authenticate()
            logger.info("Kaggle API authenticated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up Kaggle API: {e}")
            return False
    
    @monitor_performance
    def fast_preprocess_text(self, text):
        """Optimized preprocessing for speed"""
        if pd.isna(text) or not text:
            return ""
        
        try:
            text = str(text).lower().strip()
            
            # Quick hate word detection
            words = text.split()
            hate_word_count = sum(1 for word in words if word in self.hate_words_set)
            
            # Quick pattern matching
            pattern_matches = sum(1 for pattern in self.compiled_patterns[:5] if pattern.search(text))  # Check only first 5 patterns for speed
            
            # Basic cleaning
            text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
            text = re.sub(r'@\w+|#\w+', ' ', text)
            text = re.sub(r'(.)\1{2,}', r'\1', text)  # Remove repeated chars
            text = re.sub(r'[^\w\s]', ' ', text)
            text = ' '.join(text.split())
            
            # Store hate indicators for quick prediction
            self._hate_indicators = hate_word_count + pattern_matches
            
            return text
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return ""
    
    @monitor_performance
    def download_dataset(self, dataset_info):
        """Optimized dataset download"""
        if not self.kaggle_api:
            logger.error("Kaggle API not initialized")
            return None
            
        try:
            dataset_name = dataset_info['kaggle_id']
            download_path = f'./{dataset_info["name"]}_dataset'
            
            logger.info(f"Downloading dataset: {dataset_name}")
            
            os.makedirs(download_path, exist_ok=True)
            
            self.kaggle_api.dataset_download_files(
                dataset_name, 
                path=download_path, 
                unzip=True
            )
            
            # Find CSV files
            csv_files = []
            for root, dirs, files in os.walk(download_path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            logger.info(f"Found CSV files: {csv_files}")
            return csv_files
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return None
    
    @monitor_performance
    def load_and_process_dataset(self):
        """Optimized dataset loading and processing"""
        try:
            # Download dataset
            csv_files = self.download_dataset(self.dataset_info[0])
            
            if not csv_files:
                logger.error("Failed to download dataset")
                return None
            
            # Load first CSV file
            csv_file = csv_files[0]
            
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    logger.info(f"Loaded {csv_file} with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.error(f"Could not read {csv_file}")
                return None
            
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            # Find text and label columns
            text_col = None
            label_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['text', 'tweet', 'content', 'sentence', 'message']):
                    text_col = col
                elif any(keyword in col_lower for keyword in ['label', 'class', 'hate', 'target']):
                    label_col = col
            
            if not text_col:
                text_col = df.columns[0]
            if not label_col:
                label_col = df.columns[-1]
                
            logger.info(f"Using text column: {text_col}, label column: {label_col}")
            
            # Create normalized dataframe
            processed_df = pd.DataFrame()
            processed_df['text'] = df[text_col].astype(str)
            processed_df['original_label'] = df[label_col]
            
            # Normalize labels to binary
            processed_df['label'] = self.normalize_labels(df[label_col])
            
            # Remove invalid data
            processed_df = processed_df.dropna()
            processed_df = processed_df[processed_df['text'].str.len() > 5]
            
            # OPTIMIZATION: Limit dataset size for faster training
            if len(processed_df) > self.config['max_samples']:
                # Keep balanced sample
                hate_samples = processed_df[processed_df['label'] == 1]
                non_hate_samples = processed_df[processed_df['label'] == 0]
                
                max_per_class = self.config['max_samples'] // 2
                
                if len(hate_samples) > max_per_class:
                    hate_samples = hate_samples.sample(max_per_class, random_state=42)
                if len(non_hate_samples) > max_per_class:
                    non_hate_samples = non_hate_samples.sample(max_per_class, random_state=42)
                
                processed_df = pd.concat([hate_samples, non_hate_samples], ignore_index=True)
                processed_df = processed_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            logger.info(f"Final dataset shape: {processed_df.shape}")
            logger.info(f"Label distribution: {processed_df['label'].value_counts().to_dict()}")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            return None
    
    def normalize_labels(self, labels):
        """Fast label normalization"""
        try:
            if labels.dtype == 'object':
                # String labels
                hate_indicators = ['hate', 'hateful', 'offensive', 'abusive', 'toxic', '1', 'yes', 'true']
                
                def is_hate(label):
                    if pd.isna(label):
                        return 0
                    label_str = str(label).lower().strip()
                    return int(any(indicator in label_str for indicator in hate_indicators))
                
                return labels.apply(is_hate)
            else:
                # Numeric labels
                unique_vals = sorted(labels.unique())
                if len(unique_vals) == 2:
                    min_val, max_val = min(unique_vals), max(unique_vals)
                    return labels.map({min_val: 0, max_val: 1})
                else:
                    return (labels != 0).astype(int)
                    
        except Exception as e:
            logger.error(f"Error normalizing labels: {e}")
            return pd.Series([0] * len(labels))
    
    @monitor_performance
    def train_optimized_model(self):
        """Fast training with optimizations"""
        try:
            self.training_in_progress = True
            logger.info("Starting optimized Nigerian hate speech model training...")
            
            # Load and process dataset
            df = self.load_and_process_dataset()
            
            if df is None or df.empty:
                raise ValueError("Could not load dataset")
            
            # Fast preprocessing
            logger.info("Preprocessing texts...")
            texts = []
            labels = []
            
            for idx, row in df.iterrows():
                processed_text = self.fast_preprocess_text(row['text'])
                if processed_text:  # Only keep non-empty texts
                    texts.append(processed_text)
                    labels.append(row['label'])
            
            logger.info(f"Processed {len(texts)} valid texts")
            
            if len(texts) < 100:
                raise ValueError("Not enough valid texts for training")
            
            # Fit vectorizer and transform texts
            logger.info("Vectorizing texts...")
            X = self.vectorizer.fit_transform(texts)
            y = np.array(labels)
            
            logger.info(f"Feature matrix shape: {X.shape}")
            logger.info(f"Label distribution: {np.bincount(y)}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            logger.info("Training model...")
            self.model.fit(X_train, y_train)
            
            # Evaluate
            logger.info("Evaluating model...")
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            self.performance_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'training_samples': len(texts),
                'test_samples': len(y_test),
                'vocabulary_size': X.shape[1]
            }
            
            if len(np.unique(y_test)) == 2:
                self.performance_metrics['auc_roc'] = roc_auc_score(y_test, y_prob)
            
            logger.info(f"Training completed!")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")
            
            self.is_trained = True
            self.save_model()
            
            return self.performance_metrics
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return {"error": str(e)}
        finally:
            self.training_in_progress = False
    
    def save_model(self):
        """Save the trained model"""
        try:
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'performance_metrics': self.performance_metrics,
                'config': self.config,
                'hate_patterns': self.nigerian_hate_patterns,
                'hate_words_set': self.hate_words_set,
                'training_timestamp': datetime.now().isoformat(),
                'model_version': 'optimized_nigerian_hate_speech_v1'
            }
            
            with open('nigerian_hate_speech_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
    
    def load_model(self):
        """Load a previously saved model"""
        try:
            with open('nigerian_hate_speech_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.performance_metrics = model_data.get('performance_metrics', {})
            
            self.is_trained = True
            logger.info("Model loaded successfully")
            return True
            
        except FileNotFoundError:
            logger.info("No saved model found")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, text):
        """Fast prediction with enhanced Nigerian hate detection"""
        if not self.is_trained:
            return {"error": "Model not trained yet"}
        
        if self.training_in_progress:
            return {"error": "Model training in progress, please wait"}
        
        try:
            # Preprocess text
            processed_text = self.fast_preprocess_text(text)
            
            if not processed_text:
                return {
                    "text": text,
                    "prediction": "Not Hate Speech",
                    "confidence": 0.6,
                    "hate_probability": 0.1,
                    "reason": "Empty text after preprocessing"
                }
            
            # Quick hate detection using patterns
            hate_indicators = getattr(self, '_hate_indicators', 0)
            
            # Vectorize and predict
            text_vector = self.vectorizer.transform([processed_text])
            prediction = self.model.predict(text_vector)[0]
            probabilities = self.model.predict_proba(text_vector)[0]
            
            # Enhanced prediction with pattern matching
            hate_prob = probabilities[1] if len(probabilities) > 1 else 0.0
            
            # Boost confidence for clear hate patterns
            if hate_indicators > 2:
                hate_prob = min(hate_prob + 0.3, 0.95)
                prediction = 1
            elif hate_indicators > 0:
                hate_prob = min(hate_prob + 0.1, 0.9)
            
            # Final prediction
            final_prediction = 1 if hate_prob > 0.5 else 0
            confidence = max(probabilities)
            
            result = {
                "text": text,
                "prediction": "Hate Speech" if final_prediction == 1 else "Not Hate Speech",
                "confidence": float(confidence),
                "hate_probability": float(hate_prob),
                "pattern_matches": int(hate_indicators),
                "processed_text": processed_text,
                "model_version": "optimized_nigerian_v1"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"error": f"Prediction error: {str(e)}"}
    
    def predict_batch(self, texts):
        """Batch prediction"""
        if not self.is_trained:
            return {"error": "Model not trained yet"}
        
        try:
            results = []
            for text in texts:
                result = self.predict(text)
                results.append(result)
            
            return {"predictions": results}
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return {"error": f"Batch prediction error: {str(e)}"}
    
    def get_performance_metrics(self):
        """Get model performance metrics"""
        return {
            "performance_metrics": self.performance_metrics,
            "model_status": {
                "trained": self.is_trained,
                "training_in_progress": self.training_in_progress,
                "model_version": "optimized_nigerian_v1",
                "specialized_for": "Nigerian hate speech detection"
            },
            "configuration": self.config
        }
    
    def test_nigerian_examples(self):
        """Test with Nigerian hate speech examples"""
        test_cases = [
            # Hate speech examples
            ("You are a useless mumu person", True),
            ("This yeye government is nonsense", False),  # Criticism, not hate
            ("All yoruba people are thieves", True),
            ("I will kill you bastard", True),
            ("You oloshi go die there", True),
            ("Werey person like you", True),
            ("This ashawo no get sense", True),
            ("You be bloody fool", True),
            ("Go hug transformer", True),
            ("Mad person like you", True),
            
            # Non-hate examples
            ("I love Nigeria and its people", False),
            ("The weather is beautiful today", False),
            ("Going to market to buy food", False),
            ("Education is very important", False),
            ("Let's work together for peace", False),
            ("Nigerian music is amazing", False),
            ("Lagos traffic is crazy", False),  # Common expression, not hate
            ("This government need to do better", False),  # Criticism
            ("I dey go church today", False),
            ("Make we chop rice", False)
        ]
        
        results = []
        correct_predictions = 0
        
        for text, expected_hate in test_cases:
            prediction_result = self.predict(text)
            is_hate_predicted = prediction_result.get("prediction") == "Hate Speech"
            is_correct = is_hate_predicted == expected_hate
            
            if is_correct:
                correct_predictions += 1
            
            results.append({
                "text": text,
                "expected": "Hate" if expected_hate else "Not Hate",
                "predicted": prediction_result.get("prediction", "Error"),
                "confidence": prediction_result.get("confidence", 0),
                "hate_probability": prediction_result.get("hate_probability", 0),
                "correct": is_correct
            })
        
        accuracy = correct_predictions / len(test_cases)
        
        return {
            "test_results": results,
            "accuracy": accuracy,
            "correct_predictions": f"{correct_predictions}/{len(test_cases)}"
        }

# Initialize the detector
detector = OptimizedNigerianHateSpeechDetector()

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for model predictions - flexible format"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        text = data.get('text', '').strip()
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        
        # Handle both old and new formats
        predicted_label = data.get('predicted_label') or data.get('prediction')
        actual_label = data.get('actual_label')
        
        if not predicted_label:
            return jsonify({"error": "Missing predicted label"}), 400
        
        if actual_label is None:
            return jsonify({"error": "Missing actual label"}), 400
        
        # Convert numeric labels to string format
        def convert_label(label):
            if isinstance(label, (int, float)):
                return "Hate Speech" if label == 1 else "Not Hate Speech"
            return str(label)
        
        predicted_label = convert_label(predicted_label)
        actual_label = convert_label(actual_label)
        
        user_comment = data.get('comment', '')
        confidence = data.get('confidence', 0)
        
        if len(text) > 1000:
            return jsonify({"error": "Text too long (max 1000 characters)"}), 400
        
        # Create feedback entry
        feedback_entry = {
            'text': text,
            'predicted_label': predicted_label,
            'actual_label': actual_label,
            'user_comment': user_comment,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'model_version': 'optimized_nigerian_v1',
            'is_correct': predicted_label == actual_label
        }
        
        # Store feedback in buffer
        detector.feedback_buffer.append(feedback_entry)
        
        # Log feedback for analysis
        logger.info(f"Feedback received - Text: {text[:50]}..., Predicted: {predicted_label}, Actual: {actual_label}")
        
        # Keep only last 1000 feedback entries to manage memory
        if len(detector.feedback_buffer) > 1000:
            detector.feedback_buffer = detector.feedback_buffer[-1000:]
        
        return jsonify({
            "status": "success",
            "message": "Feedback submitted successfully",
            "feedback_id": len(detector.feedback_buffer),
            "total_feedback": len(detector.feedback_buffer)
        })
        
    except Exception as e:
        logger.error(f"Feedback submission error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
    
# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_trained": detector.is_trained,
        "training_in_progress": detector.training_in_progress,
        "model_version": "optimized_nigerian_v1",
        "specialized_for": "Fast Nigerian hate speech detection",
        "kaggle_api_available": KAGGLE_AVAILABLE
    })



@app.route('/train', methods=['POST'])
def train_model():
    if detector.training_in_progress:
        return jsonify({
            "status": "error",
            "message": "Training already in progress"
        })
    
    if not KAGGLE_AVAILABLE:
        return jsonify({
            "status": "error",
            "message": "Kaggle API not available"
        })
    
    def train_async():
        try:
            logger.info("Starting async training...")
            metrics = detector.train_optimized_model()
            
            if 'error' not in metrics:
                logger.info("Training completed successfully")
            else:
                logger.error(f"Training failed: {metrics['error']}")
                
        except Exception as e:
            logger.error(f"Async training error: {str(e)}")
    
    # Start training in background
    thread = Thread(target=train_async)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "status": "success",
        "message": "Optimized Nigerian hate speech model training started",
        "estimated_time": "2-5 minutes"
    })

@app.route('/predict', methods=['POST'])
def predict_hate_speech():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        
        if len(text) > 1000:  # Reduced limit for faster processing
            return jsonify({"error": "Text too long (max 1000 characters)"}), 400
        
        result = detector.predict(text)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({"error": "No texts provided"}), 400
        
        if len(texts) > 20:  # Reduced for faster processing
            return jsonify({"error": "Too many texts (max 20)"}), 400
        
        result = detector.predict_batch(texts)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/model/metrics')
def get_model_metrics():
    try:
        metrics = detector.get_performance_metrics()
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/status')
def model_status():
    return jsonify({
        "trained": detector.is_trained,
        "training_in_progress": detector.training_in_progress,
        "status": "ready" if detector.is_trained else ("training" if detector.training_in_progress else "not trained"),
        "model_version": "optimized_nigerian_v1",
        "specialized_for": "Fast Nigerian hate speech detection",
        "performance_metrics": detector.performance_metrics,
        "kaggle_api_available": KAGGLE_AVAILABLE
    })

@app.route('/test/examples')
def test_examples():
    try:
        if not detector.is_trained:
            return jsonify({"error": "Model not trained yet"}), 400
        
        results = detector.test_nigerian_examples()
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.route('/feedback/all', methods=['GET'])
def get_all_feedback():
    """Return all collected feedback entries"""
    try:
        if not detector.feedback_buffer:
            return jsonify({"message": "No feedback available yet"}), 200

        return jsonify({
            "total_feedback": len(detector.feedback_buffer),
            "feedback_entries": detector.feedback_buffer
        }), 200
    except Exception as e:
        logger.error(f"Error fetching feedback: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


# ==============================
# 🔹 RETRAIN MODEL ROUTE (Old + New)
# ==============================
@app.route('/retrain', methods=['POST'])
def retrain_with_feedback():
    """Retrain the model using both original dataset + user feedback"""
    try:
        # Path to your original training data file
        original_dataset_path = 'original_dataset.csv'  # ⚠️ update if filename different

        # Check if any feedback is available
        if not detector.feedback_buffer:
            return jsonify({"error": "No feedback available for retraining"}), 400

        logger.info("Starting retraining with original data + feedback data...")

        # 1️⃣ Load original dataset
        if not os.path.exists(original_dataset_path):
            return jsonify({"error": "Original dataset not found"}), 400

        df_old = pd.read_csv(original_dataset_path)

        # Ensure consistent column names
        df_old.rename(columns={df_old.columns[0]: 'text', df_old.columns[1]: 'label'}, inplace=True)

        # Convert labels to binary (1 = hate speech, 0 = not hate speech)
        df_old['label'] = df_old['label'].apply(lambda x: 1 if str(x).lower() == 'hate speech' else 0)

        # 2️⃣ Load feedback data
        df_feedback = pd.DataFrame(detector.feedback_buffer)
        df_feedback['label'] = df_feedback['actual_label'].apply(
            lambda x: 1 if str(x).lower() == 'hate speech' else 0
        )

        # 3️⃣ Combine old + new data
        df_combined = pd.concat([df_old, df_feedback], ignore_index=True)

        # 4️⃣ Preprocess text
        texts = [detector.fast_preprocess_text(t) for t in df_combined['text']]
        labels = df_combined['label'].values

        # 5️⃣ Rebuild vectorizer with combined data
        detector.vectorizer = TfidfVectorizer(max_features=5000)
        X = detector.vectorizer.fit_transform(texts)

        # 6️⃣ Retrain model
        detector.model.fit(X, labels)
        detector.save_model()

        logger.info("✅ Retraining complete — model updated with feedback + old data")

        # 7️⃣ Clear feedback after retraining
        detector.feedback_buffer = []
        with open('feedback_data.json', 'w') as f:
            json.dump(detector.feedback_buffer, f, indent=4)

        return jsonify({
            "status": "success",
            "message": "Model retrained successfully using old data + new feedback",
            "total_samples": len(df_combined)
        }), 200

    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Auto-train model on startup
def auto_train_on_startup():
    """Auto-train model on startup if conditions are met"""
    time.sleep(10)  # Wait for app to start
    if not detector.is_trained and not detector.training_in_progress and KAGGLE_AVAILABLE:
        try:
            logger.info("Auto-training optimized model on startup...")
            detector.train_optimized_model()
        except Exception as e:
            logger.error(f"Auto-training failed: {e}")

if __name__ == '__main__':
    # Try to load existing model first
    if not detector.load_model():
        logger.info("No existing model found.")
        
        # Start auto-training in background for production
        if (os.environ.get('RENDER') or os.environ.get('ENVIRONMENT') == 'production') and KAGGLE_AVAILABLE:
            thread = Thread(target=auto_train_on_startup)
            thread.daemon = True
            thread.start()
        elif not KAGGLE_AVAILABLE:
            logger.warning("Kaggle API not available. Cannot download datasets.")
    
    # Get port from environment variable
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )
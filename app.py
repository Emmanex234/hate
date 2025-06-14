import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
import re
import string
import pickle
import kagglehub
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
        logger.info(f"Function: {func.__name__}")
        logger.info(f"Duration: {end_time - start_time:.4f}s")
        logger.info(f"Success: {success}")
        
        if success:
            return result
        else:
            raise Exception(error)
    return wrapper

class AdvancedHateSpeechDetector:
    def __init__(self):
        # Environment configuration
        self.environment = os.environ.get('ENVIRONMENT', 'render_free')
        self.config = self.load_config()
        
        # Multiple models for ensemble
        self.models = {
            'tfidf_lr': LogisticRegression(random_state=42, max_iter=1000),
            'tfidf_svm': SVC(probability=True, random_state=42),
        }
        
        # Vectorizers
        self.vectorizers = {
            'tfidf': TfidfVectorizer(
                max_features=self.config['max_features'],
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 3),
                max_df=0.95,
                min_df=2
            )
        }
        
        # Ensemble weights
        self.ensemble_weights = [0.6, 0.4]  # LR gets more weight
        
        self.is_trained = False
        self.training_in_progress = False
        self.feedback_buffer = []
        self.retrain_threshold = 100
        self.performance_metrics = {}
        
        # Multiple datasets configuration
        self.datasets = [
            {
                'name': 'nigerian_multilingual',
                'kaggle_id': 'sharonibejih/nigerian-multilingual-hate-speech',
                'primary': True
            },
            {
                'name': 'hate_speech_detection',
                'kaggle_id': 'mrmorj/hate-speech-and-offensive-language-detection',
                'primary': False
            },
            {
                'name': 'twitter_hate_speech',
                'kaggle_id': 'arkhoshghalb/twitter-sentiment-analysis-hatred-speech',
                'primary': False
            },
            {
                'name': 'cyberbullying_detection',
                'kaggle_id': 'andrewmvd/cyberbullying-classification',
                'primary': False
            }
        ]
        
    def load_config(self):
        """Load environment-specific configuration"""
        if self.environment == 'production':
            return {
                'max_features': 10000,
                'use_ensemble': True,
                'retrain_frequency': 'weekly'
            }
        elif self.environment == 'render_free':
            return {
                'max_features': 3000,
                'use_ensemble': True,
                'retrain_frequency': 'manual'
            }
        else:  # development
            return {
                'max_features': 5000,
                'use_ensemble': True,
                'retrain_frequency': 'daily'
            }
    
    @monitor_performance
    def advanced_preprocess_text(self, text):
        """Enhanced preprocessing with context preservation"""
        if pd.isna(text):
            return {"text": "", "features": {}}
        
        try:
            # Store original for analysis
            original_text = str(text)
            
            # Normalize unicode
            text = unicodedata.normalize('NFKD', text)
            
            # Extract features before cleaning
            features = {
                'original_length': len(original_text),
                'exclamation_count': text.count('!'),
                'question_count': text.count('?'),
                'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
                'has_urls': bool(re.search(r'http|www', text)),
                'has_mentions': bool(re.search(r'@\w+', text)),
                'has_hashtags': bool(re.search(r'#\w+', text)),
                'repeated_chars': len(re.findall(r'(.)\1{2,}', text)),
                'word_count': len(text.split()),
            }
            
            # Sentiment analysis
            try:
                blob = TextBlob(text)
                features['sentiment_polarity'] = blob.sentiment.polarity
                features['sentiment_subjectivity'] = blob.sentiment.subjectivity
            except:
                features['sentiment_polarity'] = 0
                features['sentiment_subjectivity'] = 0
            
            # Advanced cleaning
            text = str(text).lower()
            
            # Remove URLs but preserve the fact they existed
            text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text, flags=re.MULTILINE)
            
            # Replace mentions and hashtags with placeholders
            text = re.sub(r'@\w+', ' MENTION ', text)
            text = re.sub(r'#\w+', ' HASHTAG ', text)
            
            # Handle repeated characters (hateeeee -> hate)
            text = re.sub(r'(.)\1{2,}', r'\1\1', text)
            
            # Remove punctuation but keep sentence structure
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return {"text": text, "features": features}
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return {"text": "", "features": {}}
    
    @monitor_performance
    def load_multiple_datasets(self):
        """Load and combine multiple hate speech datasets"""
        combined_data = []
        successful_datasets = []
        
        for dataset_info in self.datasets:
            try:
                logger.info(f"Attempting to load dataset: {dataset_info['name']}")
                
                # Download dataset
                path = kagglehub.dataset_download(dataset_info['kaggle_id'])
                logger.info(f"Dataset {dataset_info['name']} downloaded to: {path}")
                
                # Find CSV files
                csv_files = []
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.csv'):
                            csv_files.append(os.path.join(root, file))
                
                if not csv_files:
                    logger.warning(f"No CSV files found for {dataset_info['name']}")
                    continue
                
                # Load and process each CSV
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file, encoding='utf-8')
                        if df.empty:
                            continue
                            
                        logger.info(f"Loaded {dataset_info['name']} with shape: {df.shape}")
                        
                        # Normalize dataset format
                        normalized_df = self.normalize_dataset_format(df, dataset_info['name'])
                        
                        if normalized_df is not None and not normalized_df.empty:
                            # Sample data for memory efficiency
                            max_samples = 5000 if dataset_info['primary'] else 2000
                            if len(normalized_df) > max_samples:
                                normalized_df = normalized_df.sample(n=max_samples, random_state=42)
                            
                            normalized_df['dataset_source'] = dataset_info['name']
                            combined_data.append(normalized_df)
                            successful_datasets.append(dataset_info['name'])
                            logger.info(f"Successfully processed {dataset_info['name']}: {len(normalized_df)} samples")
                            
                    except Exception as e:
                        logger.error(f"Error processing CSV {csv_file}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error loading dataset {dataset_info['name']}: {e}")
                continue
        
        if combined_data:
            final_df = pd.concat(combined_data, ignore_index=True)
            logger.info(f"Combined dataset shape: {final_df.shape}")
            logger.info(f"Successfully loaded datasets: {successful_datasets}")
            logger.info(f"Label distribution: {final_df['label'].value_counts().to_dict()}")
            return final_df
        else:
            logger.error("No datasets could be loaded successfully")
            return None
    
    def normalize_dataset_format(self, df, dataset_name):
        """Normalize different dataset formats to standard format"""
        try:
            # Define column mappings for different datasets
            text_columns = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['text', 'tweet', 'comment', 'content', 'message'])]
            
            label_columns = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['label', 'hate', 'class', 'target', 'category'])]
            
            if not text_columns or not label_columns:
                # Try to infer from column positions
                if len(df.columns) >= 2:
                    text_col = df.columns[0]
                    label_col = df.columns[-1]  # Often the last column
                else:
                    return None
            else:
                text_col = text_columns[0]
                label_col = label_columns[0]
            
            # Create normalized dataframe
            normalized_df = pd.DataFrame()
            normalized_df['text'] = df[text_col]
            normalized_df['original_label'] = df[label_col]
            
            # Normalize labels to binary (0: not hate, 1: hate)
            normalized_df['label'] = self.normalize_labels(df[label_col], dataset_name)
            
            # Remove rows with missing values
            normalized_df = normalized_df.dropna()
            
            # Remove empty texts
            normalized_df = normalized_df[normalized_df['text'].str.len() > 0]
            
            # Ensure balanced dataset (optional)
            if len(normalized_df['label'].unique()) == 2:
                # Balance the dataset if too imbalanced
                label_counts = normalized_df['label'].value_counts()
                min_count = min(label_counts.values)
                max_count = max(label_counts.values)
                
                if max_count > min_count * 3:  # If more than 3:1 ratio
                    balanced_df = normalized_df.groupby('label').apply(
                        lambda x: x.sample(min(len(x), min_count * 2), random_state=42)
                    ).reset_index(drop=True)
                    return balanced_df
            
            return normalized_df
            
        except Exception as e:
            logger.error(f"Error normalizing dataset {dataset_name}: {e}")
            return None
    
    def normalize_labels(self, labels, dataset_name):
        """Convert various label formats to binary"""
        try:
            if labels.dtype == 'object':
                unique_labels = labels.unique()
                logger.info(f"Dataset {dataset_name} unique labels: {unique_labels}")
                
                # Define hate speech indicators
                hate_indicators = [
                    'hate', 'hateful', 'offensive', 'abusive', 'toxic', 'harassment',
                    'bullying', 'spam', 'threat', '1', 'yes', 'true', 'positive'
                ]
                
                def is_hate_speech(label):
                    label_str = str(label).lower().strip()
                    return any(indicator in label_str for indicator in hate_indicators)
                
                return labels.apply(lambda x: 1 if is_hate_speech(x) else 0)
            else:
                # Numeric labels
                unique_vals = sorted(labels.unique())
                if len(unique_vals) == 2:
                    # Binary: map to 0,1
                    return labels.map({unique_vals[0]: 0, unique_vals[1]: 1})
                else:
                    # Multi-class: assume 0 is non-hate, others are hate
                    return (labels != 0).astype(int)
                    
        except Exception as e:
            logger.error(f"Error normalizing labels for {dataset_name}: {e}")
            return labels
    
    @monitor_performance
    def extract_advanced_features(self, processed_data):
        """Extract multiple feature types"""
        try:
            # Text features
            texts = [item['text'] for item in processed_data]
            
            # TF-IDF features
            tfidf_features = self.vectorizers['tfidf'].transform(texts)
            
            # Linguistic features
            linguistic_features = []
            for item in processed_data:
                features = item['features']
                feature_vec = [
                    features.get('original_length', 0),
                    features.get('exclamation_count', 0),
                    features.get('question_count', 0),
                    features.get('caps_ratio', 0),
                    features.get('has_urls', 0),
                    features.get('has_mentions', 0),
                    features.get('has_hashtags', 0),
                    features.get('repeated_chars', 0),
                    features.get('word_count', 0),
                    features.get('sentiment_polarity', 0),
                    features.get('sentiment_subjectivity', 0),
                ]
                linguistic_features.append(feature_vec)
            
            linguistic_features = np.array(linguistic_features)
            
            # Combine features
            from scipy.sparse import hstack, csr_matrix
            combined_features = hstack([
                tfidf_features,
                csr_matrix(linguistic_features)
            ])
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    @monitor_performance
    def train_ensemble_model(self, X_train, y_train):
        """Train ensemble of models"""
        try:
            # Train individual models
            trained_models = []
            
            for name, model in self.models.items():
                logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
                trained_models.append((name, model))
            
            # Create ensemble
            self.ensemble = VotingClassifier(
                estimators=trained_models,
                voting='soft'  # Use probabilities
            )
            
            self.ensemble.fit(X_train, y_train)
            
            return True
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            return False
    
    @monitor_performance
    def comprehensive_evaluation(self, X_test, y_test):
        """Detailed model evaluation"""
        try:
            predictions = self.ensemble.predict(X_test)
            probabilities = self.ensemble.predict_proba(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, predictions),
                'precision': precision_recall_fscore_support(y_test, predictions, average='weighted')[0],
                'recall': precision_recall_fscore_support(y_test, predictions, average='weighted')[1],
                'f1': precision_recall_fscore_support(y_test, predictions, average='weighted')[2],
                'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
                'classification_report': classification_report(y_test, predictions, output_dict=True)
            }
            
            # AUC score
            if len(np.unique(y_test)) == 2:
                metrics['auc_roc'] = roc_auc_score(y_test, probabilities[:, 1])
            
            # Individual model performance
            for name, model in self.models.items():
                model_pred = model.predict(X_test)
                metrics[f'{name}_accuracy'] = accuracy_score(y_test, model_pred)
            
            self.performance_metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return {"error": str(e)}
    
    @monitor_performance
    def load_and_train_model(self):
        """Main training function with multiple datasets"""
        try:
            self.training_in_progress = True
            logger.info("Starting advanced model training with multiple datasets...")
            
            # Load multiple datasets
            df = self.load_multiple_datasets()
            
            if df is None or df.empty:
                raise ValueError("No datasets could be loaded")
            
            # Preprocess all texts
            logger.info("Preprocessing texts...")
            processed_data = []
            for text in df['text']:
                processed = self.advanced_preprocess_text(text)
                processed_data.append(processed)
            
            # Fit vectorizer on processed texts
            texts = [item['text'] for item in processed_data if item['text']]
            logger.info(f"Fitting vectorizer on {len(texts)} texts...")
            self.vectorizers['tfidf'].fit(texts)
            
            # Extract features
            logger.info("Extracting advanced features...")
            X = self.extract_advanced_features(processed_data)
            y = df['label'].values
            
            if X is None:
                raise ValueError("Feature extraction failed")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Training set size: {X_train.shape}")
            logger.info(f"Test set size: {X_test.shape}")
            
            # Train ensemble model
            success = self.train_ensemble_model(X_train, y_train)
            
            if not success:
                raise ValueError("Ensemble training failed")
            
            # Evaluate model
            logger.info("Evaluating model...")
            metrics = self.comprehensive_evaluation(X_test, y_test)
            
            logger.info(f"Model training completed!")
            logger.info(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            logger.info(f"F1-Score: {metrics.get('f1', 'N/A'):.4f}")
            
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            return {"error": str(e)}
        finally:
            self.training_in_progress = False
    
    def save_model(self):
        """Save the trained model and components"""
        try:
            model_data = {
                'ensemble': self.ensemble,
                'models': self.models,
                'vectorizers': self.vectorizers,
                'performance_metrics': self.performance_metrics,
                'config': self.config,
                'training_timestamp': datetime.now().isoformat()
            }
            
            with open('advanced_hate_speech_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Advanced model saved successfully")
            
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
    
    def load_model(self):
        """Load a previously saved model"""
        try:
            with open('advanced_hate_speech_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.ensemble = model_data['ensemble']
            self.models = model_data['models']
            self.vectorizers = model_data['vectorizers']
            self.performance_metrics = model_data.get('performance_metrics', {})
            
            self.is_trained = True
            logger.info("Advanced model loaded successfully")
            return True
            
        except FileNotFoundError:
            logger.info("No saved model found")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, text):
        """Predict with enhanced features"""
        if not self.is_trained:
            return {"error": "Model not trained yet"}
        
        if self.training_in_progress:
            return {"error": "Model training in progress, please wait"}
        
        try:
            # Preprocess text
            processed = self.advanced_preprocess_text(text)
            
            if not processed['text']:
                return {"error": "Text is empty after preprocessing"}
            
            # Extract features
            features = self.extract_advanced_features([processed])
            
            if features is None:
                return {"error": "Feature extraction failed"}
            
            # Make prediction
            prediction = self.ensemble.predict(features)[0]
            probabilities = self.ensemble.predict_proba(features)[0]
            
            # Get individual model predictions for transparency
            individual_predictions = {}
            for name, model in self.models.items():
                try:
                    pred = model.predict(features)[0]
                    prob = model.predict_proba(features)[0]
                    individual_predictions[name] = {
                        'prediction': int(pred),
                        'confidence': float(max(prob))
                    }
                except:
                    pass
            
            result = {
                "text": text,
                "prediction": "Hate Speech" if prediction == 1 else "Not Hate Speech",
                "confidence": float(max(probabilities)),
                "hate_probability": float(probabilities[1]) if len(probabilities) > 1 else 0.0,
                "linguistic_features": processed['features'],
                "individual_models": individual_predictions,
                "model_version": "advanced_ensemble"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"error": f"Prediction error: {str(e)}"}
    
    def predict_batch(self, texts):
        """Batch prediction for multiple texts"""
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
    
    def collect_feedback(self, text, prediction, user_feedback):
        """Collect user feedback for model improvement"""
        self.feedback_buffer.append({
            'text': text,
            'predicted': prediction,
            'actual': user_feedback,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Feedback collected. Buffer size: {len(self.feedback_buffer)}")
        
        if len(self.feedback_buffer) >= self.retrain_threshold:
            logger.info("Feedback threshold reached. Consider retraining.")
    
    def get_performance_metrics(self):
        """Get detailed model performance metrics"""
        return {
            "performance_metrics": self.performance_metrics,
            "model_status": {
                "trained": self.is_trained,
                "training_in_progress": self.training_in_progress,
                "feedback_buffer_size": len(self.feedback_buffer)
            },
            "configuration": self.config
        }

# Initialize the advanced detector
detector = AdvancedHateSpeechDetector()

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Enhanced health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_trained": detector.is_trained,
        "training_in_progress": detector.training_in_progress,
        "model_version": "advanced_ensemble",
        "environment": detector.environment,
        "datasets_configured": len(detector.datasets)
    })

@app.route('/train', methods=['POST'])
def train_model():
    """Train the advanced model"""
    if detector.training_in_progress:
        return jsonify({
            "status": "error",
            "message": "Training already in progress"
        })
    
    def train_async():
        try:
            logger.info("Starting async advanced training...")
            metrics = detector.load_and_train_model()
            
            if 'error' not in metrics:
                logger.info("Advanced training completed successfully")
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
        "message": "Advanced training started with multiple datasets. Check /status for progress.",
        "datasets": [d['name'] for d in detector.datasets]
    })

@app.route('/predict', methods=['POST'])
def predict_hate_speech():
    """Enhanced prediction endpoint"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        
        if len(text) > 2000:  # Increased limit
            return jsonify({"error": "Text too long (max 2000 characters)"}), 400
        
        result = detector.predict(text)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({"error": "No texts provided"}), 400
        
        if len(texts) > 50:  # Reasonable limit
            return jsonify({"error": "Too many texts (max 50)"}), 400
        
        result = detector.predict_batch(texts)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Collect user feedback for model improvement"""
    try:
        data = request.get_json()
        required_fields = ['text', 'prediction', 'actual_label']
        
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
        
        detector.collect_feedback(
            data['text'], 
            data['prediction'], 
            data['actual_label']
        )
        
        return jsonify({"status": "feedback recorded"})
        
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/model/metrics')
def get_model_metrics():
    """Get detailed model performance metrics"""
    try:
        metrics = detector.get_performance_metrics()
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/status')
def model_status():
    """Enhanced model status endpoint"""
    return jsonify({
        "trained": detector.is_trained,
        "training_in_progress": detector.training_in_progress,
        "status": "ready" if detector.is_trained else ("training" if detector.training_in_progress else "not trained"),
        "model_version": "advanced_ensemble",
        "datasets": [d['name'] for d in detector.datasets],
        "feedback_buffer_size": len(detector.feedback_buffer),
        "performance_metrics": detector.performance_metrics
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Auto-train model on startup
def auto_train_on_startup():
    """Automatically train model on startup if not already trained"""
    time.sleep(10)  # Wait longer for app to fully start
    if not detector.is_trained and not detector.training_in_progress:
        try:
            logger.info("Auto-training advanced model on startup...")
            detector.load_and_train_model()
        except Exception as e:
            logger.error(f"Auto-training failed: {e}")

if __name__ == '__main__':
    # Try to load existing model first
    if not detector.load_model():
        logger.info("No existing advanced model found.")
        # Start auto-training in background for production
        if os.environ.get('RENDER') or os.environ.get('ENVIRONMENT') == 'production':
            thread = Thread(target=auto_train_on_startup)
            thread.daemon = True
            thread.start()
    
    # Get port from environment variable
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )
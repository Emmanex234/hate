# config.py
import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DatasetConfig:
    """Configuration for individual datasets"""
    name: str
    kaggle_id: str
    primary: bool = False
    max_samples: int = 5000
    text_columns: List[str] = None
    label_columns: List[str] = None
    preprocessing_rules: Dict[str, Any] = None

class Config:
    """Main configuration class for the hate speech detector"""
    
    def __init__(self):
        self.environment = os.environ.get('ENVIRONMENT', 'render_free')
        self.setup_environment_config()
        self.setup_dataset_config()
        self.setup_model_config()
        self.setup_api_config()
    
    def setup_environment_config(self):
        """Configure settings based on environment"""
        if self.environment == 'production':
            self.memory_limit = '2GB'
            self.cpu_cores = 4
            self.max_concurrent_requests = 50
            self.enable_monitoring = True
            self.log_level = 'INFO'
            
        elif self.environment == 'render_free':
            self.memory_limit = '512MB'
            self.cpu_cores = 1
            self.max_concurrent_requests = 10
            self.enable_monitoring = False
            self.log_level = 'WARNING'
            
        else:  # development
            self.memory_limit = '1GB'
            self.cpu_cores = 2
            self.max_concurrent_requests = 20
            self.enable_monitoring = True
            self.log_level = 'DEBUG'
    
    def setup_dataset_config(self):
        """Configure datasets based on environment"""
        base_datasets = [
            DatasetConfig(
                name='nigerian_multilingual',
                kaggle_id='sharonibejih/nigerian-multilingual-hate-speech',
                primary=True,
                max_samples=8000,
                text_columns=['text', 'tweet', 'comment'],
                label_columns=['label', 'hate_speech', 'class']
            ),
            DatasetConfig(
                name='hate_speech_detection',
                kaggle_id='mrmorj/hate-speech-and-offensive-language-detection',
                primary=False,
                max_samples=5000,
                text_columns=['tweet', 'text'],
                label_columns=['class', 'label']
            ),
            DatasetConfig(
                name='twitter_hate_speech',
                kaggle_id='arkhoshghalb/twitter-sentiment-analysis-hatred-speech',
                primary=False,
                max_samples=3000,
                text_columns=['text', 'tweet'],
                label_columns=['label', 'class', 'hate']
            ),
            DatasetConfig(
                name='cyberbullying_detection',
                kaggle_id='andrewmvd/cyberbullying-classification',
                primary=False,
                max_samples=4000,
                text_columns=['text', 'tweet'],
                label_columns=['cyberbullying_type', 'label']
            )
        ]
        
        # Additional datasets for production environment
        if self.environment == 'production':
            additional_datasets = [
                DatasetConfig(
                    name='toxic_comments',
                    kaggle_id='julian3833/jigsaw-toxic-comment-classification-challenge',
                    primary=False,
                    max_samples=10000,
                    text_columns=['comment_text'],
                    label_columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
                ),
                DatasetConfig(
                    name='hate_offensive',
                    kaggle_id='zackthoutt/hate-speech-and-offensive-language',  
                    primary=False,
                    max_samples=6000,
                    text_columns=['tweet'],
                    label_columns=['class']
                )
            ]
            base_datasets.extend(additional_datasets)
        
        # Limit datasets for memory-constrained environments
        if self.environment == 'render_free':
            self.datasets = base_datasets[:3]  # Use only first 3 datasets
            for dataset in self.datasets:
                dataset.max_samples = min(dataset.max_samples, 2000)
        else:
            self.datasets = base_datasets
    
    def setup_model_config(self):
        """Configure model parameters based on environment"""
        if self.environment == 'production':
            self.model_config = {
                'max_features': 15000,
                'ngram_range': (1, 4),
                'use_ensemble': True,
                'ensemble_models': ['tfidf_lr', 'tfidf_svm', 'tfidf_nb'],
                'ensemble_weights': [0.4, 0.4, 0.2],
                'cross_validation_folds': 5,
                'hyperparameter_tuning': True,
                'retrain_frequency': 'weekly',
                'min_df': 2,
                'max_df': 0.95,
                'use_advanced_features': True,
                'feature_selection': True
            }
            
        elif self.environment == 'render_free':
            self.model_config = {
                'max_features': 3000,
                'ngram_range': (1, 2),
                'use_ensemble': True,
                'ensemble_models': ['tfidf_lr', 'tfidf_nb'],
                'ensemble_weights': [0.7, 0.3],
                'cross_validation_folds': 3,
                'hyperparameter_tuning': False,
                'retrain_frequency': 'monthly',
                'min_df': 3,
                'max_df': 0.9,
                'use_advanced_features': False,
                'feature_selection': False
            }
            
        else:  # development
            self.model_config = {
                'max_features': 8000,
                'ngram_range': (1, 3),
                'use_ensemble': True,
                'ensemble_models': ['tfidf_lr', 'tfidf_svm', 'tfidf_nb'],
                'ensemble_weights': [0.4, 0.35, 0.25],
                'cross_validation_folds': 4,
                'hyperparameter_tuning': True,
                'retrain_frequency': 'daily',
                'min_df': 2,
                'max_df': 0.92,
                'use_advanced_features': True,
                'feature_selection': True
            }
    
    def setup_api_config(self):
        """Configure API settings based on environment"""
        base_api_config = {
            'host': '0.0.0.0',
            'debug': self.environment == 'development',
            'cors_enabled': True,
            'rate_limiting': True,
            'request_timeout': 30,
            'max_text_length': 1000,
            'batch_processing': False,
            'cache_predictions': True,
            'cache_ttl': 3600,  # 1 hour
            'enable_logging': True,
            'log_predictions': self.environment != 'production'
        }
        
        if self.environment == 'production':
            self.api_config = {
                **base_api_config,
                'port': int(os.environ.get('PORT', 8000)),
                'workers': 4,
                'rate_limit': '100/minute',
                'enable_auth': True,
                'api_key_required': True,
                'ssl_enabled': True,
                'batch_processing': True,
                'max_batch_size': 50,
                'cache_ttl': 7200,  # 2 hours
                'monitoring_endpoint': '/health',
                'metrics_endpoint': '/metrics'
            }
            
        elif self.environment == 'render_free':
            self.api_config = {
                **base_api_config,
                'port': int(os.environ.get('PORT', 10000)),
                'workers': 1,
                'rate_limit': '20/minute',
                'enable_auth': False,
                'api_key_required': False,
                'ssl_enabled': False,
                'max_text_length': 500,
                'cache_ttl': 1800,  # 30 minutes
                'monitoring_endpoint': '/status'
            }
            
        else:  # development
            self.api_config = {
                **base_api_config,
                'port': int(os.environ.get('PORT', 5000)),
                'workers': 2,
                'rate_limit': '200/minute',
                'enable_auth': False,
                'api_key_required': False,
                'ssl_enabled': False,
                'reload': True,
                'monitoring_endpoint': '/dev-status',
                'test_endpoint': '/test'
            }
    
    def get_kaggle_credentials(self):
        """Get Kaggle API credentials from environment variables"""
        return {
            'username': os.environ.get('KAGGLE_USERNAME'),
            'key': os.environ.get('KAGGLE_KEY')
        }
    
    def get_database_config(self):
        """Get database configuration based on environment"""
        if self.environment == 'production':
            return {
                'url': os.environ.get('DATABASE_URL'),
                'pool_size': 10,
                'max_overflow': 20,
                'pool_timeout': 30,
                'pool_recycle': 3600
            }
        elif self.environment == 'render_free':
            return {
                'url': os.environ.get('DATABASE_URL', 'sqlite:///hate_speech.db'),
                'pool_size': 2,
                'max_overflow': 5,
                'pool_timeout': 10,
                'pool_recycle': 1800
            }
        else:  # development
            return {
                'url': 'sqlite:///dev_hate_speech.db',
                'pool_size': 5,
                'max_overflow': 10,
                'pool_timeout': 20,
                'pool_recycle': 1800,
                'echo': True  # SQL logging for development
            }
    
    def validate_config(self):
        """Validate configuration settings"""
        required_vars = []
        
        if self.environment == 'production':
            required_vars = ['DATABASE_URL', 'KAGGLE_USERNAME', 'KAGGLE_KEY']
        elif self.environment == 'render_free':
            required_vars = ['KAGGLE_USERNAME', 'KAGGLE_KEY']
        
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Validate memory limits
        memory_mb = int(self.memory_limit.replace('MB', '').replace('GB', '')) * (1024 if 'GB' in self.memory_limit else 1)
        if memory_mb < 256:
            raise ValueError(f"Memory limit too low: {self.memory_limit}")
        
        return True

# Create global config instance
config = Config()

# Validate configuration on import
try:
    config.validate_config()
except ValueError as e:
    print(f"Configuration warning: {e}")
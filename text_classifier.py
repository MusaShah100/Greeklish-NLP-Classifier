#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Traditional Machine Learning Models for Greeklish-English Text Classification

This script trains traditional ML models (Logistic Regression, SVM, Random Forest)
on the preprocessed text data and evaluates their performance.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import pickle
import re
import json
from datetime import datetime
import joblib

class SimpleGreeklishClassifier:
    """Simple classifier for Greeklish vs English text."""
    
    def __init__(self):
        """Initialize the classifier."""
        # Directory paths
        self.data_path = 'data/processed_texts.csv'  # Update data path
        self.models_dir = 'models'
        self.results_dir = 'evaluation'
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Other attributes
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.vectorizers = {}
        self.models = {}
        self.results = {}
    
    def load_and_prepare_data(self):
        """Load and prepare data for training."""
        print("Loading and preparing data...")
        
        # Load the data
        try:
            df = pd.read_csv(self.data_path)
            print(f"Loaded {len(df)} records from {self.data_path}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
        
        # Check required columns
        if 'cleaned_text' in df.columns and 'label' in df.columns:
            text_column = 'cleaned_text'
        elif 'tokens' in df.columns and 'label' in df.columns:
            text_column = 'tokens'
        elif 'text' in df.columns and 'label' in df.columns:
            text_column = 'text'
        else:
            print(f"Data missing required columns. Found: {df.columns.tolist()}")
            print(f"Required: text and label columns")
            return False
        
        print(f"Using '{text_column}' column for text features")
        
        # Display data distribution
        print("\nClass Distribution:")
        print(df['label'].value_counts())
        
        # Split data into train and test sets
        texts = df[text_column].values
        labels = df['label'].values
        
        # Convert labels to binary format (1 for greeklish, 0 for english)
        labels_binary = np.array([1 if label == 'greeklish' else 0 for label in labels])
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            texts, labels_binary, test_size=0.2, random_state=42, stratify=labels_binary
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}")
        
        return True
    
    def train_models(self):
        """Train traditional ML models with TF-IDF features."""
        print("\n==== Training Models with TF-IDF ====")
        
        # Create TF-IDF vectorizer with settings optimized for Greeklish detection
        tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            analyzer='char_wb',  # Character n-grams with word boundaries are better for language detection
            sublinear_tf=True,   # Apply sublinear tf scaling (1 + log(tf))
            min_df=2             # Minimum document frequency
        )
        
        # Keep the vectorizer for later use
        self.vectorizers['tfidf'] = tfidf_vectorizer
        
        # Transform the data
        X_train_tfidf = tfidf_vectorizer.fit_transform(self.X_train)
        X_test_tfidf = tfidf_vectorizer.transform(self.X_test)
        
        # Add custom Greeklish features
        X_train_enhanced = self.add_greeklish_features(X_train_tfidf, self.X_train)
        X_test_enhanced = self.add_greeklish_features(X_test_tfidf, self.X_test)
        
        # Train and evaluate Logistic Regression
        print("\nTraining Logistic Regression...")
        lr_model = self.train_logistic_regression(X_train_enhanced, self.y_train)
        self.models['logistic_regression'] = lr_model
        
        # Make predictions
        lr_preds = lr_model.predict(X_test_enhanced)
        
        # Evaluate
        lr_accuracy = accuracy_score(self.y_test, lr_preds)
        print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
        print("\nClassification Report:")
        lr_report = classification_report(self.y_test, lr_preds, 
                                        target_names=['english', 'greeklish'],
                                        output_dict=True)
        print(classification_report(self.y_test, lr_preds, 
                                    target_names=['english', 'greeklish']))
        
        # Store results
        self.results['logistic_regression'] = {
            'accuracy': lr_accuracy,
            'report': lr_report
        }
        
        # Train and evaluate SVM
        print("\nTraining SVM...")
        svm_model = self.train_svm(X_train_enhanced, self.y_train)
        self.models['svm'] = svm_model
        
        # Make predictions
        svm_preds = svm_model.predict(X_test_enhanced)
        
        # Evaluate
        svm_accuracy = accuracy_score(self.y_test, svm_preds)
        print(f"SVM Accuracy: {svm_accuracy:.4f}")
        print("\nClassification Report:")
        svm_report = classification_report(self.y_test, svm_preds, 
                                          target_names=['english', 'greeklish'],
                                          output_dict=True)
        print(classification_report(self.y_test, svm_preds, 
                                    target_names=['english', 'greeklish']))
        
        # Store results
        self.results['svm'] = {
            'accuracy': svm_accuracy,
            'report': svm_report
        }
        
        # Train and evaluate Random Forest
        print("\nTraining Random Forest...")
        rf_model = self.train_random_forest(X_train_enhanced, self.y_train)
        self.models['random_forest'] = rf_model
        
        # Make predictions
        rf_preds = rf_model.predict(X_test_enhanced)
        
        # Evaluate
        rf_accuracy = accuracy_score(self.y_test, rf_preds)
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        print("\nClassification Report:")
        rf_report = classification_report(self.y_test, rf_preds, 
                                         target_names=['english', 'greeklish'],
                                         output_dict=True)
        print(classification_report(self.y_test, rf_preds, 
                                   target_names=['english', 'greeklish']))
        
        # Store results
        self.results['random_forest'] = {
            'accuracy': rf_accuracy,
            'report': rf_report
        }
        
        # Save models
        with open(f'{self.models_dir}/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        
        with open(f'{self.models_dir}/logistic_regression_model.pkl', 'wb') as f:
            pickle.dump(lr_model, f)
        
        with open(f'{self.models_dir}/svm_model.pkl', 'wb') as f:
            pickle.dump(svm_model, f)
        
        with open(f'{self.models_dir}/random_forest_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)
        
        # Save metadata about custom features
        with open(f'{self.models_dir}/custom_features_info.json', 'w') as f:
            json.dump({
                'num_custom_features': 5,
                'feature_names': [
                    'vowel_consonant_ratio',
                    'greeklish_combo_frequency',
                    'repeated_vowels_frequency',
                    'mp_frequency',
                    'nt_frequency'
                ]
            }, f)
        
        print("\nAll models trained and saved successfully.")
    
    def train_logistic_regression(self, X_train, y_train):
        """Train a logistic regression model with cross-validation."""
        model = LogisticRegression(
            C=1.0,  # Regularization strength
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model
        model.fit(X_train, y_train)
        return model
    
    def train_svm(self, X_train, y_train):
        """Train an SVM model with cross-validation."""
        model = SVC(
            kernel='linear',
            C=1.0,
            class_weight='balanced',
            random_state=42,
            probability=True
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model
        model.fit(X_train, y_train)
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train a random forest model with cross-validation."""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model
        model.fit(X_train, y_train)
        return model
    
    def add_greeklish_features(self, X_vec, texts):
        """Add custom features to help detect Greeklish text."""
        greeklish_features = np.zeros((len(texts), 8))
        
        for i, text in enumerate(texts):
            text = text.lower()
            
            # Check for Greek alphabet characters
            greek_chars = sum(1 for c in text if '\u0370' <= c <= '\u03FF')
            if greek_chars / (len(text) + 1) > 0.3:  # If more than 30% Greek characters
                greeklish_features[i] = [0, 0, 0, 0, 0, 0, 1, 0]  # Mark as Greek alphabet
                continue
            
            # Feature 1: Ratio of Latin characters used in Greek
            greek_like = sum(1 for c in text if c in 'abcdefghiklmnopqrstuvwxyz')
            greeklish_features[i, 0] = greek_like / (len(text) + 1)
            
            # Feature 2: Common Greeklish digraphs
            greeklish_features[i, 1] = len(re.findall(r'(th|ch|ps|ks|mp|nt|ts|tz|gk|nx)', text)) / (len(text) + 1)
            
            # Feature 3: Vowel patterns common in Greeklish
            greeklish_features[i, 2] = len(re.findall(r'(ou|ei|ai|oi|ui|au|eu)', text)) / (len(text) + 1)
            
            # Feature 4: Consonant patterns common in Greeklish
            greeklish_features[i, 3] = len(re.findall(r'[bcdfgjklmnpqrstvwxz]{3,}', text)) / (len(text) + 1)
            
            # Feature 5: Common Greeklish endings
            greeklish_features[i, 4] = len(re.findall(r'(os|as|is|eis|ous|es|ai|oi)$', text)) / (len(text) + 1)
            
            # Feature 6: Ratio of Greek-like word structure
            words = text.split()
            greek_like_words = sum(1 for w in words if re.search(r'^[bcdfgjklmnpqrstvwxz]*[aeiou]+', w))
            greeklish_features[i, 5] = greek_like_words / (len(words) + 1)
            
            # Feature 7: Is Greek alphabet (already set to 0)
            
            # Feature 8: Mixed language indicator
            eng_words = sum(1 for w in words if re.match(r'^[a-z]+$', w) and len(w) > 2)
            greeklish_features[i, 7] = eng_words / (len(words) + 1)
        
        # Combine with TF-IDF features
        X_enhanced = np.hstack([X_vec, greeklish_features])
        return X_enhanced
    
    def compare_models(self):
        """Compare the performance of all trained models."""
        print("\n==== Model Comparison ====")
        
        # Extract metrics for comparison
        model_names = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for model_name, result in self.results.items():
            model_names.append(model_name)
            accuracies.append(result['accuracy'])
            
            # Extract metrics from classification report
            precisions.append(result['report']['greeklish']['precision'])
            recalls.append(result['report']['greeklish']['recall'])
            f1_scores.append(result['report']['greeklish']['f1-score'])
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Model': model_names,
            'Accuracy': accuracies,
            'Precision': precisions,
            'Recall': recalls,
            'F1-Score': f1_scores
        })
        
        # Sort by accuracy
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        # Display comparison
        print("\nModel Performance Comparison:")
        print(comparison_df)
        
        # Save comparison to CSV
        comparison_df.to_csv(f'{self.results_dir}/model_comparison.csv', index=False)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i+1)
            plt.bar(comparison_df['Model'], comparison_df[metric])
            plt.ylabel(metric)
            plt.title(f'Model Comparison - {metric}')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/model_comparison.png')
        
        # Determine best model
        best_model = comparison_df.iloc[0]['Model']
        print(f"\nBest performing model: {best_model}")
        
        return comparison_df
    
    def save_best_model(self):
        """Save the best model based on F1-score."""
        # Load comparison data
        comparison_df = pd.read_csv(f'{self.results_dir}/model_comparison.csv')
        
        # Sort by F1-Score
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        # Determine best model
        best_model = comparison_df.iloc[0]['Model']
        print(f"\nBest model (by F1-Score): {best_model}")
        
        # Copy best model to final model file
        with open(f'{self.models_dir}/{best_model}_model.pkl', 'rb') as src:
            with open(f'{self.models_dir}/final_model.pkl', 'wb') as dst:
                dst.write(src.read())
        with open(f'{self.models_dir}/tfidf_vectorizer.pkl', 'rb') as src:
            with open(f'{self.models_dir}/final_vectorizer.pkl', 'wb') as dst:
                dst.write(src.read())
        print(f"Saved {best_model} as final model.")
        
        # Save model type
        with open(f'{self.models_dir}/model_type.txt', 'w') as f:
            f.write(best_model)
            
        print("Best model saved as final_model.")
    
    def create_inference_function(self):
        """Create an inference function for the best model."""
        # Determine best model type
        with open(f'{self.models_dir}/model_type.txt', 'r') as f:
            model_type = f.read().strip()
        
        print(f"\nCreating inference function for {model_type} model...")
        
        # Create inference.py file
        inference_code = '''
import pickle
import re
import os
import numpy as np

class TextClassifier:
    def __init__(self, model_dir="models"):
        """Initialize the classifier."""
        self.model_dir = model_dir
        
        # Check if model files exist
        if not os.path.exists(f"{model_dir}/final_model.pkl"):
            raise FileNotFoundError(f"Model file not found at {model_dir}/final_model.pkl")
        if not os.path.exists(f"{model_dir}/final_vectorizer.pkl"):
            raise FileNotFoundError(f"Vectorizer file not found at {model_dir}/final_vectorizer.pkl")
        
        # Determine model type (if exists)
        if os.path.exists(f"{model_dir}/model_type.txt"):
            with open(f"{model_dir}/model_type.txt", 'r') as f:
                self.model_type = f.read().strip()
        else:
            self.model_type = "unknown"
        
        # Load model and vectorizer
        with open(f"{model_dir}/final_model.pkl", 'rb') as f:
            self.model = pickle.load(f)
        with open(f"{model_dir}/final_vectorizer.pkl", 'rb') as f:
            self.vectorizer = pickle.load(f)
            
        # Check if custom features are used
        self.uses_custom_features = os.path.exists(f"{model_dir}/custom_features_info.json")
        
        if self.uses_custom_features:
            import json
            with open(f"{model_dir}/custom_features_info.json", 'r') as f:
                self.feature_info = json.load(f)
            print("Using enhanced Greeklish detection with custom features")
    
    def clean_text(self, text):
        """Basic text cleaning."""
        text = text.lower()
        text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\\s+', ' ', text)
        return text.strip()
    
    def add_greeklish_features(self, X_vec, text):
        """Add custom features to help detect Greeklish text."""
        # Convert sparse matrix to dense for feature addition
        X_dense = X_vec.toarray()
        
        # Create a feature vector with additional Greeklish indicators
        additional_features = np.zeros(8)
        
        # Feature 1: Ratio of Latin characters used in Greek
        greek_like = sum(1 for c in text if c in 'abcdefghiklmnopqrstuvwxyz')
        additional_features[0] = greek_like / (len(text) + 1)
        
        # Feature 2: Common Greeklish digraphs
        greeklish_combos = ['th', 'ch', 'ps', 'ks', 'mp', 'nt', 'ts', 'tz', 'gk', 'nx']
        combo_count = sum(text.count(combo) for combo in greeklish_combos)
        additional_features[1] = combo_count / (len(text) + 1)
        
        # Feature 3: Vowel patterns common in Greeklish
        vowels = 'ouei'
        vowel_count = sum(text.count(vowel) for vowel in vowels)
        additional_features[2] = vowel_count / (len(text) + 1)
        
        # Feature 4: Consonant patterns common in Greeklish
        consonants = 'bcdfgjklmnpqrstvwxz'
        consonant_count = sum(text.count(consonant) for consonant in consonants)
        additional_features[3] = consonant_count / (len(text) + 1)
        
        # Feature 5: Common Greeklish endings
        endings = 'osasis'
        ending_count = sum(text.endswith(ending) for ending in endings)
        additional_features[4] = ending_count / (len(text) + 1)
        
        # Feature 6: Ratio of Greek-like word structure
        words = text.split()
        greek_like_words = sum(1 for w in words if re.search(r'^[bcdfgjklmnpqrstvwxz]*[aeiou]+', w))
        additional_features[5] = greek_like_words / (len(words) + 1)
        
        # Feature 7: Is Greek alphabet (already set to 0)
        
        # Feature 8: Mixed language indicator
        eng_words = sum(1 for w in words if re.match(r'^[a-z]+$', w) and len(w) > 2)
        additional_features[7] = eng_words / (len(words) + 1)
        
        # Combine TF-IDF features with custom features
        X_enhanced = np.hstack((X_dense, additional_features.reshape(1, -1)))
        
        return X_enhanced
    
    def predict(self, text):
        """Predict the class of a text."""
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Transform text with vectorizer
        text_vector = self.vectorizer.transform([cleaned_text])
        
        # Add custom features if used in training
        if self.uses_custom_features:
            text_vector = self.add_greeklish_features(text_vector, cleaned_text)
        
        # Make prediction
        prediction = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0]
        
        # Important: In the training, 0 = English, 1 = Greeklish
        is_greeklish = (prediction == 1)
        confidence = probabilities[1 if is_greeklish else 0]  # Get the probability for the predicted class
        
        # Check for Greeklish patterns directly for very short texts
        if len(cleaned_text) < 15 and not is_greeklish:
            # Additional check for short texts that might be misclassified
            greeklish_combos = ['th', 'ph', 'ps', 'ks', 'ts', 'ch', 'ou', 'ei', 'ai', 'oi', 'av', 'ev', 'mp', 'nt']
            combo_count = sum(cleaned_text.count(combo) for combo in greeklish_combos)
            if combo_count / len(cleaned_text) > 0.25:  # High density of Greeklish patterns
                is_greeklish = True
                confidence = 0.75  # Moderate confidence for this override
        
        # Return result
        return {
            'prediction': 'greeklish' if is_greeklish else 'english',
            'probability': float(confidence),
            'cleaned_text': cleaned_text
        }

# Example usage when run directly
if __name__ == "__main__":
    try:
        classifier = TextClassifier()
        
        # Test with some examples
        examples = [
            "This is pure English text with no Greeklish elements.",
            "Auto einai ena mikro paradeigma Greeklish.",
            "Kalimera! Ti kaneis simera?",
            "Machine learning is a fascinating field of study."
        ]
        
        print("\n===== Testing Greeklish-English Classifier =====")
        for example in examples:
            result = classifier.predict(example)
            print(f"\nText: {example}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['probability']:.4f}")
        
        print("\n===== Interactive Mode =====")
        print("Type a text to classify (or type 'quit', 'exit', or 'q' to exit):")
        
        while True:
            user_input = input("\nEnter text (or 'quit' to exit): ")
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Exiting interactive mode.")
                break
                
            result = classifier.predict(user_input)
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['probability']:.4f}")
    
    except Exception as e:
        print(f"Error: {e}")
'''
        
        with open('simple_inference.py', 'w') as f:
            f.write(inference_code.strip())
        
        print("Inference function created in simple_inference.py")

    def save_model(self, model_name):
        """Save the model and vectorizer."""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save model and vectorizer using joblib
        joblib.dump(self.models[model_name], os.path.join('models', 'final_model.joblib'))
        joblib.dump(self.vectorizers[model_name], os.path.join('models', 'vectorizer.joblib'))
        
        print(f"Best model saved as final_model.")

def main():
    """Main function to train and evaluate models."""
    print("==== Simplified Greeklish vs English Text Classification ====")
    print("Note: This script uses only traditional ML models (no TensorFlow required)")
    
    # Check if processed_texts.csv exists
    if not os.path.exists('data/processed_texts.csv'):  # Update path check
        print("Error: data/processed_texts.csv not found.")
        print("Make sure to run the text scraping and preprocessing steps first.")
        return
    
    try:
        # Initialize classifier
        classifier = SimpleGreeklishClassifier()
        
        # Load and prepare data
        if not classifier.load_and_prepare_data():
            print("Error preparing data. Exiting.")
            return
        
        # Train models
        classifier.train_models()
        
        # Compare models
        comparison_df = classifier.compare_models()
        
        # Save best model
        classifier.save_best_model()
        
        # Create inference function
        classifier.create_inference_function()
        
        print("\n==== Training Complete ====")
        print("Results and models saved in 'evaluation/' and 'models/' directories.")
        print("To make predictions, use simple_inference.py")
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
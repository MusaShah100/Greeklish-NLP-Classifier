#!/usr/bin/env python3
"""
Evaluate the trained Greeklish-English classifier with detailed metrics and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve
)
import pickle
import os
import re
from collections import Counter
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

class ClassifierEvaluator:
    def __init__(self, model_dir='models', data_path='data/processed_texts.csv'):
        self.model_dir = model_dir
        self.data_path = data_path
        self.model = None
        self.vectorizer = None
        self.model_type = None
        self.results = {}
        self.text_column = None
        
        # Create output directory for visualizations
        os.makedirs('evaluation', exist_ok=True)
    
    def load_model(self):
        """Load the trained model and vectorizer."""
        try:
            self.model = joblib.load(os.path.join('models', 'final_model.joblib'))
            self.vectorizer = joblib.load(os.path.join('models', 'vectorizer.joblib'))
            print(f"Model type: {type(self.model).__name__}")
            print("Model and vectorizer loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def prepare_test_data(self):
        """Prepare test data using the same vectorizer used during training."""
        try:
            # Load and preprocess test data
            df = pd.read_csv(self.data_path)
            print(f"Dataset loaded: {len(df)} records")
            
            # Use the same text column as training
            if self.text_column is None:
                if 'cleaned_text' in df.columns:
                    self.text_column = 'cleaned_text'
                elif 'tokens' in df.columns:
                    self.text_column = 'tokens'
                else:
                    self.text_column = 'text'
            
            print(f"Using '{self.text_column}' column for text features")
            texts = df[self.text_column].values
            
            # Use stratified sampling for test set
            _, X_test, _, y_test = train_test_split(
                texts, df['label'].values,
                test_size=0.2,
                random_state=42,
                stratify=df['label'].values
            )
            
            print(f"Test set prepared: {len(X_test)} samples")
            print("Class distribution:")
            print(pd.Series(y_test).value_counts())
            print()
            
            # Transform using the loaded vectorizer
            X_test_transformed = self.vectorizer.transform(X_test)
            
            # Add custom features if the model expects them
            if hasattr(self.model, 'n_features_in_'):
                expected_features = self.model.n_features_in_
                current_features = X_test_transformed.shape[1]
                
                if expected_features != current_features:
                    print(f"Adding custom features to match expected {expected_features} features...")
                    X_test_transformed = self.add_custom_features(X_test_transformed, X_test)
            
            self.X_test = X_test
            self.X_test_transformed = X_test_transformed
            self.y_test = y_test
            
        except Exception as e:
            print(f"Error preparing test data: {str(e)}")
            raise
    
    def add_custom_features(self, X_vec, texts):
        """Add custom features to match the training features."""
        # Add Greeklish detection features
        greeklish_features = np.zeros((len(texts), 5))
        for i, text in enumerate(texts):
            # Ratio of Greek-like characters
            greek_chars = sum(1 for c in text.lower() if c in 'abcdefghiklmnopqrstuvwxyz')
            total_chars = len(text)
            if total_chars > 0:
                greeklish_features[i, 0] = greek_chars / total_chars
            
            # Common Greeklish patterns
            greeklish_features[i, 1] = len(re.findall(r'[aeiou]{2,}', text.lower())) / (len(text) + 1)
            greeklish_features[i, 2] = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]{3,}', text.lower())) / (len(text) + 1)
            greeklish_features[i, 3] = len(re.findall(r'[aeiou]', text.lower())) / (len(text) + 1)
            greeklish_features[i, 4] = len(re.findall(r'th|ch|ps|ks', text.lower())) / (len(text) + 1)
        
        # Combine with TF-IDF features
        X_enhanced = hstack([X_vec, greeklish_features])
        return X_enhanced
    
    def evaluate(self):
        """Evaluate the model on test data."""
        print("\nEvaluating model performance...")
        
        # Transform test data
        X_test_transformed = self.X_test_transformed
        
        # Make predictions
        self.y_pred = self.model.predict(X_test_transformed)
        self.y_pred_proba = self.model.predict_proba(X_test_transformed)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        
        # Store results
        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(self.y_test, self.y_pred),
            'classification_report': classification_report(
                self.y_test, self.y_pred, 
                target_names=['english', 'greeklish'], 
                output_dict=True
            )
        }
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred, target_names=['english', 'greeklish']))
    
    def plot_confusion_matrix(self):
        """Plot and save confusion matrix."""
        print("\nGenerating confusion matrix visualization...")
        
        cm = self.results['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['english', 'greeklish'],
            yticklabels=['english', 'greeklish']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save figure
        plt.savefig('evaluation/confusion_matrix.png', dpi=300)
        print("Confusion matrix saved to evaluation/confusion_matrix.png")
    
    def plot_roc_curve(self):
        """Plot and save ROC curve."""
        print("\nGenerating ROC curve...")
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        # Save figure
        plt.savefig('evaluation/roc_curve.png', dpi=300)
        print("ROC curve saved to evaluation/roc_curve.png")
    
    def plot_precision_recall_curve(self):
        """Plot and save precision-recall curve."""
        print("\nGenerating precision-recall curve...")
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Plot precision-recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.fill_between(recall, precision, alpha=0.2, color='blue')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(alpha=0.3)
        
        # Save figure
        plt.savefig('evaluation/precision_recall_curve.png', dpi=300)
        print("Precision-recall curve saved to evaluation/precision_recall_curve.png")
    
    def analyze_feature_importance(self, top_n=20):
        """Analyze feature importance for interpretable models."""
        print("\nAnalyzing feature importance...")
        
        # Check if model supports feature importance
        if not hasattr(self.model, 'coef_') and not hasattr(self.model, 'feature_importances_'):
            print("Model does not support direct feature importance analysis")
            return
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Extract importance values based on model type
        if hasattr(self.model, 'coef_'):
            # For linear models like Logistic Regression and SVM
            importance = self.model.coef_[0]
        else:
            # For tree models like Random Forest
            importance = self.model.feature_importances_
        
        # Create DataFrame of features and importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(importance)  # Using absolute values for importance
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Plot top features
        top_features = feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # Save figure
        plt.savefig('evaluation/feature_importance.png', dpi=300)
        
        # Save feature importance to CSV
        feature_importance.to_csv('evaluation/feature_importance.csv', index=False)
        print(f"Top {top_n} feature importance saved to evaluation/feature_importance.png")
        print("Full feature importance saved to evaluation/feature_importance.csv")
    
    def plot_metrics_comparison(self):
        """Plot and save comparison of key metrics."""
        print("\nGenerating metrics comparison visualization...")
        
        # Get metrics
        metrics = {
            'Accuracy': self.results['accuracy'],
            'Precision': self.results['precision'],
            'Recall': self.results['recall'],
            'F1 Score': self.results['f1_score']
        }
        
        # Create DataFrame
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })
        
        # Plot metrics
        plt.figure(figsize=(10, 6))
        bar_plot = sns.barplot(x='Metric', y='Value', data=metrics_df, palette='viridis')
        
        # Add value labels on top of each bar
        for i, value in enumerate(metrics_df['Value']):
            bar_plot.text(i, value + 0.01, f'{value:.4f}', 
                         ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'Performance Metrics - {type(self.model).__name__.replace("_", " ").title()}')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        # Save figure
        plt.savefig('evaluation/metrics_comparison.png', dpi=300)
        print("Metrics comparison saved to evaluation/metrics_comparison.png")
    
    def misclassified_examples(self, n=10):
        """Identify and analyze misclassified examples."""
        print("\nAnalyzing misclassified examples...")
        
        # Find misclassified indices
        misclassified_indices = np.where(self.y_pred != self.y_test)[0]
        
        if len(misclassified_indices) == 0:
            print("No misclassified examples found!")
            return
        
        # Create DataFrame with misclassified examples
        misclassified_df = pd.DataFrame({
            'text': self.X_test[misclassified_indices],
            'true_label': ['english' if y == 0 else 'greeklish' for y in self.y_test[misclassified_indices]],
            'predicted_label': ['english' if y == 0 else 'greeklish' for y in self.y_pred[misclassified_indices]],
            'confidence': self.y_pred_proba[misclassified_indices]
        })
        
        # Sort by confidence
        misclassified_df = misclassified_df.sort_values('confidence', ascending=False)
        
        # Save misclassified examples to CSV
        misclassified_df.to_csv('evaluation/misclassified_examples.csv', index=False)
        
        # Print sample of misclassified examples
        print(f"\nTop {min(n, len(misclassified_df))} misclassified examples:")
        for i, (_, row) in enumerate(misclassified_df.head(n).iterrows()):
            print(f"{i+1}. Text: {row['text'][:100]}{'...' if len(row['text']) > 100 else ''}")
            print(f"   True label: {row['true_label']}, Predicted label: {row['predicted_label']}")
            print(f"   Confidence: {row['confidence']:.4f}")
            print("")
        
        print(f"All {len(misclassified_df)} misclassified examples saved to evaluation/misclassified_examples.csv")
    
    def evaluate_and_visualize(self):
        """Run full evaluation and create all visualizations."""
        try:
            # Load model
            self.load_model()
            
            # Prepare test data
            self.prepare_test_data()
            
            # Evaluate model
            self.evaluate()
            
            # Generate visualizations
            self.plot_confusion_matrix()
            self.plot_roc_curve()
            self.plot_precision_recall_curve()
            self.analyze_feature_importance()
            self.plot_metrics_comparison()
            self.misclassified_examples()
            
            # Save overall results to CSV
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                'Value': [
                    self.results['accuracy'], 
                    self.results['precision'], 
                    self.results['recall'], 
                    self.results['f1_score']
                ]
            })
            metrics_df.to_csv('evaluation/performance_metrics.csv', index=False)
            
            print("\nEvaluation complete! All results saved to 'evaluation' directory.")
            print("Files generated:")
            print("- confusion_matrix.png - Visualization of correct/incorrect predictions")
            print("- roc_curve.png - Receiver Operating Characteristic curve")
            print("- precision_recall_curve.png - Precision-Recall trade-off")
            print("- feature_importance.png - Top important features for classification")
            print("- feature_importance.csv - Full list of feature importance")
            print("- metrics_comparison.png - Comparison of performance metrics")
            print("- performance_metrics.csv - Summary of performance metrics")
            print("- misclassified_examples.csv - Analysis of misclassified examples")
            
        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("==== Greeklish-English Classifier Evaluation ====")
    evaluator = ClassifierEvaluator()
    evaluator.evaluate_and_visualize() 
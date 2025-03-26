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
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def add_greeklish_features(self, X_vec, text):
        """Add custom features to help detect Greeklish text."""
        # Convert sparse matrix to dense for feature addition
        X_dense = X_vec.toarray()
        
        # Create a feature vector with additional Greeklish indicators
        additional_features = np.zeros(5)
        
        # Feature 1: Ratio of vowels to consonants (higher in Greeklish)
        text_str = text.lower()
        vowels = sum(1 for c in text_str if c in 'aeiouy')
        consonants = sum(1 for c in text_str if c in 'bcdfghjklmnpqrstvwxz')
        vowel_ratio = vowels / (consonants + 1)  # Add 1 to avoid division by zero
        additional_features[0] = vowel_ratio
        
        # Feature 2: Presence of characteristic Greeklish letter combinations
        greeklish_combos = ['th', 'ph', 'ps', 'ks', 'ts', 'ch', 'ou', 'ei', 'ai', 'oi', 'av', 'ev']
        combo_count = sum(text_str.count(combo) for combo in greeklish_combos)
        additional_features[1] = combo_count / (len(text_str) + 1)
        
        # Feature 3: Frequency of repeated vowels (common in Greeklish)
        repeated_vowels = sum(1 for v in 'aeiouy' if v+v in text_str)
        additional_features[2] = repeated_vowels / (len(text_str) + 1)
        
        # Feature 4: Presence of 'mp' which often represents 'b' in Greeklish
        mp_count = text_str.count('mp')
        additional_features[3] = mp_count / (len(text_str) + 1)
        
        # Feature 5: Presence of 'nt' which often represents 'd' in Greeklish
        nt_count = text_str.count('nt')
        additional_features[4] = nt_count / (len(text_str) + 1)
        
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
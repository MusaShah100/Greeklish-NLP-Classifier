#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the Greeklish-English Text Classifier.
This script demonstrates the classifier by predicting the language of example texts.
"""

import sys
from simple_inference import TextClassifier

def main():
    """Test the Greeklish-English text classifier with example texts."""
    
    # Initialize the classifier
    print("Initializing text classifier...")
    classifier = TextClassifier()
    
    # Example texts
    examples = [
        # English examples
        "This is a sample English text for testing the classifier.",
        "Machine learning models can effectively distinguish between different languages.",
        "The performance metrics show high accuracy on the test set.",
        
        # Greeklish examples - clear examples with distinctive patterns
        "Kalimera file! Ti nea? Pws eiste simera? Elpizoume na eiste kala.",
        "I glossa einai ena poly simantiko ergaleio epikoinonias kai politismou.",
        "To Greeklish xrisimopoieitai syxna sta mesa koinonikis diktyosis kai stis syntomografies.",
        "Signomi pou argisa na apantiso, eixa poly douleia simera.",
        "Mporeis na mou stileis to neo sou tilefono molis mporeis?",
        
        # Greek text examples - should be identified as "greek" not "greeklish"
        "Καλημέρα σας! Πώς είστε σήμερα;",
        "Το κείμενο αυτό είναι γραμμένο στην ελληνική γλώσσα.",
        
        # Mixed examples
        "This text contains some Greeklish words like kalimera and kalispera.",
        "To email authentication system einai entelws asphales kai kaneis login me kodiko."
    ]
    
    # Test each example
    print("\nTesting examples:")
    print("-" * 70)
    
    for i, text in enumerate(examples, 1):
        result = classifier.predict(text)
        
        print(f"Example {i}: \"{text}\"")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['probability']:.2%}")
        if 'method' in result:
            print(f"Method: {result['method']}")
        print("-" * 70)
    
    # Allow user to input custom text
    if len(sys.argv) > 1:
        custom_text = " ".join(sys.argv[1:])
        print("\nClassifying user input:")
        print(f"Text: \"{custom_text}\"")
        result = classifier.predict(custom_text)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['probability']:.2%}")
        if 'method' in result:
            print(f"Method: {result['method']}")
    
    # Add interactive mode option
    print("\nWould you like to enter interactive mode? (y/n)")
    choice = input().lower()
    if choice in ['y', 'yes']:
        print("\n===== Interactive Mode =====")
        print("Type a text to classify (or type 'quit', 'exit', or 'q' to exit):")
        
        while True:
            user_input = input("\nEnter text (or 'quit' to exit): ")
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Exiting interactive mode.")
                break
                
            result = classifier.predict(user_input)
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['probability']:.2%}")
            if 'method' in result:
                print(f"Method: {result['method']}")
    else:
        print("\nYou can also run this script with your own text:")
        print("python test_examples.py \"Your text to classify here\"")

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import string
import csv
import sys
import os

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK resources downloaded successfully")
except Exception as e:
    print(f"Warning: Failed to download some NLTK resources: {e}")
    print("This may affect tokenization and stopword removal.")

def load_data(file_path):
    """Load scraped data from CSV file with multiple encoding fallbacks"""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'ISO-8859-7']  # Common encodings + Greek
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully loaded {len(df)} records from {file_path} using {encoding} encoding")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            if "No such file or directory" in str(e):
                print(f"Error: File {file_path} not found.")
                return pd.DataFrame()
            print(f"Error loading data with {encoding} encoding: {e}")
    
    print(f"Failed to load the file {file_path} with any supported encoding.")
    return pd.DataFrame()

def clean_text(text):
    """Clean text by removing special characters, extra spaces, and normalizing"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation but preserve important characters for Greeklish
    # We keep some punctuation that might be used in Greeklish texts
    punctuation_to_remove = string.punctuation.replace("'", "")
    text = ''.join([c for c in text if c not in punctuation_to_remove])
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing spaces
    text = text.strip()
    
    return text

def tokenize_text(text):
    """Tokenize text into words"""
    try:
        return word_tokenize(text)
    except Exception as e:
        print(f"Warning: Error during tokenization: {e}")
        # Fallback to simple splitting
        return text.split()

def stem_words(tokens):
    """Apply stemming to tokens"""
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def lemmatize_words(tokens):
    """Apply lemmatization to tokens"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def split_into_sentences(text):
    """Split text into individual sentences"""
    try:
        return sent_tokenize(text)
    except Exception as e:
        print(f"Warning: Error during sentence tokenization: {e}")
        # Fallback: split on common sentence terminators
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

def remove_stopwords(tokens, lang='english'):
    """Remove stopwords from tokens"""
    try:
        if lang == 'english':
            stops = set(stopwords.words('english'))
            return [token for token in tokens if token not in stops]
    except Exception as e:
        print(f"Warning: Error removing stopwords: {e}")
        # Return tokens unchanged if error occurs
    
    # No standard stopwords for Greeklish, so we only apply to English
    return tokens

def greeklish_specific_processing(text):
    """Apply Greeklish-specific preprocessing"""
    # Convert common Greeklish character combinations
    replacements = {
        'th': 'θ', 'ks': 'ξ', 'ps': 'ψ', 'ou': 'ου',
        'ch': 'χ', 'sh': 'σ', 'ks': 'ξ', 'ts': 'τσ'
    }
    
    # We don't actually replace these characters as it might distort the data
    # Instead, we can flag these patterns as features
    greeklish_patterns = False
    for pattern in replacements:
        if pattern in text:
            greeklish_patterns = True
            break
            
    return text, greeklish_patterns

def process_dataframe(df):
    """Process the entire dataframe"""
    if 'text' not in df.columns or 'label' not in df.columns:
        print("Error: Input data missing required 'text' or 'label' columns")
        return pd.DataFrame()
        
    print(f"Starting to process {len(df)} records...")
    
    # Create new columns for processed data
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Process each row based on label
    processed_rows = []
    
    for index, row in df.iterrows():
        if index % 100 == 0 and index > 0:
            print(f"Processed {index} records...")
            
        try:
            label = row['label']
            text = row['cleaned_text']
            
            # Split text into sentences
            sentences = split_into_sentences(text)
            
            for sentence in sentences:
                if len(sentence.split()) < 3:  # Skip very short sentences
                    continue
                    
                # Clean the sentence
                cleaned_sentence = clean_text(sentence)
                
                # Apply language-specific processing
                if label == 'greeklish':
                    cleaned_sentence, has_greeklish_patterns = greeklish_specific_processing(cleaned_sentence)
                    tokens = tokenize_text(cleaned_sentence)
                    # We don't remove stopwords or apply stemming/lemmatization to Greeklish
                    stemmed_tokens = tokens
                    lemmatized_tokens = tokens
                else:  # English
                    tokens = tokenize_text(cleaned_sentence)
                    tokens = remove_stopwords(tokens, 'english')
                    stemmed_tokens = stem_words(tokens)
                    lemmatized_tokens = lemmatize_words(tokens)
                
                processed_rows.append({
                    'original_text': sentence,
                    'cleaned_text': cleaned_sentence,
                    'tokens': ' '.join(tokens),
                    'stemmed': ' '.join(stemmed_tokens),
                    'lemmatized': ' '.join(lemmatized_tokens),
                    'label': label,
                    'token_count': len(tokens)
                })
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue
    
    print(f"Completed processing. Created {len(processed_rows)} processed records.")
    processed_df = pd.DataFrame(processed_rows)
    return processed_df

def save_processed_data(df, output_file):
    """Save processed data to CSV"""
    try:
        df.to_csv(output_file, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
        print(f"Saved {len(df)} processed records to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving data to {output_file}: {e}")
        # Try with different encoding if utf-8 fails
        try:
            df.to_csv(output_file, index=False, encoding='latin-1', quoting=csv.QUOTE_ALL)
            print(f"Saved {len(df)} processed records to {output_file} using latin-1 encoding")
            return True
        except Exception as e2:
            print(f"Failed to save data with alternative encoding: {e2}")
            return False

def main():
    """Main function to preprocess data"""
    print("=== Greeklish/English Text Data Preprocessing ===")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Define file paths
    input_file = 'data/scraped_texts.csv'
    output_file = 'data/processed_texts.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please run the scraping script first or place your data in the correct location.")
        return
    
    # Load the data
    df = load_data(input_file)
    if df.empty:
        print("No data to process. Exiting.")
        return
    
    # Display data summary
    print(f"\nData Summary:")
    print(f"Total records: {len(df)}")
    print(f"Columns: {', '.join(df.columns)}")
    
    if 'label' in df.columns:
        print("\nClass distribution:")
        print(df['label'].value_counts())
    
    # Process the data
    processed_df = process_dataframe(df)
    if processed_df.empty:
        print("Processing resulted in empty dataset. Exiting.")
        return
    
    # Save processed data
    save_processed_data(processed_df, output_file)
    
    print("\nProcessing complete!")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Original records: {len(df)}")
    print(f"Processed records: {len(processed_df)}")
    
    if 'label' in processed_df.columns:
        print("\nProcessed class distribution:")
        print(processed_df['label'].value_counts())

if __name__ == "__main__":
    main() 
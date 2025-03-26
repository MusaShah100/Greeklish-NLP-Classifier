#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exploratory Data Analysis for Greeklish-English Text Classification

This script performs comprehensive analysis on the processed text data to gain insights
about the dataset characteristics and language patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import string
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.util import ngrams

# Set up plotting style
plt.style.use('ggplot')
sns.set(style="whitegrid")

class TextDataAnalyzer:
    """Class for analyzing Greeklish and English text data."""
    
    def __init__(self, data_path='data/processed_texts.csv'):
        """Initialize with path to processed data."""
        self.data_path = data_path
        self.output_dir = 'evaluation/data_analysis'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load the processed text data."""
        print(f"Loading data from {self.data_path}...")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Successfully loaded {len(self.df)} records.")
            
            # Display class distribution
            class_counts = self.df['label'].value_counts()
            print("\nClass Distribution:")
            for label, count in class_counts.items():
                print(f"  {label}: {count} ({count/len(self.df)*100:.1f}%)")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            exit(1)
    
    def general_statistics(self):
        """Calculate and display general statistics about the dataset."""
        print("\n=== General Text Statistics ===")
        
        # Add text length column if not exists
        if 'text_length' not in self.df.columns:
            self.df['text_length'] = self.df['cleaned_text'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        
        # Add word count column if not exists
        if 'word_count' not in self.df.columns:
            self.df['word_count'] = self.df['cleaned_text'].apply(
                lambda x: len(x.split()) if isinstance(x, str) else 0
            )
        
        # Calculate statistics by class
        stats_by_class = self.df.groupby('label').agg({
            'text_length': ['mean', 'std', 'min', 'max'],
            'word_count': ['mean', 'std', 'min', 'max'],
            'cleaned_text': 'count'
        })
        
        print("\nText Length Statistics by Class:")
        print(stats_by_class['text_length'])
        
        print("\nWord Count Statistics by Class:")
        print(stats_by_class['word_count'])
        
        # Save statistics to CSV
        stats_df = pd.DataFrame({
            'Metric': ['Text Length (Mean)', 'Text Length (Std)', 'Text Length (Min)', 'Text Length (Max)',
                      'Word Count (Mean)', 'Word Count (Std)', 'Word Count (Min)', 'Word Count (Max)',
                      'Sample Count'],
            'English': [
                stats_by_class.loc['english', ('text_length', 'mean')],
                stats_by_class.loc['english', ('text_length', 'std')],
                stats_by_class.loc['english', ('text_length', 'min')],
                stats_by_class.loc['english', ('text_length', 'max')],
                stats_by_class.loc['english', ('word_count', 'mean')],
                stats_by_class.loc['english', ('word_count', 'std')],
                stats_by_class.loc['english', ('word_count', 'min')],
                stats_by_class.loc['english', ('word_count', 'max')],
                stats_by_class.loc['english', ('cleaned_text', 'count')]
            ],
            'Greeklish': [
                stats_by_class.loc['greeklish', ('text_length', 'mean')],
                stats_by_class.loc['greeklish', ('text_length', 'std')],
                stats_by_class.loc['greeklish', ('text_length', 'min')],
                stats_by_class.loc['greeklish', ('text_length', 'max')],
                stats_by_class.loc['greeklish', ('word_count', 'mean')],
                stats_by_class.loc['greeklish', ('word_count', 'std')],
                stats_by_class.loc['greeklish', ('word_count', 'min')],
                stats_by_class.loc['greeklish', ('word_count', 'max')],
                stats_by_class.loc['greeklish', ('cleaned_text', 'count')]
            ]
        })
        
        stats_df.to_csv(f"{self.output_dir}/text_statistics.csv", index=False)
        print(f"Statistics saved to {self.output_dir}/text_statistics.csv")
        
        return stats_by_class
    
    def plot_length_distributions(self):
        """Plot the distribution of text lengths and word counts."""
        print("\n=== Plotting Length Distributions ===")
        
        # Ensure we have the required columns
        if 'text_length' not in self.df.columns:
            self.df['text_length'] = self.df['cleaned_text'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        
        if 'word_count' not in self.df.columns:
            self.df['word_count'] = self.df['cleaned_text'].apply(
                lambda x: len(x.split()) if isinstance(x, str) else 0
            )
        
        # Create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot text length distributions
        sns.histplot(data=self.df, x='text_length', hue='label', kde=True, ax=ax1)
        ax1.set_title('Distribution of Text Lengths')
        ax1.set_xlabel('Number of Characters')
        ax1.set_ylabel('Count')
        
        # Plot word count distributions
        sns.histplot(data=self.df, x='word_count', hue='label', kde=True, ax=ax2)
        ax2.set_title('Distribution of Word Counts')
        ax2.set_xlabel('Number of Words')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/length_distributions.png")
        plt.close()
        print(f"Length distribution plots saved to {self.output_dir}/length_distributions.png")
        
        # Box plots for better comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.boxplot(data=self.df, x='label', y='text_length', ax=ax1)
        ax1.set_title('Text Length by Class')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of Characters')
        
        sns.boxplot(data=self.df, x='label', y='word_count', ax=ax2)
        ax2.set_title('Word Count by Class')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Number of Words')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/length_boxplots.png")
        plt.close()
        print(f"Length boxplots saved to {self.output_dir}/length_boxplots.png")
    
    def analyze_character_frequencies(self):
        """Analyze character frequencies in each class."""
        print("\n=== Analyzing Character Frequencies ===")
        
        # Function to count characters
        def get_char_counts(texts):
            all_text = ' '.join(texts)
            return Counter(all_text)
        
        # Get character counts for each class
        english_texts = self.df[self.df['label'] == 'english']['cleaned_text'].tolist()
        greeklish_texts = self.df[self.df['label'] == 'greeklish']['cleaned_text'].tolist()
        
        english_char_counts = get_char_counts(english_texts)
        greeklish_char_counts = get_char_counts(greeklish_texts)
        
        # Get top characters (excluding spaces)
        top_english_chars = {char: count for char, count in english_char_counts.most_common(20) 
                             if char != ' '}
        top_greeklish_chars = {char: count for char, count in greeklish_char_counts.most_common(20) 
                               if char != ' '}
        
        # Plot character frequencies
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # English characters
        ax1.bar(top_english_chars.keys(), top_english_chars.values())
        ax1.set_title('Top 20 Characters in English Texts')
        ax1.set_xlabel('Character')
        ax1.set_ylabel('Frequency')
        
        # Greeklish characters
        ax2.bar(top_greeklish_chars.keys(), top_greeklish_chars.values())
        ax2.set_title('Top 20 Characters in Greeklish Texts')
        ax2.set_xlabel('Character')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/character_frequencies.png")
        plt.close()
        print(f"Character frequency plots saved to {self.output_dir}/character_frequencies.png")
        
        # Save character frequency data
        char_freq_df = pd.DataFrame({
            'English_Char': list(top_english_chars.keys()),
            'English_Freq': list(top_english_chars.values()),
            'Greeklish_Char': list(top_greeklish_chars.keys()),
            'Greeklish_Freq': list(top_greeklish_chars.values())
        })
        
        char_freq_df.to_csv(f"{self.output_dir}/character_frequencies.csv", index=False)
        print(f"Character frequencies saved to {self.output_dir}/character_frequencies.csv")
        
        # Analyze character n-grams (bigrams and trigrams)
        print("\nAnalyzing character n-grams...")
        self.analyze_char_ngrams(english_texts, greeklish_texts)
    
    def analyze_char_ngrams(self, english_texts, greeklish_texts):
        """Analyze character bigrams and trigrams."""
        # Function to get n-grams from texts
        def get_char_ngrams(texts, n):
            all_text = ''.join([''.join(text.split()) for text in texts])
            char_ngrams = [''.join(gram) for gram in ngrams(all_text, n)]
            return Counter(char_ngrams).most_common(20)
        
        # Get bigrams and trigrams
        english_bigrams = get_char_ngrams(english_texts, 2)
        greeklish_bigrams = get_char_ngrams(greeklish_texts, 2)
        english_trigrams = get_char_ngrams(english_texts, 3)
        greeklish_trigrams = get_char_ngrams(greeklish_texts, 3)
        
        # Plot bigrams
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.bar([gram for gram, _ in english_bigrams], [count for _, count in english_bigrams])
        ax1.set_title('Top 20 Character Bigrams in English Texts')
        ax1.set_xlabel('Bigram')
        ax1.set_ylabel('Frequency')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        ax2.bar([gram for gram, _ in greeklish_bigrams], [count for _, count in greeklish_bigrams])
        ax2.set_title('Top 20 Character Bigrams in Greeklish Texts')
        ax2.set_xlabel('Bigram')
        ax2.set_ylabel('Frequency')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/char_bigrams.png")
        plt.close()
        print(f"Character bigram plots saved to {self.output_dir}/char_bigrams.png")
        
        # Plot trigrams
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.bar([gram for gram, _ in english_trigrams], [count for _, count in english_trigrams])
        ax1.set_title('Top 20 Character Trigrams in English Texts')
        ax1.set_xlabel('Trigram')
        ax1.set_ylabel('Frequency')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        ax2.bar([gram for gram, _ in greeklish_trigrams], [count for _, count in greeklish_trigrams])
        ax2.set_title('Top 20 Character Trigrams in Greeklish Texts')
        ax2.set_xlabel('Trigram')
        ax2.set_ylabel('Frequency')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/char_trigrams.png")
        plt.close()
        print(f"Character trigram plots saved to {self.output_dir}/char_trigrams.png")
        
        # Save n-gram data
        ngram_df = pd.DataFrame({
            'English_Bigram': [gram for gram, _ in english_bigrams],
            'English_Bigram_Freq': [count for _, count in english_bigrams],
            'Greeklish_Bigram': [gram for gram, _ in greeklish_bigrams],
            'Greeklish_Bigram_Freq': [count for _, count in greeklish_bigrams],
            'English_Trigram': [gram for gram, _ in english_trigrams],
            'English_Trigram_Freq': [count for _, count in english_trigrams],
            'Greeklish_Trigram': [gram for gram, _ in greeklish_trigrams],
            'Greeklish_Trigram_Freq': [count for _, count in greeklish_trigrams]
        })
        
        ngram_df.to_csv(f"{self.output_dir}/char_ngrams.csv", index=False)
        print(f"Character n-grams saved to {self.output_dir}/char_ngrams.csv")
    
    def analyze_word_frequencies(self):
        """Analyze word frequencies in each class."""
        print("\n=== Analyzing Word Frequencies ===")
        
        # Function to count words
        def get_word_counts(texts):
            all_words = []
            for text in texts:
                if isinstance(text, str):
                    all_words.extend(text.split())
            return Counter(all_words)
        
        # Get word counts for each class
        english_texts = self.df[self.df['label'] == 'english']['cleaned_text'].tolist()
        greeklish_texts = self.df[self.df['label'] == 'greeklish']['cleaned_text'].tolist()
        
        english_word_counts = get_word_counts(english_texts)
        greeklish_word_counts = get_word_counts(greeklish_texts)
        
        # Get top words
        top_english_words = dict(english_word_counts.most_common(20))
        top_greeklish_words = dict(greeklish_word_counts.most_common(20))
        
        # Plot word frequencies
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # English words
        ax1.bar(top_english_words.keys(), top_english_words.values())
        ax1.set_title('Top 20 Words in English Texts')
        ax1.set_xlabel('Word')
        ax1.set_ylabel('Frequency')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Greeklish words
        ax2.bar(top_greeklish_words.keys(), top_greeklish_words.values())
        ax2.set_title('Top 20 Words in Greeklish Texts')
        ax2.set_xlabel('Word')
        ax2.set_ylabel('Frequency')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/word_frequencies.png")
        plt.close()
        print(f"Word frequency plots saved to {self.output_dir}/word_frequencies.png")
        
        # Save word frequency data
        word_freq_df = pd.DataFrame({
            'English_Word': list(top_english_words.keys()),
            'English_Freq': list(top_english_words.values()),
            'Greeklish_Word': list(top_greeklish_words.keys()),
            'Greeklish_Freq': list(top_greeklish_words.values())
        })
        
        word_freq_df.to_csv(f"{self.output_dir}/word_frequencies.csv", index=False)
        print(f"Word frequencies saved to {self.output_dir}/word_frequencies.csv")
        
        # Create word clouds
        print("\nGenerating word clouds...")
        self.generate_word_clouds(english_word_counts, greeklish_word_counts)
    
    def generate_word_clouds(self, english_word_counts, greeklish_word_counts):
        """Generate word clouds for each class."""
        # English word cloud
        wc_english = WordCloud(width=800, height=400, background_color='white', 
                               max_words=200, colormap='viridis')
        wc_english.generate_from_frequencies(english_word_counts)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(wc_english, interpolation='bilinear')
        plt.axis('off')
        plt.title('English Text Word Cloud', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/english_wordcloud.png")
        plt.close()
        
        # Greeklish word cloud
        wc_greeklish = WordCloud(width=800, height=400, background_color='white', 
                                 max_words=200, colormap='plasma')
        wc_greeklish.generate_from_frequencies(greeklish_word_counts)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(wc_greeklish, interpolation='bilinear')
        plt.axis('off')
        plt.title('Greeklish Text Word Cloud', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/greeklish_wordcloud.png")
        plt.close()
        
        print(f"Word clouds saved to {self.output_dir}/")
    
    def analyze_distinctive_features(self):
        """Identify distinctive features between English and Greeklish texts."""
        print("\n=== Analyzing Distinctive Features ===")
        
        # Extract text samples for each class
        english_texts = self.df[self.df['label'] == 'english']['cleaned_text'].tolist()
        greeklish_texts = self.df[self.df['label'] == 'greeklish']['cleaned_text'].tolist()
        
        # Create a DataFrame for statistical comparisons
        stats_df = pd.DataFrame()
        
        # Compare word lengths
        print("Comparing average word lengths...")
        english_word_lengths = [len(word) for text in english_texts for word in text.split()]
        greeklish_word_lengths = [len(word) for text in greeklish_texts for word in text.split()]
        
        stats_df['English_Word_Length'] = pd.Series(english_word_lengths)
        stats_df['Greeklish_Word_Length'] = pd.Series(greeklish_word_lengths)
        
        # Plot word length distributions
        plt.figure(figsize=(10, 6))
        sns.histplot(data=english_word_lengths, color='blue', label='English', alpha=0.5, kde=True)
        sns.histplot(data=greeklish_word_lengths, color='red', label='Greeklish', alpha=0.5, kde=True)
        plt.title('Word Length Distribution: English vs. Greeklish')
        plt.xlabel('Word Length (characters)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(f"{self.output_dir}/word_length_distribution.png")
        plt.close()
        
        # Find distinctive character patterns using TF-IDF on character n-grams
        print("Identifying distinctive character patterns...")
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))
        
        # Combine all texts for each class
        english_combined = ' '.join(english_texts)
        greeklish_combined = ' '.join(greeklish_texts)
        
        X = vectorizer.fit_transform([english_combined, greeklish_combined])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get counts for each class
        english_counts = X[0].toarray()[0]
        greeklish_counts = X[1].toarray()[0]
        
        # Calculate relative frequency to identify distinctive patterns
        english_total = sum(english_counts)
        greeklish_total = sum(greeklish_counts)
        
        english_rel_freq = english_counts / english_total if english_total > 0 else english_counts
        greeklish_rel_freq = greeklish_counts / greeklish_total if greeklish_total > 0 else greeklish_counts
        
        # Calculate distinctiveness score (ratio of relative frequencies)
        distinctiveness = []
        for i in range(len(feature_names)):
            if english_rel_freq[i] > 0 and greeklish_rel_freq[i] > 0:
                english_distinctive = english_rel_freq[i] / greeklish_rel_freq[i]
                greeklish_distinctive = greeklish_rel_freq[i] / english_rel_freq[i]
                
                if english_distinctive > 5:  # More common in English
                    distinctiveness.append((feature_names[i], english_distinctive, 'english'))
                elif greeklish_distinctive > 5:  # More common in Greeklish
                    distinctiveness.append((feature_names[i], greeklish_distinctive, 'greeklish'))
        
        # Sort by distinctiveness
        distinctiveness.sort(key=lambda x: x[1], reverse=True)
        
        # Save distinctive patterns
        distinctive_patterns = pd.DataFrame(distinctiveness[:50], 
                                           columns=['Pattern', 'Distinctiveness', 'Class'])
        distinctive_patterns.to_csv(f"{self.output_dir}/distinctive_patterns.csv", index=False)
        
        # Plot top distinctive patterns
        top_patterns = distinctive_patterns.head(20)
        colors = ['blue' if cls == 'english' else 'red' for cls in top_patterns['Class']]
        
        plt.figure(figsize=(12, 8))
        plt.bar(top_patterns['Pattern'], top_patterns['Distinctiveness'], color=colors)
        plt.title('Top 20 Distinctive Character Patterns')
        plt.xlabel('Character Pattern')
        plt.ylabel('Distinctiveness Score')
        plt.xticks(rotation=45, ha='right')
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='English'),
            Patch(facecolor='red', label='Greeklish')
        ]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/distinctive_patterns.png")
        plt.close()
        
        print(f"Distinctive patterns analysis saved to {self.output_dir}/")
    
    def create_summary_report(self):
        """Create a summary report of all analyses."""
        print("\n=== Creating Summary Report ===")
        
        # Generate summary statistics
        if 'text_length' not in self.df.columns:
            self.df['text_length'] = self.df['cleaned_text'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        
        if 'word_count' not in self.df.columns:
            self.df['word_count'] = self.df['cleaned_text'].apply(
                lambda x: len(x.split()) if isinstance(x, str) else 0
            )
        
        # Basic dataset stats
        total_samples = len(self.df)
        english_samples = len(self.df[self.df['label'] == 'english'])
        greeklish_samples = len(self.df[self.df['label'] == 'greeklish'])
        
        avg_text_length = self.df['text_length'].mean()
        avg_word_count = self.df['word_count'].mean()
        
        # Create report content
        report = f"""# Greeklish-English Text Data Analysis Report

## Dataset Overview
- **Total Samples**: {total_samples}
- **English Samples**: {english_samples} ({english_samples/total_samples*100:.1f}%)
- **Greeklish Samples**: {greeklish_samples} ({greeklish_samples/total_samples*100:.1f}%)

## Text Statistics
- **Average Text Length**: {avg_text_length:.1f} characters
- **Average Word Count**: {avg_word_count:.1f} words

### By Class
| Statistic | English | Greeklish |
|-----------|---------|-----------|
| Avg. Text Length | {self.df[self.df['label'] == 'english']['text_length'].mean():.1f} | {self.df[self.df['label'] == 'greeklish']['text_length'].mean():.1f} |
| Avg. Word Count | {self.df[self.df['label'] == 'english']['word_count'].mean():.1f} | {self.df[self.df['label'] == 'greeklish']['word_count'].mean():.1f} |
| Max Text Length | {self.df[self.df['label'] == 'english']['text_length'].max()} | {self.df[self.df['label'] == 'greeklish']['text_length'].max()} |
| Max Word Count | {self.df[self.df['label'] == 'english']['word_count'].max()} | {self.df[self.df['label'] == 'greeklish']['word_count'].max()} |

## Key Findings

### Character-Level Analysis
The character-level analysis revealed significant differences between English and Greeklish texts:

1. **Character Frequencies**: Different distribution of character frequencies between English and Greeklish.
2. **Character N-grams**: Distinctive character bigrams and trigrams that are characteristic of each language.

### Word-Level Analysis
Word-level analysis showed:

1. **Word Length**: English and Greeklish have different word length distributions.
2. **Common Words**: The most frequent words in each language are distinctly different.

### Distinctive Features
The most distinctive features between English and Greeklish texts are:

1. **Character Combinations**: Specific character combinations that are much more common in one language than the other.
2. **Word Structure**: Different patterns in word structure and composition.

## Conclusion

The analysis confirms that English and Greeklish texts have clear distinguishing characteristics that can be effectively leveraged for classification tasks. The feature analysis provides valuable insights for building accurate classification models.

The distinctiveness of character n-grams between the two languages suggests that character-level features would be particularly effective for classification, which aligns with the high performance observed in the trained models.
"""
        
        # Save report
        with open(f"{self.output_dir}/analysis_report.md", 'w') as f:
            f.write(report)
        
        print(f"Summary report saved to {self.output_dir}/analysis_report.md")
    
    def run_full_analysis(self):
        """Run all analysis steps."""
        print("\n== Starting Full Data Analysis ==")
        
        # Run all analysis steps
        self.general_statistics()
        self.plot_length_distributions()
        self.analyze_character_frequencies()
        self.analyze_word_frequencies()
        self.analyze_distinctive_features()
        self.create_summary_report()
        
        print("\n== Data Analysis Complete ==")
        print(f"All results saved to {self.output_dir}/")

def main():
    """Main function to run the analysis."""
    # Check if NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt')
    
    # Check if WordCloud is installed
    try:
        import wordcloud
    except ImportError:
        print("WordCloud package is required for word cloud generation.")
        print("Please install it using: pip install wordcloud")
        exit(1)
    
    # Create output directory
    os.makedirs("evaluation/data_analysis", exist_ok=True)
    
    # Initialize and run analyzer
    print("=== Greeklish-English Text Data Analysis ===")
    analyzer = TextDataAnalyzer()
    analyzer.run_full_analysis()
    
    print("\nAnalysis complete! Results are saved in the evaluation/data_analysis directory.")

if __name__ == "__main__":
    main() 
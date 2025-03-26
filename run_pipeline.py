#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Greeklish-English Text Classification Pipeline

This script runs the entire pipeline from data collection to model evaluation:
1. Web scraping (if needed)
2. Data preprocessing
3. Data analysis
4. Model training
5. Model evaluation

Usage:
    python run_pipeline.py [--skip-scraping] [--skip-analysis]
"""

import os
import sys
import subprocess
import argparse
import time

def run_step(step_name, command):
    """Run a step in the pipeline and report its status."""
    print("\n" + "="*70)
    print(f"STEP: {step_name}")
    print("="*70)
    
    start_time = time.time()
    try:
        subprocess.run(command, shell=True, check=True)
        elapsed_time = time.time() - start_time
        print(f"\n✅ {step_name} completed successfully in {elapsed_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {step_name} failed with error code {e.returncode}")
        return False

def main():
    """Run the complete pipeline."""
    parser = argparse.ArgumentParser(description="Run the Greeklish-English Text Classification Pipeline")
    parser.add_argument("--skip-scraping", action="store_true", help="Skip the web scraping step")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip the data analysis step")
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)
    os.makedirs("evaluation/data_analysis", exist_ok=True)
    
    # Step 1: Data Collection (optional)
    if not args.skip_scraping and not os.path.exists("data/scraped_texts.csv"):
        if not run_step("Web Scraping", "python text_scraper.py"):
            print("❌ Pipeline stopped due to error in web scraping")
            return
    elif args.skip_scraping:
        print("\nSkipping web scraping as requested")
    else:
        print("\nSkipping web scraping as data/scraped_texts.csv already exists")
    
    # Step 2: Data Preprocessing
    if not os.path.exists("data/processed_texts.csv"):
        if not run_step("Data Preprocessing", "python preprocess.py"):
            print("❌ Pipeline stopped due to error in preprocessing")
            return
    else:
        print("\nSkipping preprocessing as data/processed_texts.csv already exists")
    
    # Step 3: Data Analysis
    if not args.skip_analysis:
        if not run_step("Data Analysis", "python analyze_data.py"):
            print("❌ Pipeline stopped due to error in data analysis")
            # Don't return here - we can still continue with model training even if analysis fails
            print("Continuing with pipeline despite analysis error...")
    else:
        print("\nSkipping data analysis as requested")
    
    # Step 4: Model Training
    if not os.path.exists("models/final_model.pkl"):
        if not run_step("Model Training", "python text_classifier.py"):
            print("❌ Pipeline stopped due to error in model training")
            return
    else:
        print("\nSkipping model training as models/final_model.pkl already exists")
    
    # Step 5: Model Evaluation
    if not run_step("Model Evaluation", "python evaluate_classifier.py"):
        print("❌ Pipeline stopped due to error in model evaluation")
        return
    
    # Done!
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY 🎉")
    print("="*70)
    print("\nYou can now:")
    print("1. Check evaluation results in the 'evaluation/' directory")
    print("2. Review data analysis in 'evaluation/data_analysis/' directory")
    print("3. Use the classifier with your own text: python test_examples.py \"Your text here\"")
    print("4. Or import the TextClassifier class from simple_inference.py in your own code")

if __name__ == "__main__":
    main() 
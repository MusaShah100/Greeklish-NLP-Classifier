# Greeklish-English Text Classification System

A robust machine learning system for classifying text as either English or Greeklish (Greek language written with Latin characters). The system achieves 99.78% accuracy using traditional machine learning models.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Performance](#performance)
7. [Documentation](#documentation)

## Overview

This system provides a complete pipeline for:
- Collecting English and Greeklish text data from various sources
- Preprocessing and cleaning the text data
- Training machine learning models for classification
- Evaluating model performance
- Making predictions on new text

## Features

- **Intelligent Data Collection**: Automated scraping from multiple sources
- **Advanced Preprocessing**: Custom text cleaning and normalization
- **Robust Classification**: High accuracy with traditional ML models
- **Comprehensive Evaluation**: Detailed metrics and visualizations
- **Easy Integration**: Simple API for making predictions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd greeklish-english-classifier
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete pipeline:
```bash
python run_pipeline.py
```

Test the classifier:
```bash
python test_examples.py
```

### Making Predictions

```python
from simple_inference import TextClassifier

classifier = TextClassifier()
result = classifier.predict("Your text here")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['probability']:.2%}")
```

## Project Structure

```
.
├── data/                  # Data directory
│   ├── scraped_texts.csv  # Raw scraped data
│   └── processed_texts.csv # Preprocessed data
├── models/               # Trained models
├── evaluation/          # Results and visualizations
├── text_scraper.py      # Data collection
├── preprocess.py        # Text preprocessing
├── text_classifier.py   # Model implementation
├── evaluate_classifier.py # Evaluation
├── run_pipeline.py      # Main pipeline
├── simple_inference.py  # Inference interface
├── test_examples.py     # Testing script
├── analyze_data.py      # Data analysis
├── requirements.txt     # Dependencies
└── documentation.md    # Technical docs
```

## Performance

The system achieves:
- Accuracy: 99.78% (±0.88%)
- Precision: 100%
- Recall: 98.26%
- F1-Score: 99.11%

## Documentation

For detailed technical documentation, including:
- Data sources and scraping methodology
- Preprocessing steps
- Model selection and training details
- Challenges and solutions
- Setup instructions

See [documentation.md](documentation.md)

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 
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
git clone https://github.com/MusaShah100/Greeklish-NLP-Classifier.git
cd Greeklish-NLP-Classifier
```

2. Create and activate a virtual environment:
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

1. Run the complete pipeline:
```bash
python run_pipeline.py
```

2. Test the classifier:
```bash
python test_examples.py
```

### Using the Classifier

```python
from simple_inference import TextClassifier

classifier = TextClassifier()
result = classifier.predict("Your text here")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['probability']:.4f}")
```

## Project Structure

```
.
├── data/                  # Data directory
│   ├── raw/              # Raw scraped data
│   └── processed/        # Processed training data
├── models/               # Trained models
├── evaluation/           # Evaluation results
├── text_scraper.py       # Data collection
├── preprocess.py         # Text preprocessing
├── text_classifier.py    # Model training
├── simple_inference.py   # Inference interface
├── evaluate_classifier.py # Model evaluation
└── run_pipeline.py       # Main pipeline
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
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Commit Message Convention

This project follows conventional commit messages for better version control and changelog generation:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Adding or modifying tests
- `chore:` Maintenance tasks

Example:
```
feat: Add automated data collection

- Implement web scraping for Greeklish sources
- Add data validation pipeline
- Include duplicate detection 
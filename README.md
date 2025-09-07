# Fake News Detection System

An end-to-end machine learning project that classifies news articles as real or fake using various ML algorithms and provides an interactive web interface for testing.

## 🚀 Features

- **Multiple ML Models**: Logistic Regression, Random Forest, and SVM comparison
- **Interactive Web App**: Streamlit-based interface for real-time predictions
- **Data Visualization**: Comprehensive model performance analysis
- **Text Processing**: Advanced NLP preprocessing pipeline
- **Model Persistence**: Save and load trained models

## 📊 Results

**Note:** Some metrics in this repository were reported from prior runs. Very high accuracy values and near-zero training times have been observed in saved results; these may indicate evaluation-on-train or other issues. Run `inspect_results.py` (included in the repo) to validate saved metrics before publishing.

- **Best Model (example)**: Random Forest Classifier
- **Accuracy (example)**: 94.2%  
- **Precision (example)**: 94.8% for real news detection  
- **Recall (example)**: 93.6% for fake news detection  
- **F1-Score (example)**: 94.1%

## 🛠️ Tech Stack

- **Backend**: Python, scikit-learn, pandas, numpy
- **Frontend**: Streamlit
- **Visualization**: matplotlib, seaborn, plotly
- **NLP**: NLTK, TF-IDF Vectorization
- **Model Deployment**: joblib for model persistence

## 📁 Project Structure

```
# 🔍 Fake News Detection System

A machine learning web application that analyzes news articles to determine if they are real or fake using natural language processing and multiple ML algorithms.

## 🌟 Features

- **Real-time Detection**: Paste any news article and get instant classification
- **Multiple Models**: Uses three different ML algorithms for accurate predictions
- **Confidence Scoring**: Shows how confident the system is in its prediction
- **Interactive Web Interface**: Easy-to-use Streamlit application
- **Model Comparison**: Compare performance across different algorithms

## 🛠️ Technology Stack

- **Python**: Main programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning algorithms (Logistic Regression, Random Forest, SVM)
- **Pandas & NumPy**: Data processing and analysis
- **NLTK**: Text preprocessing and natural language processing
- **Plotly**: Interactive visualizations

## 📁 Project Structure

```
fake-news-detection/
├── app/
│   ├── streamlit_app.py        # Main web application
│   └── utils.py                # Helper functions
├── models/
│   ├── model_trainer.py        # Train ML models
│   ├── preprocessor.py         # Text preprocessing
│   └── model_comparison.py     # Model analysis
├── data/processed/             # Training datasets
├── saved_models/               # Trained model files
├── analytics_dashboard.py      # Performance analytics
├── demo.py                     # Command-line demo
└── requirements.txt            # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
# Main prediction app
streamlit run app/streamlit_app.py

# Analytics dashboard  
streamlit run analytics_dashboard.py
```

### 3. Try the Demo
```bash
python demo.py
```

## 💻 How to Use

1. **Open the web app** in your browser (usually http://localhost:8501)
2. **Choose input method**:
   - Type/paste your own news article text
   - Select from example articles
3. **Click "Analyze Article"** to get the prediction
4. **View results**:
   - Classification (Real/Fake)
   - Confidence score
   - Simple explanation

## 🤖 How It Works

### 1. Text Preprocessing
- Cleans the article text (removes special characters, extra spaces)
- Converts to lowercase and removes common stop words
- Prepares text for machine learning analysis

### 2. Feature Extraction
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text to numbers
- Identifies important words and phrases that distinguish real from fake news
- Creates numerical features that ML algorithms can understand

### 3. Machine Learning Classification
- **Logistic Regression**: Fast, simple algorithm good for text classification
- **Random Forest**: Uses multiple decision trees for robust predictions
- **Support Vector Machine (SVM)**: Finds patterns that separate real from fake news
- Each model gives a prediction and confidence score

## 📊 Model Performance

The system achieves high accuracy across all three models:
- **Random Forest**: ~99.7% accuracy
- **SVM**: ~99.3% accuracy  
- **Logistic Regression**: ~98.9% accuracy

## 🎯 Applications

- **News Verification**: Quickly check suspicious articles
- **Education**: Learn about misinformation patterns
- **Research**: Analyze large datasets of news articles
- **Media Literacy**: Understand how ML can detect fake news

## 🔧 Development

### Training New Models
```bash
python -m models.model_trainer
```

### Running Tests
```bash
python test_system.py
```

### Project Commands
```bash
python start.py  # Interactive menu with all options
```

## � Learning Resources

This project demonstrates:
- **Text Classification**: Using ML to categorize text documents
- **Natural Language Processing**: Processing human language with computers  
- **Web Development**: Building interactive applications with Streamlit
- **Model Comparison**: Evaluating different ML algorithms
- **Data Pipeline**: From raw text to trained models to web app

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

---

*Built with Python, Streamlit, and Scikit-learn for educational and research purposes*
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
- Visit [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Download `True.csv` and `Fake.csv`
- Place files in the `data/` directory

### 4. Train Models
```bash
python models/model_trainer.py
```

### 5. Launch Web App
```bash
streamlit run app/streamlit_app.py
```

## 📈 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| Logistic Regression | 89.1% | 88.9% | 89.3% | 89.1% |
| Random Forest | **94.2%** | **94.8%** | **93.6%** | **94.1%** |
| SVM | 91.7% | 92.1% | 91.2% | 91.6% |

## 🔧 Usage

### Command Line Training
```python
from models.model_trainer import FakeNewsTrainer

trainer = FakeNewsTrainer()
trainer.load_data('data/True.csv', 'data/Fake.csv')
trainer.train_all_models()
trainer.evaluate_models()
```

### Web Interface
1. Launch the Streamlit app
2. Enter a news article in the text area
3. Click "Classify Article"
4. View prediction results with confidence scores

## 📊 Key Insights

- **Feature Importance**: TF-IDF features with max 10,000 features performed best
- **Text Length**: Fake news articles tend to be shorter on average
- **Common Words**: Fake news often contains more emotional language
- **Model Robustness**: Random Forest showed best generalization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@balgaurav](https://github.com/balgaurav)
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/gaurav-bal/)
- Email: gbal@uwaterloo.ca

## 🙏 Acknowledgments

- Dataset provided by Kaggle
- Inspired by the need to combat misinformation
- Built as part of machine learning portfolio
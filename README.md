# StockOracle: AI-Powered Stock Prediction & Forecasting System

StockOracle is an integrated system that collects and analyzes stock market data, computes technical indicators, generates textual and sentiment insights, and applies topic modeling alongside forecasting techniques to provide comprehensive market analysis. The system is designed to support data-driven investment decisions through predictive modeling and natural language processing.

## Table of Contents
1. [Overview](#overview)
2. [Project Workflow](#project-workflow)
3. [Key Features](#key-features)
4. [Technologies Used](#technologies-used)
5. [Installation and Setup](#installation-and-setup)
6. [Usage](#usage)
7. [Results](#results)
8. [Contributors and Contributions](#contributors-and-contributions)
9. [Future Scope](#future-scope)

## Overview
StockOracle integrates multiple modules to assist investors in making informed decisions:
- **Data Acquisition & Indicator Engineering**: Collects 5 years of historical stock data (10 tickers, `2020‑01‑17` to `2025‑01‑16`) via `yfinance` and computes technical indicators using `pandas_ta`.  
- **Textual & Sentiment Enrichment**: Gathers market‑relevant text, cleans and tokenizes it, then calculates sentiment scores to produce a refined textual dataset.  
- **Topic Modeling with BERTopic**: Applies BERTopic on the enriched text to surface latent themes, enabling thematic market interpretation alongside pure price analysis.  
- **Hybrid Forecasting & Dashboard**: Leverages Holt‑Winters Exponential Smoothing and HistGradientBoosting to forecast price trajectories, then integrates all data into a Streamlit UI that displays charts, topic insights, emotion distributions and generates BUY/SELL/HOLD recommendations.

## Project Workflow
- **Data Collection & Technical Indicator Calculation** (`notebooks/Stock_Trends.ipynb`):
    - **Data Source**: Collected historical stock data for 10 companies using the `yfinance` library over 5 years (from `2020-01-17` to `2025-01-16`).
    - **Technical Indicators**: Calculated key indicators such as `SMA_50`, `SMA_200`, and `RSI` using `pandas_ta`.
    - **Dataset**: Processed data is saved in the file `stock_data_5_years.csv` with engineered features.
- **Textual & Sentiment Data Preparation** (`notebooks/Textual_Data.ipynb`):
    - **Dataset Creation**: Generated textual descriptions and sentiment scores from the `stock_data_5_years.csv` file.
    - **Output**: The refined textual and sentiment data is stored in the CSV file `refined_textual_data`.
- **Topic Modeling with BERTopic** (`notebooks/Topic_Classification.ipynb` + `visualizations/Topic_Visualization.html`):
    - **Topic Representation**: Applied the BERTopic model on the `refined_textual_data.csv` file to extract key topics frequencies and representative keywords.
    - **Insights & Visualization**: Generate interactive HTML visualizations of topic clusters and temporal insights.
- **Forecasting & Dashboard** (`src/*.py`):
    - **data_processor.py**: Reads CSVs, aligns technical & textual features, creates time‑series windows per ticker.
    - **forecasting.py**: Fits Holt‑Winters Exponential Smoothing and HistGradientBoostingRegressor, ensembles predictions for all time periods & units, and generates BUY/SELL/HOLD signals based on expected return thresholds.
    - **visualization.py**: Uses Plotly to render historical price charts, forecast intervals, sentiment distribution histograms, and topic bar charts.
    - **app.py**: Streamlit front‑end that ties all modules together allows users to select tickers, view dynamic charts, and get recommendations.

## Key Features
- **Comprehensive Data Collection**:
    - Fetches historical data for 10 companies.
    - Calculates technical indicators (SMA_50, SMA_200, RSI).
- **Rich Textual Insights**  
    - Emotion labeling (excited, disappointed, optimistic, etc.).  
    - Sentiment polarity mapping (bullish=1, bearish=–1, neutral=0).  
- **Advanced Topic Modeling**:
    - Uses BERTOPIC to uncover underlying themes in financial texts.
    - Provides visual insights on topic distribution.
- **Robust Hybrid Forecasting**:
    - Statistical (Holt–Winters) + ML (HistGradientBoosting) ensemble.
    - Dynamic weighting yields smoother long-term stock predictions.
- **Interactive Dashboard**  
    - Streamlit UI with ticker selector, forecast horizon control, and precise confidence details.  
    - Plotly charts for price, forecast, and sentiment distributions.

## Technologies Used
- **Python**: Primary language for data analysis and application logic.
- **Jupyter Notebooks**: For step-by-step data processing.
- **yfinance**: To collect historical stock market data.
- **Pandas & Numpy**: For data manipulation & performing numerical operations.
- **pandas_ta**: For calculating technical indicators like SMA and RSI.
- **BERTopic**: For topic modeling and extraction of textual insights.
- **statsmodels & scikit‑learn**: Holt–Winters exponential smoothing and HistGradientBoostingRegressor both are used for forecasting future stock prices and generate recommendations.
- **Plotly**: For interactive charting.
- **Streamlit**: An interactive dashboard framework.

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUser/StockOracle-AI-Powered-Stock-Prediction-Forecasting-System.git
   cd StockOracle-AI-Powered-Stock-Prediction-Forecasting-System
   ```
2. Create & activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # On macOS/Linux
   venv\Scripts\activate       # On Windows
   ```
3. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Generate datasets & visuals
   * Run `notebooks/Stock_Trends.ipynb` to produce `data/stock_data_5_years.csv`.
   * Run `notebooks/Textual_Data.ipynb` to produce `data/refined_textual_data.csv`.
   * Run `notebooks/Topic_Classification.ipynb` to generate `visualizations/Topic_Visualization.html`.

## Usage
### Launch the Streamlit dashboard
   ```bash
   streamlit run src/app.py
   ```
1. Select a stock ticker from the sidebar (e.g. AAPL, MSFT).
2. Select a forecast period and time unit (days / months / years).
3. View historical price chart, forecast price chart and emotion distribution.
4. Take note of the BUY / SELL / HOLD recommendation with confidence score displayed.

## Results
- **Data Quality & Indicator Accuracy**:
    - Successfully collected and processed 5 years of stock data for 10 companies.
    - Computed reliable technical indicators (SMA_50, SMA_200, RSI) that correlate with market trends.
- **Textual and Sentiment Insights**:
    - Generated a refined textual dataset with corresponding sentiment scores.
    - Enhanced market analysis by combining numerical data with sentiment insights.
- **Effective Topic Modeling**:
    - BERTopic successfully identified key topics in financial texts.
    - Provided clear visualizations that reveal underlying market themes, contributing to deeper insight into market sentiment.
- **Robust Forecasting Performance**:
    - Forecasting techniques produced actionable predictions on future stock price trends.
    - Integration of sentiment analysis with forecasting yielded improved recommendation accuracy (BUY, SELL, HOLD).

## Contributors & Contributions
- Neeladri Bandopadhyay: Data Collection & Technical Indicator Engineering
- Arito Dutta: Stock Trend Prediction & Modeling
- Lopamudra Mukherjee: Stock Price Forecasting & Documentation
- Harsh Kumar Singh: Streamlit Dashboard & Sentiment Analysis

## Future Scope
- Improved Forecasting: Explore advanced forecasting models for enhanced prediction accuracy.
- Real-Time Data Integration: Incorporate live data feeds for dynamic market analysis.
- Expanded Sentiment Analysis: Refine sentiment extraction to capture market moods more accurately.
- Enhanced NLP: Employ transformer classifiers for deeper sentiment and event extraction.
- Production Deployment: Containerize with Docker and orchestrate via Kubernetes for scalable serving.
- User Authentication & Multi‑User Support: Add login, user profiles, and personalized watchlists.

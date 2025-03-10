# Auto Time Series Stock Analysis

## Overview

Auto Time Series Stock Analysis is a Python-based project designed to fetch, process, and visualize stock data. It allows users to analyze historical stock prices and related metrics, perform statistical tests, and even predict future prices using deep learning models (LSTM and GRU). The project leverages data from Yahoo Finance and provides interactive visualizations with Plotly.

## Features

- **Data Acquisition & Preprocessing:**  
  Fetch stock data using `yfinance` and clean/prepare data with `pandas`.
  
- **Visualization:**  
  Generate interactive charts such as price plots, moving averages, volatility trends, seasonal decomposition, and dividend yield plots using Plotly.
  
- **Statistical Analysis:**  
  Perform tests like the Dickey-Fuller test to assess stationarity of the time series.
  
- **Deep Learning Predictions:**  
  Utilize LSTM and GRU models to forecast future stock prices.


## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/rrawatt/auto-time-series-analysis.git
   cd auto-time-series-analysis

2. **Install Dependencies::**
   ```bash
   pip install -r requirements.txt

3. **Test Use:**
   ```bash
   python main.py AAPL 2020-01-01 2021-01-01



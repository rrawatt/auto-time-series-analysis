# src/utils.py
import subprocess
import sys
import importlib.util
import argparse

def install_requirements(modules_string):
    """Installs required libraries if not already installed."""
    modules = modules_string.split()
    for module in modules:
        if importlib.util.find_spec(module) is None:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", module])
                print(f"Successfully installed {module}.")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while installing {module}: {e}")
                sys.exit(1)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Stock Analysis Script")
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('start_date', type=str, help='Start date for stock data (format: YYYY-MM-DD)')
    parser.add_argument('end_date', type=str, help='End date for stock data (format: YYYY-MM-DD)')
    return parser.parse_args()

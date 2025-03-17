import subprocess
import sys
import importlib.util
import argparse

def install_requirements(modules_string):

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

    parser = argparse.ArgumentParser(description="Stock Analysis Script")
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('start_date', type=str, help='Start date for stock data (format: YYYY-MM-DD)')
    parser.add_argument('end_date', type=str, help='End date for stock data (format: YYYY-MM-DD)')
    return parser.parse_args()

def save_fig(fig, filepath):
    """
    Attempts to save a Plotly figure to an image file using Kaleido.
    If fig.write_image() fails, it falls back to using fig.to_image().
    """
    try:
        fig.write_image(filepath, engine="kaleido")
        print(f"Saved figure to {filepath} using write_image().")
    except Exception as e:
        print(f"write_image() failed for {filepath}: {e}")
        try:
            # Fallback: use to_image() to get image bytes and write manually
            img_bytes = fig.to_image(format="png", engine="kaleido")
            with open(filepath, "wb") as f:
                f.write(img_bytes)
            print(f"Saved figure to {filepath} using to_image() fallback.")
        except Exception as e:
            print(f"Fallback failed for {filepath}: {e}")
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # DON'T CHANGE THIS !!!

from flask import Flask, render_template, redirect, url_for
import streamlit.web.bootstrap
import streamlit as st
import subprocess
import threading
import time

app = Flask(__name__)

# Global variable to track if Streamlit is running
streamlit_process = None
streamlit_url = None

def start_streamlit():
    """Start the Streamlit application in a separate process"""
    global streamlit_process
    if streamlit_process is None or streamlit_process.poll() is not None:
        # Start Streamlit on port 8501
        streamlit_process = subprocess.Popen(
            ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Give Streamlit time to start
        time.sleep(5)

@app.route('/')
def index():
    """Redirect to the Streamlit application"""
    # Start Streamlit if it's not running
    start_streamlit()
    # Redirect to the Streamlit URL
    return redirect("https://idx-stock-screener.streamlit.app")

if __name__ == "__main__":
    # Start Streamlit in a separate thread
    threading.Thread(target=start_streamlit).start()
    # Start Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

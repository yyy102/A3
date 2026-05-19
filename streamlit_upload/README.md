# A3 Streamlit Upload

This folder contains the Streamlit Cloud upload version of A3.

## Files to upload

- `app.py`: Streamlit entry point, copied from the original `nlp_app.py`
- `requirements.txt`: Python dependencies for Streamlit Cloud
- `runtime.txt`: Python version marker for this upload folder

## Streamlit Cloud settings

- Main file path: `app.py`
- Python version: `3.12`

The dependency pins keep `gensim` compatible with `scipy` and `numpy` on Streamlit Cloud.

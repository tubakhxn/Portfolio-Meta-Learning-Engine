## dev/creator: tubakhxn

# Portfolio Meta-Learning Engine

This project is an advanced quant dashboard for institutional-grade portfolio construction using meta-learning. It dynamically selects portfolio strategies based on detected market regimes and visualizes market structure using interactive 3D spectral analysis.

## What is this project?
- **Meta-learning engine**: Adapts portfolio strategies (mean-variance, risk parity, etc.) to changing market regimes.
- **Spectral decomposition**: Uses FFT to analyze the frequency structure of portfolio returns, visualized as a 3D surface.
- **Streamlit dashboard**: All visualizations are interactive and styled for a hedge-fund aesthetic.

## How to fork and run
1. **Fork this repo** on GitHub or download the code.
2. **Install requirements**:
   - Python 3.8+
   - `pip install streamlit pandas numpy plotly yfinance scipy scikit-learn`
3. **Run the app**:
   - `streamlit run portfolio_meta_learning_engine.py`
4. **Enjoy the dashboard** in your browser.

## Main files
- `portfolio_meta_learning_engine.py`: The single-file Streamlit app with all logic and visualizations.
- `README.md`: This file.

---
For any questions or improvements, contact tubakhxn.

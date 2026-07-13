# Airline Passenger Forecasting

A polished Streamlit web app for forecasting monthly airline passenger demand using a Recurrent Neural Network (RNN). This app combines interactive data exploration, model evaluation metrics, and future forecasting with modern UI styling and visualization.

## Key Features

- **Interactive Forecasting**: Choose a forecast horizon from 1 to 24 months using the sidebar slider.
- **Model Evaluation Metrics**: View MAE, MSE, and RMSE values immediately to gauge the model's accuracy.
- **Historical Trend Visualization**: Explore the original passenger dataset and historical trend line chart.
- **Future Forecast Output**: Generate future passenger forecasts and download results as CSV.
- **Modern UI Styling**: Custom CSS makes the app visually appealing with gradients, translucent panels, and polished buttons.
- **Plotly Visualizations**: Responsive interactive charts for historical and forecasted passenger data.

## Project Structure

- `app.py` — Streamlit application entry point and UI layout.
- `src/data_loader.py` — Loads and prepares the airline passenger dataset.
- `src/preprocessing.py` — Preprocesses and scales the data for model input.
- `src/sequence_generator.py` — Generates time-series sequences for RNN training.
- `src/train_test_split.py` — Splits sequences into training and testing sets.
- `src/forecast.py` — Uses the trained model to generate future predictions.
- `src/evaluate.py` — Computes performance metrics and saves results.
- `models/` — Stores saved model and scaler artifacts.
- `assets/` — UI assets such as logos and custom styles.
- `data/airline_passengers.csv` — Time-series dataset used for training and forecasting.

## Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The main libraries used are:

- `streamlit` for the web app
- `plotly` for interactive charts
- `tensorflow` and `keras` for the RNN model
- `pandas` and `numpy` for data handling
- `scikit-learn` for evaluation metrics
- `streamlit-lottie` for animated UI effects
- `requests` for fetching remote assets

## How It Works

1. The app loads `data/airline_passengers.csv` via `DataLoader`.
2. Historical passenger counts are displayed in a table and trend chart.
3. The trained RNN model is evaluated with `Evaluator` to compute MAE, MSE, and RMSE.
4. The forecast slider controls the number of future months predicted.
5. When the user clicks `Generate Forecast`, the model predicts future passenger counts and displays them with a chart and downloadable CSV.

## Run the App

Launch the app from the project root:

```bash
python -m streamlit run app.py
```

Then open the local Streamlit URL displayed in your terminal.

## Styling & UX

The app is designed for a clean, dark theme with:

- glassmorphism-style sidebar and cards
- gradient button accents
- consistent typography and spacing
- responsive layout for desktop and larger screens

## Notes

- `models/lstm_model.keras` and `models/scaler.pkl` are required for forecasting and evaluation.
- If `streamlit-lottie` is installed, the forecast view includes animated graphics for enhanced UX.

---

Made for airline passenger demand forecasting with an interactive, polished front-end experience.

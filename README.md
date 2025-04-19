### Setup

0. Read data card in [Kaggle](https://www.kaggle.com/datasets/arunavakrchakraborty/australia-weather-data?select=Weather+Training+Data.csv)
1. Run `python -m venv .venv`
2. Activate the environment via `.venv\Scripts\activate` and run `pip install -r requirements.txt`
3. Start the experiment tracker via `mlflow server --host 127.0.0.1 --port 5000` in a separate terminal. This is to observe loss and accuracy curves.
4. To re-evaluate model on existing weights, run `python main.py`.
5. To switch between **LSTM, GRU, and Transformer**, modify the `model` parameter in `config.py`.
6. To re-train model, set `use_existing_weights = False` in `config.py`.

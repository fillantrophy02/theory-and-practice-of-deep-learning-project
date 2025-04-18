### Setup

0. Read data card in [Kaggle](https://www.kaggle.com/datasets/arunavakrchakraborty/australia-weather-data?select=Weather+Training+Data.csv)
1. Run `python -m venv .venv`
2. Activate the environment and run `pip install -r requirements.txt`
3. Start the experiment tracker via `mlflow server --host 127.0.0.1 --port 5000` in a separate terminal
4. Run `python train.py`
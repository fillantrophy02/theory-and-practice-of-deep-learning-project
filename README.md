### Setup

0. Read data card in [Kaggle](https://www.kaggle.com/datasets/arunavakrchakraborty/australia-weather-data?select=Weather+Training+Data.csv)
1. **Run GRU & Seq2Seq_GRU Models**
Run from the root directory:
```bash
python -m models.gru
python -m models.gru_seq2seq
```
2. **Results for Predictions CSV**
ckpts/gru/latest_preds.csv

3. **Results for Model weights**
ckpts/gru/model_weights_gru.pth
ckpts/gru/model_weights_seq2seq.pth

4. **ROC & PR curves**
GRU:
ckpts/gru/roc_latest.png
ckpts/gru/pr_latest.png

Seq2Seq GRU:
ckpts/gru/roc_curve_s2s.png
ckpts/gru/pr_curve_s2s.png

5. **Edit Config for hyperparameter tuning**
config_custom/config_gru.py
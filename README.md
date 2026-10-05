# Telecom Marketing Campaign: Subscription Prediction

Predicting which customers will subscribe to a new plan after a telemarketing campaign (41,188 records, ~11.3% positive rate). The project compares seven supervised models on an imbalanced dataset (SMOTEENN resampling) and includes an interactive dashboard.

Written as part of the Master of Data Science and Innovation (UTS) coursework.

## What's inside

| Path | Description |
|---|---|
| `notebooks/Marketing_Campaign_Analysis.ipynb` | Full analysis: preprocessing, EDA, statistical tests, modelling, tuning |
| `telecom_dashboard.py` | Plotly Dash app: overview, EDA, campaign analysis, model benchmarking, live subscription predictor |
| `docs/Project_Report.pdf` | Written project report (cover page removed) |
| `data/` | Place the dataset here (not included) |

## Models compared
Logistic Regression, Ridge, Decision Tree, Random Forest, Gaussian Naive Bayes, KNN, LightGBM. See the report for results and limitations.

## Run the dashboard

```bash
pip install -r requirements.txt

# put TeleCom_Data-1.csv in data/ (or set TELECOM_DATA_PATH to its location)
python telecom_dashboard.py
```

Then open http://127.0.0.1:8050. Startup trains all models, which can take about a minute because of SMOTEENN.

## Data
`TeleCom_Data-1.csv` is semicolon-delimited and is not included in this repository. Add it to `data/` yourself.

## Notebook
The notebook was written in Google Colab and reads the CSV from `/content/TeleCom_Data-1.csv`. Change that path if running elsewhere.

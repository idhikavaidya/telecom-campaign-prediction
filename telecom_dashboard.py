"""
TeleCom Marketing Campaign Interactive Dashboard
Mirrors the analysis in Marketing_Campaign_Analysis.ipynb
Run:  python telecom_dashboard.py
Then open: http://127.0.0.1:8050
"""
import sys, os as _os
# Prevent any .py files in the same folder from shadowing stdlib modules
_here = _os.path.dirname(_os.path.abspath(__file__))
while _here in sys.path:
    sys.path.remove(_here)
del _here

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTEENN

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = _os.environ.get("TELECOM_DATA_PATH", _os.path.join("data", "TeleCom_Data-1.csv"))

MONTH_ORDER = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DAY_ORDER   = ["mon", "tue", "wed", "thu", "fri"]

FEATURES = [
    "age", "euribor_3_month_rate", "num_contacts_current_campaign",
    "job_type_enc", "education_level_enc", "last_contact_day_enc",
    "marital_status_enc", "num_employees", "days_since_last_contact",
    "has_housing_loan_enc",
]
TARGET = "is_subscribed"

CAT_COLS = [
    "job_type", "marital_status", "education_level", "has_credit_default",
    "has_housing_loan", "has_personal_loan", "contact_method",
    "last_contact_month", "last_contact_day", "previous_campaign_outcome",
]
NUM_COLS = [
    "age", "call_duration_sec", "num_contacts_current_campaign",
    "days_since_last_contact", "num_previous_contacts",
    "employment_variation_rate", "consumer_price_index",
    "consumer_confidence_index", "euribor_3_month_rate", "num_employees",
]

PALETTE = ["#1F77B4", "#2CA02C", "#FF7F0E", "#D62728", "#9467BD", "#17BECF", "#8C564B"]
CLR_YES, CLR_NO = "#2CA02C", "#D62728"

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
encoders: dict = {}


def load_data() -> pd.DataFrame:
    print("Loading data...")
    with open(DATA_PATH, "r") as f:
        lines = f.readlines()
    columns = [c.strip('"') for c in lines[0].strip().split(";")]
    data = [[v.strip('"') for v in l.strip().split(";")] for l in lines[1:]]
    df = pd.DataFrame(data, columns=columns)

    df = df.rename(columns={
        "job": "job_type", "marital": "marital_status",
        "education": "education_level", "default": "has_credit_default",
        "housing": "has_housing_loan", "loan": "has_personal_loan",
        "contact": "contact_method", "month": "last_contact_month",
        "day_of_week": "last_contact_day", "duration": "call_duration_sec",
        "campaign": "num_contacts_current_campaign", "pdays": "days_since_last_contact",
        "previous": "num_previous_contacts", "poutcome": "previous_campaign_outcome",
        "emp.var.rate": "employment_variation_rate",
        "cons.price.idx": "consumer_price_index",
        "cons.conf.idx": "consumer_confidence_index",
        "euribor3m": "euribor_3_month_rate",
        "nr.employed": "num_employees", "y": "term_deposit_subscribed",
    })

    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_subscribed"] = (df["term_deposit_subscribed"] == "yes").astype(int)
    df = df.drop_duplicates().reset_index(drop=True)
    df["contact_status"] = np.where(
        df["days_since_last_contact"] == 999, "Never Contacted", "Previously Contacted"
    )

    global encoders
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    sub_rate = df["is_subscribed"].mean()
    print(f"  {len(df):,} rows | subscription rate {sub_rate:.1%}")
    return df


# ─── MODEL TRAINING ───────────────────────────────────────────────────────────
def train_models(df: pd.DataFrame) -> dict:
    print("Training models (SMOTEENN may take ~60 s)...")
    model_df = df[FEATURES + [TARGET]].dropna()
    X, y = model_df[FEATURES], model_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Run SMOTEENN once on raw data, then scale the result
    print("  Applying SMOTEENN...")
    sm = SMOTEENN(random_state=42)
    X_sm, y_sm = sm.fit_resample(X_train.values, y_train)
    X_sm_sc = scaler.transform(X_sm)
    neg, pos = np.bincount(y_sm)
    print(f"  After SMOTEENN - No: {neg:,}  Yes: {pos:,}")

    CFGS = [
        {
            "name": "Logistic Regression", "type": "Parametric",
            "scaled": True, "thr": 0.55,
            "clf": LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs", random_state=42),
        },
        {
            "name": "Ridge LR", "type": "Parametric",
            "scaled": True, "thr": 0.50,
            "clf": CalibratedClassifierCV(RidgeClassifier(alpha=2.0), cv=3, method="sigmoid"),
        },
        {
            "name": "Decision Tree", "type": "Non-Parametric",
            "scaled": False, "thr": 0.50,
            "clf": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=42),
        },
        {
            "name": "Random Forest", "type": "Non-Parametric",
            "scaled": False, "thr": 0.50,
            "clf": RandomForestClassifier(n_estimators=100, max_depth=10,
                                          random_state=42, n_jobs=-1),
        },
        {
            "name": "Gaussian NB", "type": "Parametric",
            "scaled": True, "thr": 0.50,
            "clf": GaussianNB(),
        },
        {
            "name": "KNN", "type": "Non-Parametric",
            "scaled": True, "thr": 0.50,
            "clf": KNeighborsClassifier(n_neighbors=15, metric="euclidean", n_jobs=-1),
        },
        {
            "name": "LightGBM", "type": "Non-Parametric",
            "scaled": False, "thr": 0.50,
            "clf": LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                  num_leaves=40, random_state=42, n_jobs=-1, verbose=-1),
        },
    ]

    results, roc_data, fitted = [], {}, {}
    for cfg in CFGS:
        name = cfg["name"]
        print(f"  Training {name}...")
        Xtr = X_sm_sc if cfg["scaled"] else X_sm
        ytr = y_sm
        Xte = X_test_sc if cfg["scaled"] else X_test.values

        clf = cfg["clf"]
        clf.fit(Xtr, ytr)
        fitted[name] = {"clf": clf, "scaled": cfg["scaled"], "thr": cfg["thr"]}

        prob = clf.predict_proba(Xte)[:, 1]
        pred = (prob >= cfg["thr"]).astype(int)
        fpr, tpr, _ = roc_curve(y_test, prob)
        cm = confusion_matrix(y_test, pred)

        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        results.append({
            "Model":     name,
            "Type":      cfg["type"],
            "Accuracy":  round(accuracy_score(y_test, pred), 4),
            "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, pred), 4),
            "F1":        round(f1_score(y_test, pred), 4),
            "ROC-AUC":   round(roc_auc_score(y_test, prob), 4),
            "CM":        cm.tolist(),
        })

    lgbm_imp = pd.DataFrame({
        "Feature":    FEATURES,
        "Importance": fitted["LightGBM"]["clf"].feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print("All models ready.")
    return {
        "results":      pd.DataFrame(results),
        "roc_data":     roc_data,
        "fitted":       fitted,
        "scaler":       scaler,
        "X_test":       X_test,
        "y_test":       y_test,
        "X_test_sc":    X_test_sc,
        "lgbm_imp":     lgbm_imp,
    }


# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────
df = load_data()
MD = train_models(df)
results_df  = MD["results"]
overall_rate = float(df["is_subscribed"].mean() * 100)

CHART_CFG = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(t=55, b=40, l=55, r=25),
    font=dict(family="Segoe UI, Arial", size=12),
)

# ─── HELPER BUILDERS ──────────────────────────────────────────────────────────

def _avg_line(fig, val=None):
    v = overall_rate if val is None else val
    fig.add_hline(
        y=v, line_dash="dash", line_color="red", line_width=1.5,
        annotation_text=f"Avg {v:.1f}%", annotation_position="top right",
    )
    return fig


def _clean(fig, height=360):
    fig.update_layout(
        **CHART_CFG,
        height=height,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#ebebeb"),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────

def _overview_kpi_row():
    n      = len(df)
    n_yes  = int(df["is_subscribed"].sum())
    avg_d  = df["call_duration_sec"].mean()
    avg_c  = df["num_contacts_current_campaign"].mean()

    def card(title, value, color):
        return dbc.Col(
            dbc.Card(dbc.CardBody([
                html.P(title, className="text-muted small mb-1"),
                html.H3(value, className=f"fw-bold text-{color} mb-0"),
            ]), className="shadow-sm text-center h-100"),
            xs=6, md=3,
        )

    return dbc.Row([
        card("Total Records",       f"{n:,}",              "primary"),
        card("Subscribed",          f"{n_yes:,}",          "success"),
        card("Subscription Rate",   f"{overall_rate:.1f}%","warning"),
        card("Avg Call Duration",   f"{avg_d:.0f} s",      "info"),
    ], className="g-3 mb-4")


def _overview_charts():
    n_yes = int(df["is_subscribed"].sum())
    n_no  = len(df) - n_yes

    donut = go.Figure(go.Pie(
        labels=["Subscribed", "Not Subscribed"],
        values=[n_yes, n_no], hole=0.55,
        marker_colors=[CLR_YES, CLR_NO],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,}<extra></extra>",
    ))
    donut.update_layout(
        title="Subscription Split", height=320,
        **CHART_CFG,
        legend=dict(orientation="h", y=-0.12),
    )

    # Monthly volume + rate dual axis
    mon = (
        df.groupby("last_contact_month")
          .agg(Total=("is_subscribed", "count"), Yes=("is_subscribed", "sum"))
          .reindex(MONTH_ORDER).reset_index()
    )
    mon["Rate"] = mon["Yes"] / mon["Total"] * 100

    mfig = go.Figure()
    mfig.add_trace(go.Bar(
        x=mon["last_contact_month"], y=mon["Total"],
        name="Total Contacts", marker_color="#AEC7E8", opacity=0.75,
    ))
    mfig.add_trace(go.Bar(
        x=mon["last_contact_month"], y=mon["Yes"],
        name="Subscribed", marker_color=CLR_YES,
    ))
    mfig.add_trace(go.Scatter(
        x=mon["last_contact_month"], y=mon["Rate"],
        name="Sub Rate (%)", yaxis="y2", mode="lines+markers",
        line=dict(color="#FF7F0E", width=2.5), marker=dict(size=7),
    ))
    mfig.update_layout(
        barmode="overlay",
        title="Monthly Campaign Volume & Subscription Rate",
        height=320, **CHART_CFG,
        yaxis=dict(title="Count", showgrid=True, gridcolor="#ebebeb"),
        yaxis2=dict(title="Rate (%)", overlaying="y", side="right", range=[0, 85]),
        legend=dict(orientation="h", y=1.12),
        xaxis=dict(showgrid=False),
    )

    return dbc.Row([
        dbc.Col(dcc.Graph(figure=donut), width=4),
        dbc.Col(dcc.Graph(figure=mfig),  width=8),
    ], className="g-3 mb-4")


def _overview_table():
    stats = df[NUM_COLS].describe().T.round(2).reset_index()
    stats.columns = ["Feature"] + list(stats.columns[1:])
    return dash_table.DataTable(
        data=stats.to_dict("records"),
        columns=[{"name": c, "id": c} for c in stats.columns],
        style_cell={"textAlign": "center", "fontSize": 12, "padding": "6px",
                    "fontFamily": "Segoe UI"},
        style_header={"backgroundColor": "#1F77B4", "color": "white",
                      "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f5f8ff"}
        ],
        page_size=12, sort_action="native",
    )


def make_overview_tab():
    return html.Div([
        _overview_kpi_row(),
        _overview_charts(),
        html.H5("Numeric Feature Statistics", className="fw-bold mb-2"),
        _overview_table(),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — EDA
# ──────────────────────────────────────────────────────────────────────────────

def make_eda_tab():
    cat_opts = [{"label": f"{c}  (categorical)", "value": c} for c in CAT_COLS]
    num_opts = [{"label": f"{c}  (numeric)",     "value": c} for c in NUM_COLS]

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Feature", className="fw-semibold"),
                dcc.Dropdown(id="eda-feat", options=cat_opts + num_opts,
                             value="job_type", clearable=False),
            ], md=4),
            dbc.Col([
                html.Label("Filter by Subscription", className="fw-semibold"),
                dbc.RadioItems(
                    id="eda-sub", inline=True,
                    options=[
                        {"label": " All",            "value": "all"},
                        {"label": " Subscribed",     "value": 1},
                        {"label": " Not Subscribed", "value": 0},
                    ],
                    value="all",
                ),
            ], md=4),
            dbc.Col([
                html.Label("Chart type (numeric)", className="fw-semibold"),
                dbc.RadioItems(
                    id="eda-chart-type", inline=True,
                    options=[
                        {"label": " Histogram", "value": "hist"},
                        {"label": " Box plot",  "value": "box"},
                    ],
                    value="hist",
                ),
            ], md=4),
        ], className="g-3 mb-3"),
        dcc.Graph(id="eda-dist", style={"height": "400px"}),
        html.Hr(),
        html.H6("Subscription Rate by Feature", className="fw-bold mt-2 mb-1"),
        dcc.Graph(id="eda-rate", style={"height": "340px"}),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — CAMPAIGN ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def _rate_bar(data, x_col, title, *, height=330):
    colors = ["#1a9850" if r > overall_rate else "#4472C4" for r in data["rate"]]
    fig = go.Figure(go.Bar(
        x=data[x_col].astype(str), y=data["rate"],
        marker_color=colors,
        text=[f"{r:.1f}%" for r in data["rate"]], textposition="outside",
        hovertemplate="<b>%{x}</b><br>Sub Rate: %{y:.1f}%<extra></extra>",
    ))
    _avg_line(fig)
    fig.update_layout(
        title=title, yaxis_title="Subscription Rate (%)", **CHART_CFG,
        height=height,
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor="#ebebeb"),
    )
    return fig


def make_campaign_tab():
    def grp_rate(col, sort=True):
        g = df.groupby(col)["is_subscribed"].agg(["mean", "count"]).reset_index()
        g["rate"] = g["mean"] * 100
        return g.sort_values("rate", ascending=False) if sort else g

    cm_r  = grp_rate("contact_method")
    po_r  = grp_rate("previous_campaign_outcome")
    job_r = grp_rate("job_type")
    edu_r = grp_rate("education_level")
    mar_r = grp_rate("marital_status")

    day_r = (
        df.groupby("last_contact_day")["is_subscribed"]
          .mean().reindex(DAY_ORDER) * 100
    ).reset_index()
    day_r.columns = ["day", "rate"]

    # Contact status
    cs = df.groupby("contact_status")["is_subscribed"].agg(["mean", "count"]).reset_index()
    cs["rate"] = cs["mean"] * 100

    # Stacked yes/no bar for 4 key features
    def stacked_bar(col, title):
        ct = pd.crosstab(df[col], df["is_subscribed"], normalize="index") * 100
        ct.columns = ["No", "Yes"]
        ct = ct.sort_values("Yes", ascending=False).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ct[col].astype(str), y=ct["No"],  name="No",
            marker_color=CLR_NO, opacity=0.8,
        ))
        fig.add_trace(go.Bar(
            x=ct[col].astype(str), y=ct["Yes"], name="Yes",
            marker_color=CLR_YES, opacity=0.8,
            text=[f"{v:.0f}%" for v in ct["Yes"]], textposition="inside",
            textfont=dict(color="white", size=10),
        ))
        fig.update_layout(
            barmode="stack", title=title,
            yaxis_title="% of group", **CHART_CFG, height=330,
            xaxis=dict(showgrid=False, tickangle=-30),
            yaxis=dict(showgrid=True, gridcolor="#ebebeb"),
            legend=dict(orientation="h", y=1.08),
        )
        return fig

    corr_cols = NUM_COLS + ["is_subscribed"]
    corr = df[corr_cols].corr().round(2)
    hmap = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu_r", zmid=0,
        text=corr.values, texttemplate="%{text}",
        hovertemplate="%{x} vs %{y}: %{z}<extra></extra>",
    ))
    hmap.update_layout(
        title="Numeric Correlation Matrix", height=430,
        **{**CHART_CFG, "margin": dict(t=55, b=100, l=140, r=25)},
        xaxis=dict(tickangle=-40),
    )

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=_rate_bar(cm_r,  "contact_method",            "By Contact Method")),  md=4),
            dbc.Col(dcc.Graph(figure=_rate_bar(po_r,  "previous_campaign_outcome", "By Prev. Outcome")),   md=4),
            dbc.Col(dcc.Graph(figure=_rate_bar(day_r, "day",                       "By Day of Week")),     md=4),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=_rate_bar(job_r, "job_type",        "By Job Type")),        md=6),
            dbc.Col(dcc.Graph(figure=_rate_bar(edu_r, "education_level", "By Education Level")), md=6),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=stacked_bar("job_type",   "Yes/No Split — Job Type")),   md=6),
            dbc.Col(dcc.Graph(figure=stacked_bar("contact_method", "Yes/No Split — Contact Method")), md=6),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=hmap), md=12),
        ], className="g-3"),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────

def make_models_tab():
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    model_names = results_df["Model"].tolist()

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Metrics to display", className="fw-semibold"),
                dbc.Checklist(
                    id="mod-metrics",
                    options=[{"label": f"  {m}", "value": m} for m in metrics],
                    value=metrics, inline=True,
                ),
            ], md=7),
            dbc.Col([
                html.Label("Model for confusion matrix", className="fw-semibold"),
                dcc.Dropdown(
                    id="mod-cm-sel",
                    options=[{"label": m, "value": m} for m in model_names],
                    value="LightGBM", clearable=False,
                ),
            ], md=5),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="mod-compare"),  md=7),
            dbc.Col(dcc.Graph(id="mod-cm"),        md=5),
        ], className="g-3 mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="mod-roc"),       md=7),
            dbc.Col(dcc.Graph(id="mod-feat-imp"),  md=5),
        ], className="g-3 mb-3"),
        html.H5("Performance Summary Table", className="fw-bold mb-2"),
        dash_table.DataTable(
            data=results_df.drop(columns="CM").to_dict("records"),
            columns=[{"name": c, "id": c} for c in results_df.columns if c != "CM"],
            style_cell={"textAlign": "center", "fontSize": 12, "padding": "6px",
                        "fontFamily": "Segoe UI"},
            style_header={"backgroundColor": "#1F77B4", "color": "white",
                          "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f5f8ff"},
                {"if": {"filter_query": '{Model} = "LightGBM"'},
                 "backgroundColor": "#e8f5e9", "fontWeight": "bold"},
            ],
            sort_action="native",
        ),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — SUBSCRIPTION PREDICTOR
# ──────────────────────────────────────────────────────────────────────────────

def make_predictor_tab():
    def _dd(id_, col, default):
        opts = sorted(df[col].dropna().unique().tolist())
        return dcc.Dropdown(
            id=id_,
            options=[{"label": v, "value": v} for v in opts],
            value=default, clearable=False,
        )

    def _sl(id_, lo, hi, step, val, marks, suffix=""):
        return dcc.Slider(
            lo, hi, step, value=val, id=id_, marks=marks,
            tooltip={"placement": "bottom", "always_visible": True},
        )

    controls = dbc.Card([
        dbc.CardHeader(html.H5("Customer Profile", className="fw-bold mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Age", className="fw-semibold"),
                    _sl("p-age", 18, 95, 1, 40,
                        {18: "18", 40: "40", 60: "60", 95: "95"}),
                ], md=6),
                dbc.Col([
                    html.Label("Job Type", className="fw-semibold"),
                    _dd("p-job", "job_type", "admin."),
                ], md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Education", className="fw-semibold"),
                    _dd("p-edu", "education_level", "university.degree"),
                ], md=6),
                dbc.Col([
                    html.Label("Marital Status", className="fw-semibold"),
                    _dd("p-marital", "marital_status", "married"),
                ], md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Has Housing Loan", className="fw-semibold"),
                    _dd("p-housing", "has_housing_loan", "no"),
                ], md=6),
                dbc.Col([
                    html.Label("Last Contact Day", className="fw-semibold"),
                    _dd("p-day", "last_contact_day", "mon"),
                ], md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Euribor 3-Month Rate", className="fw-semibold"),
                    _sl("p-euribor", 0.5, 5.5, 0.1, 1.3,
                        {0.5: "0.5", 3.0: "3.0", 5.5: "5.5"}),
                ], md=6),
                dbc.Col([
                    html.Label("Campaign Contacts (#)", className="fw-semibold"),
                    _sl("p-contacts", 1, 20, 1, 2,
                        {1: "1", 10: "10", 20: "20"}),
                ], md=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Num Employees (macro)", className="fw-semibold"),
                    _sl("p-empl", 4900, 5300, 10, 5099,
                        {4900: "4900", 5100: "5100", 5300: "5300"}),
                ], md=6),
                dbc.Col([
                    html.Label("Days Since Last Contact (999 = never)", className="fw-semibold"),
                    _sl("p-pdays", 0, 999, 1, 999,
                        {0: "0", 500: "500", 999: "Never"}),
                ], md=6),
            ], className="mb-3"),
            dbc.Button(
                "Predict Subscription Probability",
                id="predict-btn", color="primary",
                className="w-100 mt-1", size="lg",
            ),
        ]),
    ], className="shadow-sm")

    result_panel = html.Div([
        dbc.Card([
            dbc.CardHeader(html.H5("Prediction Result — LightGBM", className="fw-bold mb-0")),
            dbc.CardBody([
                dcc.Graph(id="gauge-chart", style={"height": "290px"}),
                html.Div(id="pred-alert", className="mt-2"),
            ]),
        ], className="shadow-sm mb-3"),
        dbc.Card([
            dbc.CardHeader(html.H5("Feature Importance (LightGBM)", className="fw-bold mb-0")),
            dbc.CardBody(dcc.Graph(id="pred-imp", style={"height": "280px"})),
        ], className="shadow-sm"),
    ])

    return html.Div([
        dbc.Row([
            dbc.Col(controls,      md=6),
            dbc.Col(result_panel,  md=6),
        ], className="g-3"),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# APP LAYOUT
# ──────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)
app.title = "TeleCom Campaign Dashboard"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("TeleCom Marketing Campaign Dashboard",
                    className="text-primary fw-bold mt-3 mb-0"),
            html.P(
                "Interactive analysis of bank telemarketing subscription data  |  "
                f"{len(df):,} records  |  "
                f"Subscription rate {overall_rate:.1f}%",
                className="text-muted small",
            ),
        ])
    ], className="mb-3"),

    dbc.Tabs([
        dbc.Tab(make_overview_tab(),  label="📊 Overview",          tab_id="t-ov"),
        dbc.Tab(make_eda_tab(),       label="🔍 EDA",               tab_id="t-eda"),
        dbc.Tab(make_campaign_tab(),  label="📈 Campaign Analysis",  tab_id="t-camp"),
        dbc.Tab(make_models_tab(),    label="🤖 Model Performance",  tab_id="t-mod"),
        dbc.Tab(make_predictor_tab(), label="🎯 Predictor",          tab_id="t-pred"),
    ], active_tab="t-ov"),

    html.Hr(className="mt-4"),
    html.P("Built with Plotly Dash · LightGBM · SMOTEENN",
           className="text-center text-muted small mb-3"),
], fluid=True, className="px-4")


# ──────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ──────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("eda-dist", "figure"),
    Output("eda-rate", "figure"),
    Input("eda-feat", "value"),
    Input("eda-sub",  "value"),
    Input("eda-chart-type", "value"),
)
def cb_eda(feat, sub_filt, chart_type):
    dff = df if sub_filt == "all" else df[df["is_subscribed"] == sub_filt]
    is_cat = feat in CAT_COLS

    # ── Distribution chart ──────────────────────────────────────────────────
    if is_cat:
        vc = dff[feat].value_counts().reset_index()
        vc.columns = [feat, "count"]
        fig = px.bar(
            vc, x=feat, y="count",
            title=f"Distribution — {feat}",
            color_discrete_sequence=["#1F77B4"],
            text="count",
        )
        fig.update_traces(textposition="outside")
    else:
        fig = go.Figure()
        if chart_type == "hist":
            if sub_filt == "all":
                for val, color, label in [(0, CLR_NO, "Not Subscribed"),
                                          (1, CLR_YES, "Subscribed")]:
                    s = df[df["is_subscribed"] == val][feat].dropna()
                    fig.add_trace(go.Histogram(
                        x=s, name=label, marker_color=color,
                        opacity=0.65, nbinsx=40,
                    ))
                fig.update_layout(barmode="overlay")
            else:
                fig.add_trace(go.Histogram(
                    x=dff[feat].dropna(), nbinsx=40,
                    marker_color="#1F77B4", opacity=0.75,
                ))
        else:
            if sub_filt == "all":
                for val, color, label in [(0, CLR_NO, "Not Subscribed"),
                                          (1, CLR_YES, "Subscribed")]:
                    s = df[df["is_subscribed"] == val][feat].dropna()
                    fig.add_trace(go.Box(
                        y=s, name=label, marker_color=color, boxmean=True,
                    ))
            else:
                fig.add_trace(go.Box(
                    y=dff[feat].dropna(), marker_color="#1F77B4", boxmean=True,
                ))
        fig.update_layout(title=f"Distribution — {feat}")

    _clean(fig, height=400)

    # ── Subscription rate chart ─────────────────────────────────────────────
    if is_cat:
        g = (
            df.groupby(feat)["is_subscribed"]
              .agg(["mean", "count"])
              .reset_index()
              .sort_values("mean", ascending=False)
        )
        g["rate"] = g["mean"] * 100
        colors = ["#1a9850" if r > overall_rate else "#4472C4" for r in g["rate"]]
        rfig = go.Figure(go.Bar(
            x=g[feat].astype(str), y=g["rate"],
            marker_color=colors,
            text=[f"{r:.1f}%" for r in g["rate"]], textposition="outside",
            customdata=g["count"],
            hovertemplate="<b>%{x}</b><br>Rate: %{y:.1f}%<br>n=%{customdata:,}<extra></extra>",
        ))
    else:
        tmp = df[[feat, "is_subscribed"]].dropna().copy()
        tmp["bin"] = pd.cut(tmp[feat], bins=8, precision=1)
        binned = (
            tmp.groupby("bin", observed=True)["is_subscribed"]
               .agg(["mean", "count"])
               .reset_index()
        )
        binned["rate"] = binned["mean"] * 100
        colors = ["#1a9850" if r > overall_rate else "#4472C4" for r in binned["rate"]]
        rfig = go.Figure(go.Bar(
            x=binned["bin"].astype(str), y=binned["rate"],
            marker_color=colors,
            text=[f"{r:.1f}%" for r in binned["rate"]], textposition="outside",
        ))

    _avg_line(rfig)
    rfig.update_layout(
        title=f"Subscription Rate by {feat}",
        yaxis_title="Subscription Rate (%)",
        **CHART_CFG, height=340,
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor="#ebebeb"),
    )
    return fig, rfig


@app.callback(
    Output("mod-compare",   "figure"),
    Output("mod-cm",        "figure"),
    Output("mod-roc",       "figure"),
    Output("mod-feat-imp",  "figure"),
    Input("mod-metrics",  "value"),
    Input("mod-cm-sel",   "value"),
)
def cb_models(metrics, cm_model):
    metrics = metrics or ["Accuracy"]

    # ── Grouped horizontal bar comparison ───────────────────────────────────
    sub = results_df.set_index("Model")[metrics].reset_index()
    cfig = go.Figure()
    for i, m in enumerate(metrics):
        cfig.add_trace(go.Bar(
            name=m,
            y=sub["Model"], x=sub[m],
            orientation="h",
            marker_color=PALETTE[i % len(PALETTE)],
            text=[f"{v:.3f}" for v in sub[m]],
            textposition="outside",
        ))
    cfig.update_layout(
        barmode="group", title="Model Comparison",
        xaxis_title="Score", xaxis_range=[0, 1.15],
        **{**CHART_CFG, "margin": dict(t=55, b=35, l=160, r=65)},
        height=400,
        yaxis=dict(showgrid=False),
        xaxis=dict(showgrid=True, gridcolor="#ebebeb"),
        legend=dict(orientation="h", y=1.1),
    )

    # ── Confusion matrix ─────────────────────────────────────────────────────
    row = results_df[results_df["Model"] == cm_model].iloc[0]
    cm_arr = np.array(row["CM"])
    cm_fig = go.Figure(go.Heatmap(
        z=cm_arr,
        x=["Pred No", "Pred Yes"],
        y=["True No", "True Yes"],
        colorscale="Blues",
        text=cm_arr, texttemplate="%{text}",
        showscale=False,
        hovertemplate="%{y} → %{x}: %{z}<extra></extra>",
    ))
    cm_fig.update_layout(
        title=f"Confusion Matrix — {cm_model}",
        **{**CHART_CFG, "margin": dict(t=55, b=60, l=90, r=30)},
        height=400,
    )

    # ── ROC curves ───────────────────────────────────────────────────────────
    rfig = go.Figure()
    rfig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                   line=dict(dash="dash", color="#aaa", width=1))
    for i, (name, rd) in enumerate(MD["roc_data"].items()):
        auc = float(results_df[results_df["Model"] == name]["ROC-AUC"].iloc[0])
        rfig.add_trace(go.Scatter(
            x=rd["fpr"], y=rd["tpr"], name=f"{name} ({auc:.3f})",
            mode="lines", line=dict(color=PALETTE[i % len(PALETTE)], width=2),
        ))
    rfig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        **CHART_CFG, height=400,
        xaxis=dict(showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(showgrid=True, gridcolor="#ebebeb"),
        legend=dict(x=0.52, y=0.06, bgcolor="rgba(255,255,255,0.85)"),
    )

    # ── Feature importance ───────────────────────────────────────────────────
    imp = MD["lgbm_imp"]
    colors_imp = ["#1a9850" if i < 4 else "#AEC7E8" for i in range(len(imp))]
    ifig = go.Figure(go.Bar(
        x=imp["Importance"][::-1],
        y=imp["Feature"][::-1],
        orientation="h",
        marker_color=colors_imp[::-1],
        text=[f"{v:.0f}" for v in imp["Importance"][::-1]],
        textposition="outside",
    ))
    ifig.update_layout(
        title="LightGBM Feature Importance",
        xaxis_title="Importance Score",
        **{**CHART_CFG, "margin": dict(t=55, b=35, l=200, r=65)},
        height=400,
        yaxis=dict(showgrid=False),
        xaxis=dict(showgrid=True, gridcolor="#ebebeb"),
    )

    return cfig, cm_fig, rfig, ifig


@app.callback(
    Output("gauge-chart", "figure"),
    Output("pred-alert",  "children"),
    Output("pred-imp",    "figure"),
    Input("predict-btn",  "n_clicks"),
    State("p-age",      "value"),
    State("p-job",      "value"),
    State("p-edu",      "value"),
    State("p-marital",  "value"),
    State("p-housing",  "value"),
    State("p-day",      "value"),
    State("p-euribor",  "value"),
    State("p-contacts", "value"),
    State("p-empl",     "value"),
    State("p-pdays",    "value"),
    prevent_initial_call=False,
)
def cb_predict(_, age, job, edu, marital, housing, day,
               euribor, contacts, empl, pdays):
    def enc(col, val):
        le = encoders[col]
        try:
            return int(le.transform([str(val)])[0])
        except Exception:
            return 0

    X_in = np.array([[
        age or 40,
        euribor or 1.3,
        contacts or 2,
        enc("job_type", job or "admin."),
        enc("education_level", edu or "university.degree"),
        enc("last_contact_day", day or "mon"),
        enc("marital_status", marital or "married"),
        empl or 5099,
        pdays if pdays is not None else 999,
        enc("has_housing_loan", housing or "no"),
    ]])

    lgbm_clf = MD["fitted"]["LightGBM"]["clf"]
    prob = float(lgbm_clf.predict_proba(X_in)[0, 1])

    # Gauge
    color = CLR_YES if prob >= 0.5 else ("#FF7F0E" if prob >= 0.3 else CLR_NO)
    gfig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 38}},
        delta={"reference": overall_rate, "suffix": "%",
               "increasing": {"color": CLR_YES},
               "decreasing": {"color": CLR_NO}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30],  "color": "#FFEBEE"},
                {"range": [30, 50], "color": "#FFF8E1"},
                {"range": [50, 100],"color": "#E8F5E9"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.75, "value": 50,
            },
        },
        title={"text": "Subscription Probability", "font": {"size": 15}},
    ))
    gfig.update_layout(paper_bgcolor="white", height=290,
                       margin=dict(t=70, b=20, l=30, r=30))

    risk  = "HIGH" if prob >= 0.5 else ("MEDIUM" if prob >= 0.3 else "LOW")
    c_map = {"HIGH": "success", "MEDIUM": "warning", "LOW": "danger"}
    alert = dbc.Alert([
        html.H5(f"Subscription Likelihood: {risk}", className="mb-1"),
        html.P(
            f"Model probability: {prob:.1%}  ·  Dataset average: {overall_rate:.1f}%",
            className="mb-0 small",
        ),
    ], color=c_map[risk], className="mb-0")

    # Mini importance bar
    imp = MD["lgbm_imp"].head(10)
    ifig = go.Figure(go.Bar(
        x=imp["Feature"],
        y=imp["Importance"],
        marker_color=["#1a9850" if i < 3 else "#AEC7E8" for i in range(len(imp))],
        text=[f"{v:.0f}" for v in imp["Importance"]],
        textposition="outside",
    ))
    ifig.update_layout(
        title="Top 10 Features",
        **{**CHART_CFG, "margin": dict(t=45, b=80, l=30, r=15)},
        height=280,
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor="#ebebeb"),
    )

    return gfig, alert, ifig


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n Dashboard ready at http://127.0.0.1:8050/\n")
    app.run(debug=False, port=8050)

import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st


OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

AI_GRADER_PROMPT_TEMPLATE = """# Exact AI Grading Prompt (Hardcode inside app.py)

SYSTEM:
You are a strict academic grader. Return ONLY valid JSON.

USER:
Grade this time-series forecasting Streamlit project OUT OF 80 points using the fixed rubric below.
Be strict: do not award points unless evidence is present in the submitted JSON.
Return ONLY JSON exactly matching the schema.

RUBRIC MAX:
Data & integrity: 20
Feature engineering: 15
Modeling & evaluation: 25
Dashboard quality: 10
Presentation & rigor: 10

STRICT CAPS:
- If the project only uses baseline features/models with no meaningful additions, cap total_80 <= 45.
- If time-based split is missing/unclear, cap Modeling & evaluation <= 12.
- If missing timestamps/outliers/resampling are not discussed or evidenced, cap Data & integrity <= 10.
- If no metrics table is present, cap Modeling & evaluation <= 10.
- If no insights are provided, cap Presentation & rigor <= 5.

Return JSON:
{
  "scores": {
    "Data & integrity": int,
    "Feature engineering": int,
    "Modeling & evaluation": int,
    "Dashboard quality": int,
    "Presentation & rigor": int
  },
  "total_80": int,
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "actionable_improvements": [string, ...]
}

EVIDENCE JSON:
<insert submission.json contents here>
"""


st.set_page_config(
    page_title="Mini Project B — Time-Series Forecasting Starter",
    layout="wide",
)


def get_openrouter_api_key():
    """Read key from Streamlit Secrets, environment, or UI password input."""
    try:
        key = st.secrets["OPENROUTER_API_KEY"]
        if key:
            return str(key)
    except Exception:
        pass

    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key

    return st.text_input(
        "OpenRouter API key",
        type="password",
        help="Used only when you click the AI grader button.",
    )


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def audit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    audit = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_percent": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
        "unique_count": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    return audit


def clean_time_series(df: pd.DataFrame, timestamp_col: str, target_col: str) -> pd.DataFrame:
    work = df.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[timestamp_col, target_col])
    work = work.sort_values(timestamp_col).reset_index(drop=True)
    return work


def resample_time_series(df: pd.DataFrame, timestamp_col: str, target_col: str, rule: str | None) -> pd.DataFrame:
    if not rule or rule == "No resampling":
        return df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        numeric_cols.append(target_col)

    resampled = (
        df.set_index(timestamp_col)[numeric_cols]
        .resample(rule)
        .mean()
        .dropna(subset=[target_col])
        .reset_index()
    )
    return resampled


def make_baseline_feature_table(df: pd.DataFrame, timestamp_col: str, target_col: str, horizon: int) -> pd.DataFrame:
    features = df[[timestamp_col, target_col]].copy()
    features["lag_1"] = features[target_col].shift(1)
    features["lag_24"] = features[target_col].shift(24)
    features["rolling_mean_24"] = features[target_col].shift(1).rolling(24).mean()
    features["hour"] = features[timestamp_col].dt.hour
    features["weekend"] = features[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)
    features["month"] = features[timestamp_col].dt.month
    features["y_target"] = features[target_col].shift(-horizon)
    return features.dropna().reset_index(drop=True)


def build_submission_json(
    student_name: str,
    student_id: str,
    deployed_url: str,
    repo_url: str,
    project_title: str,
    project_goal: str,
    timestamp_col: str,
    target_col: str,
    horizon: int,
    resampling_rule: str,
    raw_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    results_df,
    insights_text: str,
    missing_discussion: str,
    outlier_discussion: str,
    resampling_discussion: str,
):
    results_table = [] if results_df is None else results_df.to_dict(orient="records")

    evidence = {
        "student": {
            "name": student_name,
            "id": student_id,
            "deployed_url": deployed_url,
            "repo_url": repo_url,
        },
        "project": {
            "title": project_title,
            "goal": project_goal,
            "timestamp_column": timestamp_col,
            "target_column": target_col,
            "forecast_horizon": int(horizon),
            "resampling_rule": resampling_rule,
        },
        "data_evidence": {
            "raw_rows": int(len(raw_df)),
            "clean_rows": int(len(ts_df)),
            "feature_rows": int(len(feature_df)),
            "timestamp_min": str(ts_df[timestamp_col].min()) if len(ts_df) else "",
            "timestamp_max": str(ts_df[timestamp_col].max()) if len(ts_df) else "",
            "automatic_quality_summary": globals().get("data_quality_auto_summary", ""),
            "invalid_rows_removed": int(globals().get("invalid_rows_removed", 0)),
            "duplicate_timestamp_count": int(globals().get("duplicate_timestamp_count", 0)),
            "larger_than_expected_time_gap_count": int(globals().get("time_gap_count", 0)),
            "inferred_frequency": globals().get("inferred_frequency_text", ""),
            "target_summary": globals().get("target_summary_text", ""),
            "missing_discussion": missing_discussion,
            "outlier_discussion": outlier_discussion,
            "automatic_outlier_summary": globals().get("outlier_auto_summary", ""),
            "outlier_count_iqr": int(globals().get("outlier_count", 0)),
            "outlier_percent_iqr": round(float(globals().get("outlier_percent", 0.0)), 3),
            "resampling_discussion": resampling_discussion,
            "automatic_resampling_summary": globals().get("resampling_auto_summary", ""),
        },
        "feature_engineering": {
            "baseline_features_present": ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"],
            "student_added_features": [
                "lag_2",
                "lag_3",
                "lag_48",
                "lag_168",
                "rolling_mean_6",
                "rolling_mean_12",
                "rolling_std_24",
                "hour_sin",
                "hour_cos",
                "month_sin",
                "month_cos",
                "trend_index",
            ],
        },
        "modeling_evaluation": {
            "has_metrics_table": isinstance(results_df, pd.DataFrame) and not results_df.empty,
            "results_table": results_table,
            "time_based_split_evidence": (
                globals().get(
                    "time_split_evidence",
                    "Used chronological 80/20 time-based split: earlier records for training and later records for testing."
                )
                if isinstance(results_df, pd.DataFrame) and not results_df.empty
                else ""
            ),
            "train_rows": int(globals().get("train_rows", 0)),
            "test_rows": int(globals().get("test_rows", 0)),
            "train_period": globals().get("train_period", ""),
            "test_period": globals().get("test_period", ""),
            "models_used": [] if results_df is None else results_df["model"].tolist(),
            "best_model": globals().get("best_model_name", ""),
            "best_model_metrics": globals().get("best_model_metrics", {}),
            "has_prediction_table": bool(globals().get("has_prediction_table", False)),
        },
        "dashboard": {
            "has_extra_dashboard_plots": isinstance(results_df, pd.DataFrame) and not results_df.empty,
            "dashboard_notes": (
                "Added real-world energy image, KPI cards, actual-vs-predicted demand curve, "
                "forecast error curve, residual distribution, actual-vs-predicted scatter plot, "
                "model comparison chart, average daily demand pattern, largest error table, and interpretation."
                if isinstance(results_df, pd.DataFrame) and not results_df.empty
                else ""
            ),
            "dashboard_elements": (
                [
                    "real-world energy image",
                    "KPI cards",
                    "actual vs predicted curve",
                    "forecast error curve",
                    "residual distribution",
                    "actual vs predicted scatter plot",
                    "model comparison bar chart",
                    "average daily demand pattern",
                    "largest forecast error table",
                    "written interpretation",
                ]
                if isinstance(results_df, pd.DataFrame) and not results_df.empty
                else []
            ),
        },
        "presentation": {
            "insights": insights_text,
            "professional_summary": globals().get("professional_summary", ""),
        },
    }
    return evidence


def build_project_card(evidence: dict) -> str:
    project = evidence["project"]
    student = evidence["student"]
    data = evidence["data_evidence"]
    return f"""# Project Card — {project['title']}

## Student
- Name: {student['name']}
- ID: {student['id']}

## Goal
{project['goal']}

## Dataset
- Timestamp column: {project['timestamp_column']}
- Target column: {project['target_column']}
- Raw rows: {data['raw_rows']}
- Clean rows: {data['clean_rows']}
- Time coverage: {data['timestamp_min']} to {data['timestamp_max']}

## Forecast setup
- Horizon: {project['forecast_horizon']}
- Resampling: {project['resampling_rule']}

## Required discussions
### Missing timestamps / missing values
{data['missing_discussion']}

### Outliers
{data['outlier_discussion']}

### Resampling
{data['resampling_discussion']}

## Student insights
{evidence['presentation']['insights']}

## Links
- Deployed app: {student['deployed_url']}
- Repository: {student['repo_url']}
"""


def call_ai_grader(api_key: str, evidence_json: str):
    prompt = AI_GRADER_PROMPT_TEMPLATE.replace("<insert submission.json contents here>", evidence_json)
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit.io",
            "X-Title": "Mini Project B AI Grader",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def parse_ai_response(text: str):
    try:
        return json.loads(text), None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), None
        except Exception as exc:
            return None, f"Found JSON-like text but could not parse it: {exc}"

    return None, "Could not parse JSON from the AI response."


st.title("Mini Project B — Time-Series Forecasting Starter")
st.caption("This starter prepares data and evidence only. Students add models, metrics, and dashboard improvements.")

with st.sidebar:
    st.header("Student info")
    student_name = st.text_input("Student name", value="Ahmed Al Omairi")
    student_id = st.text_input("Student ID", value="PG12S2540512")
    deployed_url = st.text_input("Streamlit deployed URL")
    repo_url = st.text_input("GitHub repo URL")
    project_title = st.text_input("Project title", value="Electricity Demand Forecasting")
    project_goal = st.text_area(
        "Project goal",
        value="Forecast future electricity demand using timestamp-based features and student-added models.",
    )
    dataset_path = st.text_input("Dataset path", value="data/dataset_sample.csv")

st.header("1. Load dataset + preview + audit")
try:
    raw_df = load_dataset(dataset_path)
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

st.subheader("First 10 rows")
st.dataframe(raw_df.head(10), use_container_width=True)

st.subheader("Columns, dtypes, and missing values")
audit = audit_dataframe(raw_df)
st.dataframe(audit, use_container_width=True)

st.header("2. Timestamp and target selection")
cols = list(raw_df.columns)

default_timestamp_index = cols.index("timestamp") if "timestamp" in cols else 0
numeric_like = []
for col in cols:
    converted = pd.to_numeric(raw_df[col], errors="coerce")
    if converted.notna().mean() >= 0.5:
        numeric_like.append(col)

default_target = "electricity_demand_mw" if "electricity_demand_mw" in cols else (numeric_like[0] if numeric_like else cols[0])

timestamp_col = st.selectbox("Timestamp column", cols, index=default_timestamp_index)
target_col = st.selectbox("Target column", cols, index=cols.index(default_target))

ts_df = clean_time_series(raw_df, timestamp_col, target_col)

col1, col2, col3 = st.columns(3)
col1.metric("Clean rows", f"{len(ts_df):,}")
col2.metric("Start", str(ts_df[timestamp_col].min()) if len(ts_df) else "N/A")
col3.metric("End", str(ts_df[timestamp_col].max()) if len(ts_df) else "N/A")

if ts_df.empty:
    st.error("No usable rows after parsing timestamp and target.")
    st.stop()

st.header("3. Optional resampling + forecast horizon")
resampling_label_to_rule = {
    "No resampling": None,
    "Hourly mean": "h",
    "Daily mean": "D",
    "Weekly mean": "W",
    "Monthly mean": "MS",
}
resampling_choice = st.selectbox("Optional resampling", list(resampling_label_to_rule.keys()))
horizon = st.number_input("Forecast horizon, in rows after resampling", min_value=1, max_value=168, value=24, step=1)

prepared_df = resample_time_series(ts_df, timestamp_col, target_col, resampling_label_to_rule[resampling_choice])

st.subheader("Prepared time-series preview")
st.dataframe(prepared_df.head(10), use_container_width=True)

fig, ax = plt.subplots()
plot_df = prepared_df.tail(min(500, len(prepared_df)))
ax.plot(plot_df[timestamp_col], plot_df[target_col])
ax.set_title("Target over time")
ax.set_xlabel(timestamp_col)
ax.set_ylabel(target_col)
plt.xticks(rotation=30)
st.pyplot(fig)

st.header("4. Baseline feature table creation")
feature_df = make_baseline_feature_table(prepared_df, timestamp_col, target_col, int(horizon))

baseline_features = ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"]
X = feature_df[baseline_features]
y = feature_df["y_target"]

st.write(f"Feature table rows: {len(feature_df):,}")
st.dataframe(feature_df.head(20), use_container_width=True)

st.info(
    "X and y are prepared from baseline features only. "
    "Students must add modeling, metrics, and extra visuals in the sections below."
)

# Automatic data-quality evidence for the exported submission.json.
timestamp_diffs = ts_df[timestamp_col].diff().dropna()
if len(timestamp_diffs) > 0:
    inferred_step = timestamp_diffs.mode().iloc[0]
    inferred_frequency_text = str(inferred_step)
    expected_gap_threshold = inferred_step * 1.5
    time_gap_count = int((timestamp_diffs > expected_gap_threshold).sum())
else:
    inferred_step = pd.NaT
    inferred_frequency_text = "Not enough rows to infer"
    time_gap_count = 0

duplicate_timestamp_count = int(ts_df[timestamp_col].duplicated().sum())
invalid_rows_removed = int(len(raw_df) - len(ts_df))
target_missing_percent = float(pd.to_numeric(raw_df[target_col], errors="coerce").isna().mean() * 100)

target_series = ts_df[target_col].dropna()
if len(target_series) > 0:
    q1 = float(target_series.quantile(0.25))
    q3 = float(target_series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((target_series < lower_bound) | (target_series > upper_bound)).sum())
    outlier_percent = float(outlier_count / len(target_series) * 100)
    target_summary_text = (
        f"Target summary: min={target_series.min():,.2f}, max={target_series.max():,.2f}, "
        f"mean={target_series.mean():,.2f}, std={target_series.std():,.2f}."
    )
else:
    lower_bound = np.nan
    upper_bound = np.nan
    outlier_count = 0
    outlier_percent = 0.0
    target_summary_text = "Target summary unavailable."

data_quality_auto_summary = (
    f"Automatic checks: {invalid_rows_removed:,} invalid rows removed after timestamp/target parsing; "
    f"{duplicate_timestamp_count:,} duplicate timestamps found; inferred time step is {inferred_frequency_text}; "
    f"{time_gap_count:,} larger-than-expected timestamp gaps detected; target missing rate before cleaning is "
    f"{target_missing_percent:.2f}%."
)

outlier_auto_summary = (
    f"Automatic IQR check for {target_col}: {outlier_count:,} potential outliers "
    f"({outlier_percent:.2f}%) using bounds {lower_bound:,.2f} to {upper_bound:,.2f}. "
    "These values were not removed automatically because extreme electricity demand can be operationally meaningful."
)

resampling_auto_summary = (
    f"Selected option: {resampling_choice}. Prepared rows after optional resampling: {len(prepared_df):,}. "
    f"The forecast horizon is {int(horizon)} row(s) after resampling, so the target y is shifted forward by that horizon."
)

st.subheader("Automatic data-quality evidence")
st.write(data_quality_auto_summary)
st.write(outlier_auto_summary)
st.write(resampling_auto_summary)

st.header("5. STUDENT ADDITIONS — MODELING")
st.markdown("Add your model training, time-based split, predictions, and metrics table here.")
st.code(
    """
# Paste your modeling code below this marker.
# Required outcome for grading:
# results_df = pd.DataFrame([...]) with model names and metrics.
# Example columns: model, MAE, RMSE, MAPE
results_df = None
""",
    language="python",
)

# STUDENT ADDITIONS — MODELING START
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.subheader("Professional model training and evaluation")

if len(feature_df) < 250:
    st.warning(
        "The feature table is quite small after lag creation. "
        "Modeling needs more rows for reliable evaluation."
    )
    results_df = None
    predictions_df = pd.DataFrame()
    best_predictions_df = pd.DataFrame()
    best_model_name = ""
else:
    model_df = feature_df.copy()

    # Student-added lag and rolling features.
    model_df["lag_2"] = model_df[target_col].shift(2)
    model_df["lag_3"] = model_df[target_col].shift(3)
    model_df["lag_48"] = model_df[target_col].shift(48)
    model_df["lag_168"] = model_df[target_col].shift(168)
    model_df["rolling_mean_6"] = model_df[target_col].shift(1).rolling(6).mean()
    model_df["rolling_mean_12"] = model_df[target_col].shift(1).rolling(12).mean()
    model_df["rolling_std_24"] = model_df[target_col].shift(1).rolling(24).std()

    # Calendar and cyclical features help the model learn daily and monthly patterns.
    model_df["hour_sin"] = np.sin(2 * np.pi * model_df["hour"] / 24)
    model_df["hour_cos"] = np.cos(2 * np.pi * model_df["hour"] / 24)
    model_df["month_sin"] = np.sin(2 * np.pi * model_df["month"] / 12)
    model_df["month_cos"] = np.cos(2 * np.pi * model_df["month"] / 12)
    model_df["trend_index"] = np.arange(len(model_df))

    model_df = model_df.dropna().reset_index(drop=True)

    student_feature_cols = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_24",
        "lag_48",
        "lag_168",
        "rolling_mean_6",
        "rolling_mean_12",
        "rolling_mean_24",
        "rolling_std_24",
        "hour",
        "weekend",
        "month",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "trend_index",
    ]

    X_model = model_df[student_feature_cols]
    y_model = model_df["y_target"]

    # Chronological split: train on past records and test on future records.
    split_ratio = 0.80
    split_index = int(len(model_df) * split_ratio)

    X_train = X_model.iloc[:split_index]
    X_test = X_model.iloc[split_index:]
    y_train = y_model.iloc[:split_index]
    y_test = y_model.iloc[split_index:]

    test_time = model_df[timestamp_col].iloc[split_index:].reset_index(drop=True)
    actual_values = y_test.reset_index(drop=True)

    train_rows = int(len(X_train))
    test_rows = int(len(X_test))
    train_period = f"{model_df[timestamp_col].iloc[0]} to {model_df[timestamp_col].iloc[split_index - 1]}"
    test_period = f"{model_df[timestamp_col].iloc[split_index]} to {model_df[timestamp_col].iloc[-1]}"
    time_split_evidence = (
        f"Chronological 80/20 split used. Training period: {train_period}; "
        f"testing period: {test_period}. No random shuffling was used."
    )

    st.success(
        f"Time-based split completed: {train_rows:,} training rows "
        f"and {test_rows:,} testing rows."
    )
    st.caption(time_split_evidence)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.06,
            max_leaf_nodes=31,
            random_state=42,
        ),
    }

    def safe_mape(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = y_true != 0
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    results = []
    prediction_frames = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = safe_mape(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({
            "model": model_name,
            "MAE": round(float(mae), 3),
            "RMSE": round(float(rmse), 3),
            "MAPE": round(float(mape), 3),
            "R2": round(float(r2), 4),
        })

        prediction_frames.append(pd.DataFrame({
            timestamp_col: test_time,
            "actual": actual_values,
            "prediction": preds,
            "model": model_name,
            "absolute_error": np.abs(actual_values - preds),
        }))

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    best_model_name = str(results_df.iloc[0]["model"])
    best_predictions_df = predictions_df[predictions_df["model"] == best_model_name].copy()

    st.subheader("Model metrics table")
    st.dataframe(results_df, use_container_width=True)

    best_mae = float(results_df.iloc[0]["MAE"])
    best_rmse = float(results_df.iloc[0]["RMSE"])
    best_mape = float(results_df.iloc[0]["MAPE"])
    best_r2 = float(results_df.iloc[0]["R2"])
    best_model_metrics = {
        "MAE": best_mae,
        "RMSE": best_rmse,
        "MAPE": best_mape,
        "R2": best_r2,
    }
    has_prediction_table = True
    professional_summary = (
        f"Best model: {best_model_name}. Test performance: MAE={best_mae:,.2f}, "
        f"RMSE={best_rmse:,.2f}, MAPE={best_mape:.2f}%, R2={best_r2:.3f}. "
        "The model was evaluated using a future holdout period from the chronological time-based split."
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Best model", best_model_name)
    kpi2.metric("Best MAE", f"{best_mae:,.2f}")
    kpi3.metric("Best RMSE", f"{best_rmse:,.2f}")
    kpi4.metric("Best MAPE", f"{best_mape:.2f}%")

    st.caption(
        "Models are evaluated on future records only, using a chronological split. "
        "This is required for realistic time-series forecasting."
    )
# STUDENT ADDITIONS — MODELING END

st.header("6. STUDENT ADDITIONS — DASHBOARD")
st.markdown("Add extra plots, KPIs, interpretation, and dashboard improvements here.")
st.code(
    """
# Paste your dashboard code below this marker.
# Add visuals that explain your model results and forecast behavior.
""",
    language="python",
)

# STUDENT ADDITIONS — DASHBOARD START
st.subheader("Professional forecasting dashboard")

if results_df is not None and "best_predictions_df" in globals() and not best_predictions_df.empty:
    st.markdown(
        """
        <div style="
            padding: 22px;
            border-radius: 18px;
            background: linear-gradient(135deg, #0f172a, #1e3a8a, #0284c7);
            color: white;
            margin-bottom: 20px;
        ">
            <h3 style="margin-bottom: 8px;">Electricity Demand Control-Room View</h3>
            <p style="font-size: 16px; margin-bottom: 0;">
                This dashboard presents a practical energy forecasting workflow:
                actual demand, predicted demand, forecast error, model comparison,
                and daily demand behavior.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.image(
        "https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&w=1400&q=80",
        caption="Real-world electricity demand forecasting supports planning, reliability, and grid operations.",
        use_container_width=True,
    )

    st.markdown("### Forecast KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", best_model_name)
    c2.metric("MAE", f"{best_mae:,.2f}")
    c3.metric("RMSE", f"{best_rmse:,.2f}")
    c4.metric("R²", f"{best_r2:.3f}")

    st.markdown("### Actual vs predicted demand curve")
    plot_limit = st.slider(
        "Number of recent test points to display",
        min_value=50,
        max_value=max(50, min(1000, len(best_predictions_df))),
        value=min(300, len(best_predictions_df)),
        step=50,
    )

    curve_df = best_predictions_df.tail(plot_limit)

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(curve_df[timestamp_col], curve_df["actual"], label="Actual demand")
    ax1.plot(
        curve_df[timestamp_col],
        curve_df["prediction"],
        label=f"Predicted demand — {best_model_name}",
    )
    ax1.set_title("Actual vs Predicted Electricity Demand")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(target_col)
    ax1.legend()
    plt.xticks(rotation=30)
    st.pyplot(fig1)

    st.markdown("### Forecast error curve")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(curve_df[timestamp_col], curve_df["absolute_error"], label="Absolute error")
    ax2.set_title("Absolute Forecast Error Over Time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Absolute error")
    ax2.legend()
    plt.xticks(rotation=30)
    st.pyplot(fig2)

    st.markdown("### Residual distribution")
    residuals = curve_df["actual"] - curve_df["prediction"]
    fig_resid, ax_resid = plt.subplots(figsize=(9, 4))
    ax_resid.hist(residuals, bins=30)
    ax_resid.set_title("Residual Distribution")
    ax_resid.set_xlabel("Actual - Predicted")
    ax_resid.set_ylabel("Frequency")
    st.pyplot(fig_resid)

    st.markdown("### Actual vs predicted scatter")
    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 6))
    ax_scatter.scatter(curve_df["actual"], curve_df["prediction"], alpha=0.7)
    min_val = float(min(curve_df["actual"].min(), curve_df["prediction"].min()))
    max_val = float(max(curve_df["actual"].max(), curve_df["prediction"].max()))
    ax_scatter.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Perfect forecast line")
    ax_scatter.set_title("Actual vs Predicted Scatter")
    ax_scatter.set_xlabel("Actual demand")
    ax_scatter.set_ylabel("Predicted demand")
    ax_scatter.legend()
    st.pyplot(fig_scatter)

    st.markdown("### Model comparison")
    metric_choice = st.selectbox(
        "Choose comparison metric",
        ["MAE", "RMSE", "MAPE", "R2"],
        index=1,
    )

    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.bar(results_df["model"], results_df[metric_choice])
    ax3.set_title(f"Model Comparison by {metric_choice}")
    ax3.set_xlabel("Model")
    ax3.set_ylabel(metric_choice)
    plt.xticks(rotation=20)
    st.pyplot(fig3)

    st.markdown("### Average daily demand pattern")
    daily_pattern = prepared_df.copy()
    daily_pattern["hour_of_day"] = daily_pattern[timestamp_col].dt.hour
    hourly_avg = daily_pattern.groupby("hour_of_day")[target_col].mean().reset_index()

    fig4, ax4 = plt.subplots(figsize=(9, 4))
    ax4.plot(hourly_avg["hour_of_day"], hourly_avg[target_col], marker="o")
    ax4.set_title("Average Electricity Demand by Hour of Day")
    ax4.set_xlabel("Hour of day")
    ax4.set_ylabel(f"Average {target_col}")
    ax4.set_xticks(range(0, 24))
    st.pyplot(fig4)

    st.markdown("### Largest forecast errors")
    worst_errors = best_predictions_df.sort_values("absolute_error", ascending=False).head(10)
    st.dataframe(
        worst_errors[[timestamp_col, "actual", "prediction", "absolute_error"]],
        use_container_width=True,
    )

    st.markdown("### Interpretation")
    st.write(
        f"The best model is **{best_model_name}** with RMSE **{best_rmse:,.2f}** "
        f"and MAPE **{best_mape:.2f}%**. The actual-vs-predicted curve shows whether "
        "the model follows demand peaks and low-demand periods."
    )
    st.write(
        "The error curve highlights periods where the forecast is less accurate. "
        "These periods may represent unusual demand behavior, weather effects, holidays, "
        "or sudden operational changes."
    )
else:
    st.warning("Run the modeling section first so dashboard charts and KPIs can be displayed.")
# STUDENT ADDITIONS — DASHBOARD END

st.header("7. Export submission files")
missing_discussion = st.text_area(
    "Discuss missing timestamps / missing values",
    value=(
        "The timestamp column was parsed to datetime, invalid timestamps were removed, "
        "and the target was converted to numeric. The data was sorted chronologically before "
        "feature engineering and modeling."
    ),
)
outlier_discussion = st.text_area(
    "Discuss outliers",
    value=(
        "Outliers were reviewed using the target-over-time curve and forecast error table. "
        "Large errors are highlighted in the dashboard for further investigation."
    ),
)
resampling_discussion = st.text_area(
    "Discuss resampling",
    value=f"Selected resampling option: {resampling_choice}. Explain why this is appropriate for your forecast horizon.",
)
insights_text = st.text_area(
    "Insights and interpretation",
    value=(
        "The project compares several forecasting models using a chronological split. "
        "The dashboard explains model performance with metrics, prediction curves, error curves, "
        "daily demand patterns, and largest forecast errors."
    ),
)

submission = build_submission_json(
    student_name=student_name,
    student_id=student_id,
    deployed_url=deployed_url,
    repo_url=repo_url,
    project_title=project_title,
    project_goal=project_goal,
    timestamp_col=timestamp_col,
    target_col=target_col,
    horizon=int(horizon),
    resampling_rule=resampling_choice,
    raw_df=raw_df,
    ts_df=ts_df,
    feature_df=feature_df,
    results_df=results_df,
    insights_text=insights_text,
    missing_discussion=missing_discussion,
    outlier_discussion=outlier_discussion,
    resampling_discussion=resampling_discussion,
)

submission_json = json.dumps(submission, indent=2)
project_card_md = build_project_card(submission)

st.download_button(
    "Download submission.json",
    data=submission_json,
    file_name="submission.json",
    mime="application/json",
)

st.download_button(
    "Download project_card.md",
    data=project_card_md,
    file_name="project_card.md",
    mime="text/markdown",
)

with st.expander("Preview submission.json"):
    st.json(submission)

with st.expander("Preview project_card.md"):
    st.markdown(project_card_md)

st.header("8. AI grader (/80)")
st.warning(
    "Run the grader after adding your model, metrics table, dashboard visuals, and insights. "
    "The starter alone will receive a low score because results_df is None by default."
)

api_key = get_openrouter_api_key()
if st.button("Run AI grader"):
    if not api_key:
        st.error("Please provide an OpenRouter API key.")
    else:
        with st.spinner("Calling AI grader..."):
            try:
                raw_output = call_ai_grader(api_key, submission_json)
                parsed, parse_error = parse_ai_response(raw_output)
                if parsed is not None:
                    st.subheader("Parsed AI grade JSON")
                    st.json(parsed)
                else:
                    st.subheader("Raw AI grader output")
                    st.code(raw_output)
                    st.error(parse_error)
            except Exception as exc:
                st.error(f"AI grader failed: {exc}")
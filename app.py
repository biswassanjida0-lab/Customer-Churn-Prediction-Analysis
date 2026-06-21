import pandas as pd
import numpy as np
import streamlit as st
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

st.set_page_config(page_title="Telco Churn Studio", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37,99,235,0.20), transparent 30%),
            radial-gradient(circle at top right, rgba(6,182,212,0.18), transparent 26%),
            linear-gradient(135deg, #050914 0%, #07111f 48%, #0b1325 100%);
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07101f 0%, #0b1221 100%);
        border-right: 1px solid rgba(148,163,184,0.18);
        box-shadow: 8px 0 30px rgba(0,0,0,0.20);
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1560px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stHorizontalBlock"] > div {
        border-radius: 18px;
    }

    .hero {
        background: linear-gradient(135deg, rgba(30,64,175,.72), rgba(6,95,105,.58));
        border: 1px solid rgba(125,211,252,.22);
        border-radius: 20px;
        padding: 1.15rem 1.35rem;
        box-shadow: 0 18px 45px rgba(0,0,0,.32), inset 0 1px 0 rgba(255,255,255,.08);
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 900;
        color: #f8fafc;
        letter-spacing: -.03em;
    }

    .hero p {
        margin: .35rem 0 0 0;
        color: #dbeafe;
        font-size: .98rem;
    }

     .metric-card {
     background: linear-gradient(145deg, rgba(15,23,42,.96), rgba(8,13,27,.96));
     border: 1px solid rgba(125,211,252,.14);
     border-radius: 16px;
     padding: 0.7rem;
     min-height: 90px;
     box-shadow: 0 14px 34px rgba(0,0,0,.30), inset 0 1px 0 rgba(255,255,255,.045);
     }

    .metric-label {
        color: #7dd3fc;
        font-size: .9rem;
        font-weight: 700;
    }

    .metric-value {
        font-size: 1.75rem;
        font-weight: 900;
        margin-top: .22rem;
        color: #f8fafc;
        letter-spacing: -.02em;
    }

    .metric-sub {
        margin-top: .38rem;
        font-size: .82rem;
        color: #cbd5e1;
    }

    .risk-high { color: #fb7185; }
    .risk-medium { color: #fbbf24; }
    .risk-low { color: #34d399; }

    .pill {
        display: inline-block;
        padding: .32rem .72rem;
        border-radius: 999px;
        font-size: .78rem;
        font-weight: 700;
        margin: .15rem .25rem .15rem 0;
        background: rgba(15,23,42,.72);
        border: 1px solid rgba(148,163,184,.20);
        color: #f8fafc;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
    }

    .section-card {
        background: linear-gradient(145deg, rgba(15,23,42,.88), rgba(7,12,25,.92));
        border: 1px solid rgba(125,211,252,.13);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 14px 38px rgba(0,0,0,.25);
        margin-bottom: 1rem;
    }

    .summary-box {
        background: linear-gradient(145deg, rgba(15,23,42,.92), rgba(8,13,27,.88));
        border: 1px solid rgba(125,211,252,.16);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        color: #e5e7eb;
        min-height: 180px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
    }

    .summary-title {
        color: #f8fafc;
        font-weight: 900;
        font-size: 1rem;
    }

    .section-title {
        color: #f8fafc;
        font-weight: 900;
        font-size: 1.12rem;
        margin: .35rem 0 .55rem 0;
    }

    [data-testid="stMetricValue"], h1, h2, h3, h4, h5, h6, p, label, span {
        color: inherit;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148,163,184,.16);
        border-radius: 14px;
        overflow: hidden;
    }

    .stAlert {
        border-radius: 14px;
        background: rgba(16,185,129,.16);
        border: 1px solid rgba(52,211,153,.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return pd.read_csv("telco_synthetic_hybrid_100k.csv")


@st.cache_resource
def prepare_models(df):
    df_model = df.drop(columns=["SyntheticCustomerID", "CustomerComment"]).copy()
    df_model["Churn"] = df_model["Churn"].map({"No": 0, "Yes": 1})

    X = df_model.drop("Churn", axis=1)
    y = df_model["Churn"]
    X_encoded = pd.get_dummies(X, drop_first=True).astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    log_model = LogisticRegression(max_iter=2000)
    log_model.fit(X_train, y_train)

    xgb_model = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)

    y_pred_log = log_model.predict(X_test)
    y_prob_log = log_model.predict_proba(X_test)[:, 1]

    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    metrics = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "XGBoost"],
            "Accuracy": [
                accuracy_score(y_test, y_pred_log),
                accuracy_score(y_test, y_pred_xgb),
            ],
            "Precision": [
                precision_score(y_test, y_pred_log),
                precision_score(y_test, y_pred_xgb),
            ],
            "Recall": [
                recall_score(y_test, y_pred_log),
                recall_score(y_test, y_pred_xgb),
            ],
            "F1": [
                f1_score(y_test, y_pred_log),
                f1_score(y_test, y_pred_xgb),
            ],
            "ROC-AUC": [
                roc_auc_score(y_test, y_prob_log),
                roc_auc_score(y_test, y_prob_xgb),
            ],
        }
    ).round(3)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "xgb_model": xgb_model,
        "log_model": log_model,
        "metrics": metrics,
        "training_columns": X_encoded.columns.tolist(),
        "y_pred_log": y_pred_log,
        "y_prob_log": y_prob_log,
        "y_pred_xgb": y_pred_xgb,
        "y_prob_xgb": y_prob_xgb,
    }


def align_columns(input_df, training_columns):
    encoded = pd.get_dummies(input_df, drop_first=True)
    encoded = encoded.reindex(columns=training_columns, fill_value=0)
    return encoded.astype(np.float64)


def preset_values(name):
    if name == "Low Risk":
        return {
            "tenure": 48,
            "monthly": 45.0,
            "complaints": 0,
            "call_drop": 0.8,
            "ticket": 8.0,
            "recharge": 1,
            "nps": 9,
            "sentiment_score": 0.72,
            "contract": "Two year",
            "internet": "DSL",
            "tech_support": "Yes",
            "sentiment": "Positive",
            "l1": "Service Quality",
            "category": "Network",
        }
    if name == "Medium Risk":
        return {
            "tenure": 18,
            "monthly": 72.0,
            "complaints": 2,
            "call_drop": 2.4,
            "ticket": 18.0,
            "recharge": 3,
            "nps": 6,
            "sentiment_score": 0.02,
            "contract": "One year",
            "internet": "Fiber optic",
            "tech_support": "No",
            "sentiment": "Neutral",
            "l1": "Pricing",
            "category": "Pricing",
        }
    if name == "High Risk":
        return {
            "tenure": 4,
            "monthly": 108.0,
            "complaints": 6,
            "call_drop": 6.4,
            "ticket": 39.0,
            "recharge": 8,
            "nps": 2,
            "sentiment_score": -0.68,
            "contract": "Month-to-month",
            "internet": "Fiber optic",
            "tech_support": "No",
            "sentiment": "Negative",
            "l1": "Billing Problems",
            "category": "Billing",
        }
    return None


def render_metric_card(label, value, subtitle, risk_class=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {risk_class}">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_gauge(probability, risk):
    color = "#ef4444" if risk == "High" else "#fbbf24" if risk == "Medium" else "#34d399"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={
                "suffix": "%",
                "font": {"size": 38, "color": "white"},
            },
            title={
                "text": "Churn Probability",
                "font": {"size": 18, "color": "#cbd5e1"},
            },
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#94a3b8",
                    "tickfont": {"color": "#cbd5e1", "size": 11},
                },
                "bar": {"color": color, "thickness": 0.35},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(52,211,153,0.18)"},
                    {"range": [40, 70], "color": "rgba(251,191,36,0.18)"},
                    {"range": [70, 100], "color": "rgba(248,113,113,0.18)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": probability * 100,
                },
            },
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def get_top_shap_features(model, input_encoded, top_n=6):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_encoded)
    vals = np.array(shap_values[0])
    abs_idx = np.argsort(np.abs(vals))[::-1][:top_n]
    top_df = pd.DataFrame(
        {
            "Feature": input_encoded.columns[abs_idx],
            "SHAP Impact": vals[abs_idx],
            "AbsImpact": np.abs(vals[abs_idx]),
        }
    )
    return explainer, shap_values, top_df


def plot_waterfall(explainer, shap_values, input_encoded):
    fig = plt.figure(figsize=(9, 4.8))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[0],
        feature_names=input_encoded.columns.tolist(),
        features=input_encoded.iloc[0],
        max_display=8,
        show=False,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_confusion_matrix_compact(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_roc_curve(y_true, y_prob_log, y_prob_xgb):
    fpr_log, tpr_log, _ = roc_curve(y_true, y_prob_log)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_true, y_prob_xgb)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(fpr_log, tpr_log, linewidth=2, label="Logistic Regression")
    ax.plot(fpr_xgb, tpr_xgb, linewidth=2, label="XGBoost")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def main():
    df = load_data()
    bundle = prepare_models(df)

    xgb_model = bundle["xgb_model"]
    training_columns = bundle["training_columns"]
    metrics = bundle["metrics"]
    y_test = bundle["y_test"]
    y_pred_log = bundle["y_pred_log"]
    y_pred_xgb = bundle["y_pred_xgb"]
    y_prob_log = bundle["y_prob_log"]
    y_prob_xgb = bundle["y_prob_xgb"]

    st.markdown(
        """
        <div class="hero">
            <h1>📉 Telecom Churn Intelligence Studio</h1>
            <p>Single-page demo for churn prediction, explainability, model evaluation, and CX insights using a hybrid synthetic telecom dataset.</p>
            <div style="margin-top:.7rem;">
                <span class="pill">100K rows</span>
                <span class="pill">XGBoost + Logistic</span>
                <span class="pill">SHAP explainability</span>
                <span class="pill">CX features: NPS, sentiment, L1 driver</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Customer Controls")
        st.caption("Adjust values manually or load a preset customer profile.")

        scenario = st.radio("Customer Preset", ["Custom", "Low Risk", "Medium Risk", "High Risk"])
        preset = preset_values(scenario) if scenario != "Custom" else None

        contract_options = sorted(df["Contract"].unique())
        internet_options = sorted(df["InternetService"].unique())
        tech_support_options = sorted(df["TechSupport"].unique())
        sentiment_options = sorted(df["Sentiment"].unique())
        l1_options = sorted(df["L1Driver"].unique())
        category_options = sorted(df["CommentCategory"].unique())

        tenure = st.slider("Tenure", 0, 72, preset["tenure"] if preset else 12)
        monthly = st.slider("Monthly Charges", 18.0, 130.0, preset["monthly"] if preset else 75.0)
        complaints = st.slider("Complaint Count", 0, 12, preset["complaints"] if preset else 2)
        call_drop = st.slider("Call Drop Rate", 0.5, 8.5, preset["call_drop"] if preset else 2.5)
        ticket = st.slider("Ticket Resolution Time", 4.0, 60.0, preset["ticket"] if preset else 20.0)
        recharge = st.slider("Recharge Delay Days", 0, 12, preset["recharge"] if preset else 2)
        nps = st.slider("NPS Rating", 0, 10, preset["nps"] if preset else 7)
        sentiment_score = st.slider("Sentiment Score", -0.95, 0.95, preset["sentiment_score"] if preset else 0.10)

        contract = st.selectbox(
            "Contract", contract_options, index=contract_options.index(preset["contract"]) if preset else 0
        )
        internet = st.selectbox(
            "Internet Service", internet_options, index=internet_options.index(preset["internet"]) if preset else 0
        )
        tech_support = st.selectbox(
            "Tech Support", tech_support_options, index=tech_support_options.index(preset["tech_support"]) if preset else 0
        )
        sentiment = st.selectbox(
            "Sentiment", sentiment_options, index=sentiment_options.index(preset["sentiment"]) if preset else 0
        )
        l1 = st.selectbox(
            "L1 Driver", l1_options, index=l1_options.index(preset["l1"]) if preset else 0
        )
        category = st.selectbox(
            "Comment Category", category_options, index=category_options.index(preset["category"]) if preset else 0
        )

    input_row = pd.DataFrame([
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": tenure,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": internet,
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": tech_support,
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": contract,
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": monthly,
            "TotalCharges": float(tenure * monthly),
            "Region": "North",
            "CityTier": "Tier 1",
            "CustomerSegment": "Mass",
            "PlanType": "Standard",
            "NetworkType": "4G",
            "ComplaintCount": complaints,
            "CallDropRate": call_drop,
            "TicketResolutionTime": ticket,
            "RechargeDelayDays": recharge,
            "NPSRating": nps,
            "NPSCategory": "Promoter" if nps >= 9 else "Passive" if nps >= 7 else "Detractor",
            "Sentiment": sentiment,
            "SentimentScore": sentiment_score,
            "CommentCategory": category,
            "L1Driver": l1,
        }
    ])

    input_encoded = align_columns(input_row, training_columns)
    probability = float(xgb_model.predict_proba(input_encoded)[0, 1])
    prediction = "Churn" if probability >= 0.5 else "No Churn"
    risk = "High" if probability >= 0.7 else "Medium" if probability >= 0.4 else "Low"
    risk_class = "risk-high" if risk == "High" else "risk-medium" if risk == "Medium" else "risk-low"

    explainer, shap_values, top_shap = get_top_shap_features(xgb_model, input_encoded)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_metric_card("Prediction", prediction, "Final model output", risk_class)
    with m2:
        render_metric_card("Churn Probability", f"{probability:.2%}", "Predicted risk score")
    with m3:
        render_metric_card("Risk Level", risk, "Based on model threshold", risk_class)
    with m4:
        best_auc = metrics.loc[metrics["Model"] == "XGBoost", "ROC-AUC"].iloc[0]
        render_metric_card("Model ROC-AUC", f"{best_auc:.3f}", "XGBoost test performance")
    with m5:
        best_acc = metrics.loc[metrics["Model"] == "XGBoost", "Accuracy"].iloc[0]
        render_metric_card("Model Accuracy", f"{best_acc:.3f}", "XGBoost test accuracy")

    st.markdown('<div class="section-title">Live Prediction Workspace</div>', unsafe_allow_html=True)

    g1, g2 = st.columns([1.0, 1.0])
    with g1:
        probability_gauge(probability, risk)

    with g2:
        st.markdown(
            f"""
            <div style="margin-bottom:.8rem;">
                <span class="pill">Preset: {scenario}</span>
                <span class="pill">Contract: {contract}</span>
                <span class="pill">Sentiment: {sentiment}</span>
                <span class="pill">L1 Driver: {l1}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        pred_color = "#22c55e" if prediction == "No Churn" else "#ef4444"

        if risk == "Low":
          risk_color = "#22c55e"
        elif risk == "Medium":
          risk_color = "#facc15"
        else:
          risk_color = "#ef4444"

        if probability < 0.40:
          prob_color = "#22c55e"
        elif probability < 0.70:
          prob_color = "#facc15"
        else:
          prob_color = "#ef4444"

        st.markdown(
            f"""
            <div class="summary-box">
                <span class="summary-title">Live summary</span><br><br>
                This customer is currently classified as <b style="color:{pred_color};">{prediction}</b> with a churn probability of
                <b style="color:{prob_color};">{probability:.2%}</b>. The present risk level is <b style="color:{risk_color};">{risk}</b>, influenced by service quality,
                complaint behavior, customer satisfaction, and contract conditions.
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns([1, 1.15])

    with left:
        st.write("**Customer profile**")
        st.dataframe(input_row.T, use_container_width=True, height=640)

    with right:
        st.write("**Why this customer may churn**")
        plot_waterfall(explainer, shap_values, input_encoded)
        st.write("**Top local drivers**")
        top_shap_display = top_shap[["Feature", "SHAP Impact"]].copy()
        top_shap_display["Direction"] = np.where(
            top_shap_display["SHAP Impact"] > 0, "Pushes to churn", "Pushes away"
        )
        st.dataframe(top_shap_display.round(3), use_container_width=True, height=250)

    st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)

    p1, p2 = st.columns([0.9, 1.1])
    with p1:
        st.write("**Model comparison**")
        st.dataframe(metrics, use_container_width=True, height=110)
        best_model = "XGBoost"
        st.success(f"Selected final model: {best_model}")

    with p2:
        plot_roc_curve(y_test, y_prob_log, y_prob_xgb)

    p3, p4 = st.columns(2)
    with p3:
        plot_confusion_matrix_compact(y_test, y_pred_log, "Confusion Matrix - Logistic Regression")
    with p4:
        plot_confusion_matrix_compact(y_test, y_pred_xgb, "Confusion Matrix - XGBoost")

    st.markdown('<div class="section-title">Business & CX Insights</div>', unsafe_allow_html=True)
    # NPS Score Calculation
    promoter_pct = (df["NPSCategory"] == "Promoter").mean() * 100
    detractor_pct = (df["NPSCategory"] == "Detractor").mean() * 100
    nps_score = promoter_pct - detractor_pct

    nps_class = (
      "risk-low"
      if nps_score >= 30
      else "risk-medium"
      if nps_score >= 0
      else "risk-high"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
      render_metric_card(
        "Promoters",
        f"{promoter_pct:.1f}%",
        "Rating 9-10",
        "risk-low"
    )

    with col2:
     render_metric_card(
        "Detractors",
        f"{detractor_pct:.1f}%",
        "Rating 0-6",
        "risk-high"
    )

    with col3:
      render_metric_card(
        "NPS Score",
        f"{nps_score:.1f}",
        "Promoters % - Detractors %",
        nps_class
    )
    
    i1, i2, i3 = st.columns(3)
    with i1:
        churn_by_sentiment = pd.crosstab(df["Sentiment"], df["Churn"], normalize="index") * 100
        st.write("**Churn % by Sentiment**")
        st.bar_chart(churn_by_sentiment["Yes"])
    with i2:
        churn_by_nps = df.groupby("NPSCategory")["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
        st.write("**Churn % by NPS Category**")
        st.bar_chart(churn_by_nps)
    with i3:
        churn_by_driver = pd.crosstab(df["L1Driver"], df["Churn"], normalize="index") * 100
        st.write("**Churn % by L1 Driver**")
        st.bar_chart(churn_by_driver["Yes"])

    i4, i5 = st.columns([1, 1])
    with i4:
        avg_metrics = (
            df.groupby("Churn")[["ComplaintCount", "CallDropRate", "TicketResolutionTime", "NPSRating"]]
            .mean()
            .round(2)
        )
        st.write("**Average metrics by churn group**")
        st.dataframe(avg_metrics, use_container_width=True)
    with i5:
        st.write("**Executive summary**")
        st.markdown(
            """
            - Higher **call drop rate**, **resolution time**, and **complaint count** strongly increase churn risk.
            - Lower **NPS** and weaker **sentiment score** are clear signs of dissatisfaction.
            - **XGBoost** performs best because it captures complex interactions between telecom service quality and customer experience variables.
            - The app supports both **live customer scoring** and **business insight storytelling**.
            """
        )

    with st.expander("Dataset Preview"):
        st.dataframe(df.head(50), use_container_width=True, height=420)


if __name__ == "__main__":

    main() 
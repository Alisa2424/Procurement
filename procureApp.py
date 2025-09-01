# Supplier Selection & Sustainability Dashboard
# Run in Google Colab with: !streamlit run app.py

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------
# File Upload
# ------------------------------
st.set_page_config(page_title="Supplier Selection Dashboard", layout="wide")
st.title("🌍 Supplier Selection & Sustainability Dashboard")

uploaded_file = st.file_uploader("📂 Upload Supplier CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ------------------------------
    # Data Preprocessing
    # ------------------------------
    st.subheader("📊 Raw Supplier Data")
    st.dataframe(df.head())

    # Add simulated Risk Score (0 = high risk, 100 = low risk)
    if "Risk_Score" not in df.columns:
        np.random.seed(42)
        df["Risk_Score"] = np.random.randint(50, 100, df.shape[0])

    # Normalize metrics
    df["Carbon_Footprint_Score"] = 1 - (df["Carbon Footprint (tons/year)"] / df["Carbon Footprint (tons/year)"].max())
    df["Recycling_Score"] = df["Recycling Rate (%)"] / 100
    df["Emissions_Score"] = 1 - (df["Emissions Score (0-10)"] / 10)
    df["Policy_Score"] = df["Policy Rating (0-5)"] / 5
    df["Cost_Score"] = 1 - (df["Cost ($/unit)"] / df["Cost ($/unit)"].max())
    df["LeadTime_Score"] = 1 - (df["Lead Time (days)"] / df["Lead Time (days)"].max())
    df["Risk_Score_Norm"] = df["Risk_Score"] / 100

    # Composite Sustainability Score
    df["Sustainability_Score"] = (
        df["Carbon_Footprint_Score"] +
        df["Recycling_Score"] +
        df["Emissions_Score"] +
        df["Policy_Score"]
    ) / 4

    # Composite Decision Score (Sustainability + Cost + Lead Time + Risk)
    df["Decision_Score"] = (
        df["Sustainability_Score"] * 0.4 +
        df["Cost_Score"] * 0.25 +
        df["LeadTime_Score"] * 0.15 +
        df["Risk_Score_Norm"] * 0.2
    )

    # ------------------------------
    # Filters
    # ------------------------------
    st.sidebar.header("🔍 Filter Options")

    industry_filter = st.sidebar.multiselect("Select Industry", options=df["Industry"].unique(), default=df["Industry"].unique())
    cost_filter = st.sidebar.slider("Max Cost ($/unit)", float(df["Cost ($/unit)"].min()), float(df["Cost ($/unit)"].max()), float(df["Cost ($/unit)"].max()))

    filtered_df = df[(df["Industry"].isin(industry_filter)) & (df["Cost ($/unit)"] <= cost_filter)]

    # ------------------------------
    # Top Suppliers
    # ------------------------------
    st.subheader("🏆 Top Suppliers by Decision Score")
    top_suppliers = filtered_df.sort_values("Decision_Score", ascending=False).head(10)
    st.dataframe(top_suppliers[["Supplier", "Industry", "Decision_Score", "Sustainability_Score", "Cost ($/unit)", "Lead Time (days)", "Risk_Score"]])

    # ------------------------------
    # Visualizations
    # ------------------------------
    st.subheader("📈 Visual Insights")

    # Decision Score by Supplier
    fig1 = px.bar(top_suppliers, x="Supplier", y="Decision_Score", color="Industry", title="Decision Score by Supplier")
    st.plotly_chart(fig1, use_container_width=True)

    # Radar Chart for Single Supplier
    st.subheader("📊 Supplier Performance Radar")
    supplier_choice = st.selectbox("Choose a Supplier", options=df["Supplier"].unique())
    supplier_data = df[df["Supplier"] == supplier_choice].iloc[0]

    categories = ["Sustainability_Score", "Cost_Score", "LeadTime_Score", "Risk_Score_Norm"]
    radar_values = [supplier_data[c] for c in categories]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatterpolar(
        r=radar_values,
        theta=categories,
        fill='toself',
        name=supplier_choice
    ))
    fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # Scenario Analysis
    st.subheader("🔮 Scenario Analysis")
    supplier_a = st.selectbox("Supplier A", options=df["Supplier"].unique(), index=0)
    supplier_b = st.selectbox("Supplier B", options=df["Supplier"].unique(), index=1)

    scenario_df = df[df["Supplier"].isin([supplier_a, supplier_b])]
    fig3 = px.bar(scenario_df, x="Supplier", y=["Sustainability_Score", "Cost_Score", "LeadTime_Score", "Risk_Score_Norm"], barmode="group", title="Supplier Comparison")
    st.plotly_chart(fig3, use_container_width=True)

    # ------------------------------
    # Download Option
    # ------------------------------
    st.subheader("⬇️ Download Results")
    csv = top_suppliers.to_csv(index=False).encode('utf-8')
    st.download_button("Download Top Suppliers CSV", csv, "top_suppliers.csv", "text/csv")

else:
    st.info("👆 Please upload a CSV file to get started.")

 

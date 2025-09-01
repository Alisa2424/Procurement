# app.py — Improved Sustainable Supplier Selection Tool
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="Sustainable Supplier Selection", page_icon="🌱", layout="wide")

# ---- Default sample data (small embedded CSV) ----
DEFAULT_CSV = """supplier_id,name,industry,location,carbon_footprint,recycling_rate,energy_efficiency,water_usage,waste_production,cost_element,lead_time_days,switching_cost,ISO_22000,ISO_14001,Fair_Trade,B_Corp,Organic
1,Supplier 1,Food,Asia,714.1167,29.5768,75.1662,251.4032,390.9132,5230.12,23.6,12387,0,0,0,0,0
2,Supplier 2,Food,Africa,148.4389,39.3172,69.2963,3556.8195,122.4677,4020.50,5.9,12181,0,1,0,0,0
3,Supplier 3,Electronics,Africa,298.3239,20.2386,88.6369,6627.1692,216.6789,9200.33,44.7,47465,0,0,0,0,1
4,Supplier 4,Textiles,North America,265.9346,48.5801,57.2374,2725.6397,234.4869,11111.0,37.6,17716,0,1,0,0,0
# ... (the full 50-row dataset can be used instead of this short example)
"""

# ---- Utilities ----
def safe_read_csv(file_like):
    """Read CSV from file-like object and return a DataFrame."""
    try:
        return pd.read_csv(file_like)
    except Exception:
        return pd.read_csv(StringIO(DEFAULT_CSV))

def ensure_columns(df):
    """Ensure expected columns exist and standardize names."""
    # normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # map possible variants to canonical names
    col_map = {}
    if 'cost_element' in df.columns:
        col_map['cost_element'] = 'cost_element'
    elif 'onboarding_cost_usd' in df.columns:
        col_map['onboarding_cost_usd'] = 'cost_element'  # map onboarding_cost_usd -> cost_element

    # switching cost variants
    if 'switching_cost' in df.columns:
        col_map['switching_cost'] = 'switching_cost'
    elif 'switching_cost_usd' in df.columns:
        col_map['switching_cost_usd'] = 'switching_cost'

    # lead time variants
    if 'lead_time_days' in df.columns:
        col_map['lead_time_days'] = 'lead_time_days'
    elif 'lead_time' in df.columns:
        col_map['lead_time'] = 'lead_time_days'

    df = df.rename(columns=col_map)

    # fill missing expected numeric columns with NaN (so normalization will handle it)
    expected_numeric = [
        'carbon_footprint','recycling_rate','energy_efficiency','water_usage','waste_production',
        'cost_element','lead_time_days','switching_cost'
    ]
    for col in expected_numeric:
        if col not in df.columns:
            df[col] = np.nan

    # expected certification columns
    cert_cols = ['iso_22000','iso_14001','fair_trade','b_corp','organic','rainforest_alliance']
    for c in cert_cols:
        if c not in df.columns:
            df[c] = 0

    return df

def minmax_normalize(series, invert=False):
    """Min-max normalize a pandas Series to 0-1. If invert=True, lower original is better."""
    # If constant or all NaN, return 0.5 neutral values
    if series.dropna().empty or series.max() == series.min():
        return pd.Series(0.5, index=series.index)
    norm = (series - series.min()) / (series.max() - series.min())
    return 1 - norm if invert else norm

# ---- Load data (uploaded file or default) ----
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        return safe_read_csv(StringIO(DEFAULT_CSV))
    else:
        return safe_read_csv(uploaded_file)

# ---- Composite scoring (vectorized) ----
def compute_composite_score(df, weights):
    """
    Compute a composite score (0-100) combining:
      - certifications (count / total)
      - cost_element (lower better)
      - carbon_footprint (lower better)
      - lead_time_days (lower better)
      - switching_cost (lower better)
      - recycling_rate (higher better)
      - energy_efficiency (higher better)
    `weights` is a dict of weights that should sum to 1 (we'll normalize just in case).
    """
    # Ensure canonical columns exist
    df = ensure_columns(df.copy())

    # Certification score: fraction of selected certifications present (we'll compute both cert_count and cert_frac)
    cert_columns = ['iso_22000','iso_14001','fair_trade','b_corp','organic','rainforest_alliance']
    df['cert_count'] = df[cert_columns].sum(axis=1)
    df['cert_frac'] = df['cert_count'] / len(cert_columns)

    # Normalize each metric to 0-1, with conversion so higher=better for all
    # invert=True for metrics where lower is better
    cost_norm = minmax_normalize(df['cost_element'], invert=True)
    carbon_norm = minmax_normalize(df['carbon_footprint'], invert=True)
    leadtime_norm = minmax_normalize(df['lead_time_days'], invert=True)
    switching_norm = minmax_normalize(df['switching_cost'], invert=True)
    recycling_norm = minmax_normalize(df['recycling_rate'], invert=False)
    energy_norm = minmax_normalize(df['energy_efficiency'], invert=False)
    cert_norm = minmax_normalize(df['cert_frac'], invert=False)  # already 0..1 but clip

    # Normalize weights (avoid division by zero)
    w = weights.copy()
    total_w = sum(w.values())
    if total_w == 0:
        # fallback to equal weights among selected keys
        n = len(w)
        w = {k: 1/n for k in w}
    else:
        w = {k: v/total_w for k,v in w.items()}

    # Weighted sum (vectorized)
    composite = (
        w.get('certifications', 0) * cert_norm +
        w.get('cost', 0) * cost_norm +
        w.get('carbon', 0) * carbon_norm +
        w.get('lead_time', 0) * leadtime_norm +
        w.get('switching_cost', 0) * switching_norm +
        w.get('recycling', 0) * recycling_norm +
        w.get('energy', 0) * energy_norm
    )

    # Scale to 0-100
    df['composite_score'] = (composite * 100).round(2)

    # Keep normalized component columns for UI / debugging
    df['_norm_cost'] = cost_norm
    df['_norm_carbon'] = carbon_norm
    df['_norm_leadtime'] = leadtime_norm
    df['_norm_switching'] = switching_norm
    df['_norm_recycling'] = recycling_norm
    df['_norm_energy'] = energy_norm
    df['_norm_cert'] = cert_norm

    return df

# ---- Streamlit UI ----
def main():
    st.title("🌱 Sustainable Supplier Selection — Improved")
    st.markdown("Choose certifications first, then industry & location. Adjust weights to compute a composite score that balances sustainability and operational/cost factors.")

    # Upload or use default dataset
    uploaded_file = st.sidebar.file_uploader("Upload supplier CSV (optional)", type=["csv"])
    df = load_data(uploaded_file)
    df = ensure_columns(df)

    # Sidebar: Certifications (shown first)
    st.sidebar.header("1) Certifications (choose zero or more)")
    cert_options = {
        'ISO 22000': 'iso_22000',
        'ISO 14001': 'iso_14001',
        'Fair Trade': 'fair_trade',
        'B Corp': 'b_corp',
        'Organic': 'organic',
        'Rainforest Alliance': 'rainforest_alliance'
    }
    selected_certs = st.sidebar.multiselect("Filter by certification(s) — suppliers must have at least one selected", options=list(cert_options.keys()))

    # Sidebar: Industry & Location (second)
    st.sidebar.header("2) Industry & Location")
    industries = sorted(df['industry'].dropna().unique())
    locations = sorted(df['location'].dropna().unique())
    selected_industries = st.sidebar.multiselect("Select Industry", options=industries, default=industries)
    selected_locations = st.sidebar.multiselect("Select Location", options=locations, default=locations)

    # Sidebar: Weights for composite scoring (third)
    st.sidebar.header("3) Composite Score Weights (tune to your preference)")
    w_cert = st.sidebar.slider("Certifications weight", 0.0, 1.0, 0.15, 0.01)
    w_cost = st.sidebar.slider("Cost weight (lower = better)", 0.0, 1.0, 0.20, 0.01)
    w_carbon = st.sidebar.slider("Carbon weight (lower = better)", 0.0, 1.0, 0.20, 0.01)
    w_lead = st.sidebar.slider("Lead time weight (lower = better)", 0.0, 1.0, 0.10, 0.01)
    w_switch = st.sidebar.slider("Switching cost weight (lower = better)", 0.0, 1.0, 0.10, 0.01)
    w_recycle = st.sidebar.slider("Recycling rate weight", 0.0, 1.0, 0.15, 0.01)
    w_energy = st.sidebar.slider("Energy efficiency weight", 0.0, 1.0, 0.10, 0.01)

    weights = {
        'certifications': w_cert,
        'cost': w_cost,
        'carbon': w_carbon,
        'lead_time': w_lead,
        'switching_cost': w_switch,
        'recycling': w_recycle,
        'energy': w_energy
    }

    # Apply filters
    filtered = df.copy()

    # Certification filter logic:
    # If user selected certificates: keep suppliers with ANY of the selected certs
    if selected_certs:
        cert_cols = [cert_options[name] for name in selected_certs]
        # create mask if any cert column is 1
        cert_mask = filtered[cert_cols].sum(axis=1) >= 1
        filtered = filtered[cert_mask]

    # Industry & location filters
    filtered = filtered[filtered['industry'].isin(selected_industries)]
    filtered = filtered[filtered['location'].isin(selected_locations)]

    # Guard empty result
    if filtered.empty:
        st.sidebar.error("No suppliers match the selected filters. Adjust filters.")
        st.stop()

    # Compute composite score vectorized
    scored = compute_composite_score(filtered, weights)

    # Sorting options
    st.sidebar.header("4) Sort / Select Top Suppliers")
    sort_option = st.sidebar.selectbox("Sort by", options=[
        "Composite Score (Custom Weights)",
        "Cert Count (High to Low)",
        "Cost (Low to High)",
        "Carbon (Low to High)",
        "Lead Time (Low to High)",
        "Switching Cost (Low to High)",
        "Recycling (High to Low)",
        "Energy Efficiency (High to Low)"
    ])
    top_n = st.sidebar.slider("Number of top suppliers to show", 3, 50, 10)

    if sort_option == "Composite Score (Custom Weights)":
        scored = scored.sort_values('composite_score', ascending=False)
    elif sort_option == "Cert Count (High to Low)":
        scored = scored.sort_values('cert_count', ascending=False)
    elif sort_option == "Cost (Low to High)":
        scored = scored.sort_values('cost_element', ascending=True)
    elif sort_option == "Carbon (Low to High)":
        scored = scored.sort_values('carbon_footprint', ascending=True)
    elif sort_option == "Lead Time (Low to High)":
        scored = scored.sort_values('lead_time_days', ascending=True)
    elif sort_option == "Switching Cost (Low to High)":
        scored = scored.sort_values('switching_cost', ascending=True)
    elif sort_option == "Recycling (High to Low)":
        scored = scored.sort_values('recycling_rate', ascending=False)
    elif sort_option == "Energy Efficiency (High to Low)":
        scored = scored.sort_values('energy_efficiency', ascending=False)

    top_suppliers = scored.head(top_n).reset_index(drop=True)

    # ---- Layout: main page ----
    st.header("Top Suppliers (filtered & ranked)")
    cols = st.columns([2, 1, 1])
    with cols[0]:
        display_cols = [
            'name','industry','location','composite_score','cert_count',
            'cost_element','carbon_footprint','lead_time_days','switching_cost',
            'recycling_rate','energy_efficiency'
        ]
        st.dataframe(top_suppliers[display_cols].rename(columns={
            'name':'Supplier','composite_score':'Composite Score','cert_count':'Cert Count',
            'cost_element':'Cost','carbon_footprint':'Carbon','lead_time_days':'Lead Time',
            'switching_cost':'Switching Cost','recycling_rate':'Recycling %','energy_efficiency':'Energy %'
        }), use_container_width=True)

    with cols[1]:
        fig = px.histogram(scored, x='composite_score', nbins=20, title='Composite Score Distribution')
        st.plotly_chart(fig, use_container_width=True)

    with cols[2]:
        industry_mean = scored.groupby('industry')['composite_score'].mean().reset_index()
        fig2 = px.bar(industry_mean, x='industry', y='composite_score', title='Avg Composite Score by Industry')
        st.plotly_chart(fig2, use_container_width=True)

    # Detail and scenario comparison
    st.header("Compare Suppliers (detailed view)")
    supplier_list = top_suppliers['name'].tolist()
    if not supplier_list:
        st.info("No suppliers to compare.")
        return

    s1, s2 = st.columns(2)
    with s1:
        current = st.selectbox("Current Supplier", options=supplier_list, index=0, key='cur')
    with s2:
        alternative = st.selectbox("Alternative Supplier", options=supplier_list, index=min(1, len(supplier_list)-1), key='alt')

    if current and alternative:
        cur_row = scored[scored['name'] == current].iloc[0]
        alt_row = scored[scored['name'] == alternative].iloc[0]

        # Comparison metrics
        compare_metrics = [
            ('Composite Score', 'composite_score'),
            ('Cert Count', 'cert_count'),
            ('Cost (lower better)', 'cost_element'),
            ('Carbon (lower better)', 'carbon_footprint'),
            ('Lead Time (lower better)', 'lead_time_days'),
            ('Switching Cost (lower better)', 'switching_cost'),
            ('Recycling % (higher better)', 'recycling_rate'),
            ('Energy % (higher better)', 'energy_efficiency')
        ]

        comp_x = [m[0] for m in compare_metrics]
        cur_vals = [cur_row[m[1]] for m in compare_metrics]
        alt_vals = [alt_row[m[1]] for m in compare_metrics]

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Current', x=comp_x, y=cur_vals))
        fig.add_trace(go.Bar(name='Alternative', x=comp_x, y=alt_vals))
        fig.update_layout(barmode='group', title='Supplier Comparison (Current vs Alternative)', xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Simple recommendation logic: prefer supplier with higher composite_score
        if alt_row['composite_score'] > cur_row['composite_score']:
            st.success(f"Alternative ({alternative}) has higher composite score ({alt_row['composite_score']}) than current ({cur_row['composite_score']}) — consider switching.")
        elif alt_row['composite_score'] < cur_row['composite_score']:
            st.info(f"Current supplier ({current}) scores higher ({cur_row['composite_score']}) than alternative ({alt_row['composite_score']}).")
        else:
            st.warning("Both suppliers have similar composite scores. Consider qualitative factors.")

    # Footer: show explanation of composite score
    st.markdown("---")
    st.subheader("How Composite Score is calculated")
    st.markdown("""
    Composite Score is a weighted combination of:
    - **Certifications** (fraction of certifications present),
    - **Cost** (lower is better),
    - **Carbon footprint** (lower is better),
    - **Lead time** (lower is better),
    - **Switching cost** (lower is better),
    - **Recycling rate** (higher is better),
    - **Energy efficiency** (higher is better).
    \nAdjust the sliders on the left to change the weights. We normalize each metric (min-max) and combine them so the final score is on a 0–100 scale.
    """)

if __name__ == "__main__":
    main()

 

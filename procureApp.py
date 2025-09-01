 import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Sustainable Supplier Selection",
    page_icon="🌱",
    layout="wide"
)

def generate_sample_data(num_suppliers=50):
    np.random.seed(42)
    data = {
        'supplier_id': range(1, num_suppliers + 1),
        'name': [f'Supplier {i}' for i in range(1, num_suppliers + 1)],
        'carbon_footprint': np.random.uniform(100, 1000, num_suppliers),
        'recycling_rate': np.random.uniform(20, 95, num_suppliers),
        'energy_efficiency': np.random.uniform(50, 95, num_suppliers),
        'water_usage': np.random.uniform(100, 10000, num_suppliers),
        'waste_production': np.random.uniform(10, 500, num_suppliers),
        'cost_element': np.random.uniform(1000, 10000, num_suppliers),
        'onboarding_cost_usd': np.random.uniform(1000, 8000, num_suppliers),
        'lead_time_days': np.random.uniform(5, 30, num_suppliers),
        'switching_cost_usd': np.random.uniform(500, 5000, num_suppliers),
    }
    certifications = ['ISO_22000', 'ISO_14001', 'Fair_Trade', 'B_Corp', 'Organic']
    for cert in certifications:
        data[cert] = np.random.choice([0, 1], size=num_suppliers, p=[0.6, 0.4])
    locations = ['North America', 'Europe', 'Asia', 'South America', 'Africa']
    industries = ['Electronics', 'Textiles', 'Food', 'Chemicals', 'Manufacturing']
    data['location'] = np.random.choice(locations, num_suppliers)
    data['industry'] = np.random.choice(industries, num_suppliers)
    df = pd.DataFrame(data)
    try:
        df.to_csv('supplier_data.csv', index=False)
    except Exception:
        pass
    return df

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('supplier_data.csv')
    except FileNotFoundError:
        df = generate_sample_data()
    return df

def calculate_sustainability_score(row, weights):
    cert_cols = ['ISO_22000', 'ISO_14001', 'Fair_Trade', 'B_Corp', 'Organic']
    existing_certs = [cert for cert in cert_cols if cert in row.index]
    cert_score = 0
    if existing_certs:
        cert_score = sum(row[cert] for cert in existing_certs) / len(existing_certs)

    normalized_carbon = 1 - (row['carbon_footprint'] / 1000)
    normalized_recycling = row['recycling_rate'] / 100
    normalized_energy = row['energy_efficiency'] / 100
    normalized_water = 1 - (row['water_usage'] / 10000)
    normalized_waste = 1 - (row['waste_production'] / 500)

    score = (
        weights['carbon'] * normalized_carbon +
        weights['recycling'] * normalized_recycling +
        weights['energy'] * normalized_energy +
        weights['water'] * normalized_water +
        weights['waste'] * normalized_waste +
        weights['certifications'] * cert_score
    )
    return round(score * 100, 2)

def calculate_scores(df, weights):
    df['sustainability_score'] = df.apply(lambda row: calculate_sustainability_score(row, weights), axis=1)
    return df.sort_values('sustainability_score', ascending=False)

def normalize_column(col, ascending=True):
    norm = (col - col.min()) / (col.max() - col.min())
    return norm if ascending else 1 - norm

def main():
    st.title("🌱 Sustainable Supplier Selection Tool")
    st.markdown("An AI-powered tool for evaluating and comparing suppliers based on sustainability metrics.")

    uploaded_file = st.file_uploader("Upload your supplier CSV file", type=["csv"])
    use_sample_data = st.checkbox("Use sample data instead", value=False)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        cert_cols = ['ISO_22000', 'ISO_14001', 'Fair_Trade', 'B_Corp', 'Organic']
        num_cols_defaults = {
            'cost_element': df['cost_element'].median() if 'cost_element' in df.columns else 5000,
            'onboarding_cost_usd': df['onboarding_cost_usd'].median() if 'onboarding_cost_usd' in df.columns else 3000,
            'lead_time_days': df['lead_time_days'].median() if 'lead_time_days' in df.columns else 15,
            'switching_cost_usd': df['switching_cost_usd'].median() if 'switching_cost_usd' in df.columns else 2000,
        }
        for cert in cert_cols:
            if cert not in df.columns:
                df[cert] = 0
        for col, default in num_cols_defaults.items():
            if col not in df.columns:
                df[col] = default
        st.success("Data loaded from uploaded file.")
    elif use_sample_data:
        df = generate_sample_data()
        st.info("Using generated sample data.")
    else:
        st.warning("Upload a supplier CSV file or tick 'Use sample data' to continue.")
        return

    cert_cols = ['ISO_22000', 'ISO_14001', 'Fair_Trade', 'B_Corp', 'Organic']

    st.sidebar.header("Filters")
    st.sidebar.subheader("Certifications")
    cert_filters = {}
    for cert in cert_cols:
        cert_filters[cert] = st.sidebar.checkbox(cert.replace('_', ' '), value=True)

    industries = st.sidebar.multiselect(
        "Industry", options=df['industry'].unique(), default=list(df['industry'].unique())
    )
    locations = st.sidebar.multiselect(
        "Location", options=df['location'].unique(), default=list(df['location'].unique())
    )

    st.sidebar.header("Lead Time and Cost Filters")

    min_lead_time, max_lead_time = st.sidebar.slider(
        "Select Lead Time Range (days)",
        float(df['lead_time_days'].min()),
        float(df['lead_time_days'].max()),
        (float(df['lead_time_days'].min()), float(df['lead_time_days'].max())),
    )

    min_onboarding_cost, max_onboarding_cost = st.sidebar.slider(
        "Select Onboarding Cost Range (USD)",
        float(df['onboarding_cost_usd'].min()),
        float(df['onboarding_cost_usd'].max()),
        (float(df['onboarding_cost_usd'].min()), float(df['onboarding_cost_usd'].max())),
    )

    min_switching_cost, max_switching_cost = st.sidebar.slider(
        "Select Switching Cost Range (USD)",
        float(df['switching_cost_usd'].min()),
        float(df['switching_cost_usd'].max()),
        (float(df['switching_cost_usd'].min()), float(df['switching_cost_usd'].max())),
    )

    st.sidebar.subheader("Sustainability Scoring Weights")
    weights = {
        'carbon': st.sidebar.slider("Carbon Footprint", 0.0, 0.3, 0.25, 0.05),
        'recycling': st.sidebar.slider("Recycling Rate", 0.0, 0.3, 0.15, 0.05),
        'energy': st.sidebar.slider("Energy Efficiency", 0.0, 0.3, 0.15, 0.05),
        'water': st.sidebar.slider("Water Usage", 0.0, 0.3, 0.15, 0.05),
        'waste': st.sidebar.slider("Waste Production", 0.0, 0.3, 0.15, 0.05),
        'certifications': st.sidebar.slider("Certifications", 0.0, 0.3, 0.15, 0.05),
    }
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        st.sidebar.warning(f"Weights sum to {total_weight:.2f} (should sum to 1.0)")

    filtered_df = df.copy()
    for cert, include in cert_filters.items():
        if not include:
            filtered_df = filtered_df[filtered_df[cert] == 0]
    filtered_df = filtered_df[filtered_df['industry'].isin(industries)]
    filtered_df = filtered_df[filtered_df['location'].isin(locations)]
    filtered_df = filtered_df[
        (filtered_df['lead_time_days'] >= min_lead_time)
        & (filtered_df['lead_time_days'] <= max_lead_time)
        & (filtered_df['onboarding_cost_usd'] >= min_onboarding_cost)
        & (filtered_df['onboarding_cost_usd'] <= max_onboarding_cost)
        & (filtered_df['switching_cost_usd'] >= min_switching_cost)
        & (filtered_df['switching_cost_usd'] <= max_switching_cost)
    ]

    scored_df = calculate_scores(filtered_df, weights)
    scored_df['cert_count'] = scored_df[cert_cols].sum(axis=1)

    st.sidebar.subheader("Rank suppliers by:")
    ranking_criteria = st.sidebar.selectbox(
        "Ranking Criteria",
        [
            "Sustainability Score",
            "Carbon Footprint (Low to High)",
            "Recycling Rate (High to Low)",
            "Energy Efficiency (High to Low)",
            "Water Usage (Low to High)",
            "Waste Production (Low to High)",
            "Lead Time (Low to High)",
            "Onboarding Cost (Low to High)",
            "Switching Cost (Low to High)",
            "Best Supplier (Lead Time + Cost)",
        ],
    )

    if ranking_criteria == "Sustainability Score":
        sorted_df = scored_df.sort_values('sustainability_score', ascending=False)
    elif ranking_criteria == "Best Supplier (Lead Time + Cost)":
        norm_lead = normalize_column(scored_df['lead_time_days'], ascending=True)
        norm_onboarding = normalize_column(scored_df['onboarding_cost_usd'], ascending=True)
        norm_switching = normalize_column(scored_df['switching_cost_usd'], ascending=True)
        scored_df['best_supplier_score'] = (norm_lead + norm_onboarding + norm_switching) / 3
        sorted_df = scored_df.sort_values('best_supplier_score', ascending=True)
    else:
        criterion_map = {
            "Carbon Footprint (Low to High)": ('carbon_footprint', True),
            "Recycling Rate (High to Low)": ('recycling_rate', False),
            "Energy Efficiency (High to Low)": ('energy_efficiency', False),
            "Water Usage (Low to High)": ('water_usage', True),
            "Waste Production (Low to High)": ('waste_production', True),
            "Lead Time (Low to High)": ('lead_time_days', True),
            "Onboarding Cost (Low to High)": ('onboarding_cost_usd', True),
            "Switching Cost (Low to High)": ('switching_cost_usd', True),
        }
        col, asc = criterion_map.get(ranking_criteria, ('sustainability_score', False))
        sorted_df = scored_df.sort_values(col, ascending=asc)

    top_suppliers = sorted_df.head(10)

    tabs = st.tabs(["📊 Dashboard", "🏆 Rankings", "📈 Trends", "🔮 Scenario Simulation"])

    with tabs[0]:
        st.subheader("📊 Dashboard")
        st.write("Overview and summary information can be displayed here.")

    with tabs[1]:
        st.subheader("🏆 Rankings")
        st.dataframe(
            top_suppliers[
                ['name', 'industry', 'location', 'sustainability_score', 'cert_count', 'carbon_footprint',
                 'cost_element', 'onboarding_cost_usd', 'lead_time_days', 'switching_cost_usd']
            ],
            use_container_width=True
        )

        # Supplier Details inside Rankings tab only
        st.header("Supplier Details")
        selected_supplier = st.selectbox("Select a supplier to view details", options=top_suppliers['name'].tolist())
        if selected_supplier:
            supplier_data = scored_df[scored_df['name'] == selected_supplier].iloc[0]
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sustainability Metrics")
                metrics = {
                    'Carbon Footprint': supplier_data['carbon_footprint'],
                    'Recycling Rate': supplier_data['recycling_rate'],
                    'Energy Efficiency': supplier_data['energy_efficiency'],
                    'Water Usage': supplier_data['water_usage'],
                    'Waste Production': supplier_data['waste_production'],
                    'Cost Element': supplier_data['cost_element'],
                    'Onboarding Cost': supplier_data['onboarding_cost_usd'],
                    'Lead Time (days)': supplier_data['lead_time_days'],
                    'Switching Cost': supplier_data['switching_cost_usd'],
                }
                for metric, value in metrics.items():
                    st.metric(metric, f"{value:.2f}")
            with col2:
                st.subheader("Certifications")
                certs = [cert.replace('_', ' ') for cert in cert_cols if supplier_data[cert] == 1]
                if certs:
                    for cert in certs:
                        st.success(cert)
                else:
                    st.warning("No certifications")

    with tabs[2]:
        st.subheader("📈 Trends")
        # Add plots to Trends tab: sustainability score distribution and industry average scores
        fig_dist = px.histogram(scored_df, x='sustainability_score', nbins=20, title="Sustainability Score Distribution")
        st.plotly_chart(fig_dist, use_container_width=True)

        industry_scores = scored_df.groupby('industry')['sustainability_score'].mean().reset_index()
        fig_industry = px.bar(industry_scores, x='industry', y='sustainability_score', title="Average Sustainability Score by Industry")
        st.plotly_chart(fig_industry, use_container_width=True)

    with tabs[3]:
        st.subheader("🔮 Scenario Simulation")
        col1, col2 = st.columns(2)
        with col1:
            current_supplier = st.selectbox(
                "Current Supplier",
                options=scored_df['name'].tolist(),
                key="current"
            )
        with col2:
            alternative_supplier = st.selectbox(
                "Alternative Supplier",
                options=scored_df['name'].tolist(),
                key="alternative"
            )
        if current_supplier and alternative_supplier and current_supplier != alternative_supplier:
            current = scored_df[scored_df['name'] == current_supplier].iloc[0]
            alternative = scored_df[scored_df['name'] == alternative_supplier].iloc[0]
            impact_diff = {
                'Carbon Footprint': alternative['carbon_footprint'] - current['carbon_footprint'],
                'Water Usage': alternative['water_usage'] - current['water_usage'],
                'Waste Production': alternative['waste_production'] - current['waste_production']
            }
            st.subheader("Environmental Impact Comparison")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Current',
                x=list(impact_diff.keys()),
                y=[current['carbon_footprint'], current['water_usage'], current['waste_production']],
                marker_color='blue'
            ))
            fig.add_trace(go.Bar(
                name='Alternative',
                x=list(impact_diff.keys()),
                y=[alternative['carbon_footprint'], alternative['water_usage'], alternative['waste_production']],
                marker_color='green'
            ))
            fig.update_layout(barmode='group', title_text="Environmental Impact Comparison")
            st.plotly_chart(fig, use_container_width=True)
            improvements = []
            declines = []
            for metric, diff in impact_diff.items():
                if diff < 0:
                    improvements.append(f"{metric}: {abs(diff):.2f} reduction")
                else:
                    declines.append(f"{metric}: {diff:.2f} increase")
            if improvements:
                st.success("Improvements: " + ", ".join(improvements))
            if declines:
                st.error("Declines: " + ", ".join(declines))
            if len(improvements) > len(declines):
                st.success("Recommendation: Consider switching to this supplier")
            elif len(improvements) < len(declines):
                st.error("Recommendation: Not recommended to switch to this supplier")
            else:
                st.warning("Recommendation: Mixed impact - consider other factors")
        elif current_supplier == alternative_supplier:
            st.warning("Please select different suppliers for comparison")

if __name__ == "__main__":
    main()


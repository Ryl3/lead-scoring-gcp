"""
Streamlit dashboard for lead scoring portfolio showcase.
Features:
- Lead scoring simulator
- Batch scoring from CSV
- Model performance metrics
- SHAP explainability viewer
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.core.scorer import LeadScorer, FEATURE_LABELS

# Page config
st.set_page_config(
    page_title="Lead Scoring System",
    page_icon="🎯",
    layout="wide",
)

# Initialize scorer
@st.cache_resource
def load_scorer():
    model_path = ROOT / "models" / "lead_scorer_v1.pkl"
    return LeadScorer(model_path)

try:
    scorer = load_scorer()
    model_ready = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model_ready = False

# Sidebar navigation
st.sidebar.title("🎯 Lead Scoring")
page = st.sidebar.radio("Navigation", [
    "📊 Dashboard",
    "🔮 Score Single Lead",
    "📁 Batch Score",
    "🔍 Model Insights",
])

# Tier colors
TIER_COLORS = {
    "Hot": "#E63946",
    "Warm": "#F4A261", 
    "Nurture": "#2A9D8F",
    "Suppress": "#ADB5BD",
}

if page == "📊 Dashboard":
    st.title("Lead Scoring Dashboard")
    
    if not model_ready:
        st.stop()
    
    # Load sample scored data
    sample_data_path = ROOT / "outputs" / "scores" / "stage5_final_scored_leads.csv"
    if sample_data_path.exists():
        df = pd.read_csv(sample_data_path)
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Leads", len(df))
        with col2:
            hot_count = len(df[df["tier"] == "Hot"])
            st.metric("Hot Leads", hot_count, f"{hot_count/len(df):.1%}")
        with col3:
            warm_count = len(df[df["tier"] == "Warm"])
            st.metric("Warm Leads", warm_count, f"{warm_count/len(df):.1%}")
        with col4:
            conv_rate = df["converted"].mean()
            st.metric("Avg Conversion Rate", f"{conv_rate:.1%}")
        
        # Tier distribution
        st.subheader("Lead Distribution by Tier")
        tier_counts = df["tier"].value_counts().reindex(["Hot", "Warm", "Nurture", "Suppress"])
        fig = go.Figure(data=[
            go.Bar(
                x=tier_counts.index,
                y=tier_counts.values,
                marker_color=[TIER_COLORS[t] for t in tier_counts.index],
                text=[f"{v} ({v/len(df):.1%})" for v in tier_counts.values],
                textposition="auto",
            )
        ])
        fig.update_layout(
            xaxis_title="Tier",
            yaxis_title="Number of Leads",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Score distribution
        st.subheader("Score Distribution")
        fig2 = px.histogram(
            df, 
            x="score_0_100",
            color="tier",
            color_discrete_map=TIER_COLORS,
            nbins=20,
            title="Score Distribution by Tier",
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Conversion rate by tier
        st.subheader("Conversion Rate by Tier")
        tier_conv = df.groupby("tier")["converted"].agg(["mean", "count"]).reset_index()
        tier_conv["mean"] = tier_conv["mean"] * 100
        tier_conv = tier_conv.sort_values("mean", ascending=False)
        
        fig3 = go.Figure(data=[
            go.Bar(
                x=tier_conv["tier"],
                y=tier_conv["mean"],
                marker_color=[TIER_COLORS[t] for t in tier_conv["tier"]],
                text=[f"{v:.1f}%" for v in tier_conv["mean"]],
                textposition="auto",
            )
        ])
        fig3.update_layout(
            xaxis_title="Tier",
            yaxis_title="Conversion Rate (%)",
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Sample leads table
        st.subheader("Sample Hot Leads")
        hot_leads = df[df["tier"] == "Hot"].head(10)[[
            "lead_id", "score_0_100", "lr_prob", "reason_1", "reason_2", "top_negative"
        ]]
        st.dataframe(hot_leads, use_container_width=True)
    else:
        st.info("No scored data available. Run batch scoring first.")

elif page == "🔮 Score Single Lead":
    st.title("Score a Single Lead")
    
    if not model_ready:
        st.stop()
    
    st.markdown("Enter lead features to get an instant score with AI-powered explanations.")
    
    # Feature input form
    with st.form("lead_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Identity & Firmographics")
            lead_id = st.text_input("Lead ID", value="lead_12345")
            is_decision_maker = st.selectbox("Decision Maker?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            is_smb_fit = st.selectbox("SMB Fit?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            seniority_rank = st.slider("Seniority Rank", 0, 5, 2)
            industry_tier = st.slider("Industry Tier", 1, 3, 2)
            company_size_score = st.slider("Company Size Score", 0, 4, 2)
            tech_stack_score = st.slider("Tech Stack Score", 0, 2, 1)
        
        with col2:
            st.subheader("Behavioral Signals")
            high_intent_touch_count = st.number_input("High Intent Touches", 0, 100, 3)
            visited_pricing_or_demo = st.selectbox("Visited Pricing/Demo?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            cross_channel_engaged = st.selectbox("Cross-Channel Engaged?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            pricing_page_visits_decayed = st.slider("Pricing Page Visits (decayed)", 0.0, 10.0, 2.5)
            demo_page_visits_decayed = st.slider("Demo Page Visits (decayed)", 0.0, 10.0, 1.0)
            content_engagement_decayed = st.slider("Content Engagement (decayed)", 0.0, 10.0, 3.0)
            open_rate = st.slider("Email Open Rate", 0.0, 1.0, 0.4)
        
        # Advanced features toggle
        with st.expander("Advanced Features"):
            col3, col4 = st.columns(2)
            with col3:
                email_clicks_decayed = st.slider("Email Clicks (decayed)", 0.0, 10.0, 1.0)
                email_opens_decayed = st.slider("Email Opens (decayed)", 0.0, 10.0, 2.0)
                case_study_views_decayed = st.slider("Case Study Views (decayed)", 0.0, 10.0, 0.5)
                trial_depth_score = st.slider("Trial Depth Score", 0.0, 10.0, 2.0)
                is_engaged_trial = st.selectbox("Engaged Trial?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            with col4:
                visited_last_7_days = st.selectbox("Visited Last 7 Days?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                pages_per_session = st.slider("Pages per Session", 0.0, 20.0, 5.0)
                roi_calculator_uses_decayed = st.slider("ROI Calculator Uses", 0.0, 5.0, 0.0)
                blog_views_decayed = st.slider("Blog Views (decayed)", 0.0, 10.0, 1.0)
                lead_source_tier = st.slider("Lead Source Tier", 0, 3, 2)
        
        submitted = st.form_submit_button("🎯 Score Lead", use_container_width=True)
    
    if submitted:
        # Build features dict
        features = {
            "high_intent_touch_count": float(high_intent_touch_count),
            "visited_pricing_or_demo": float(visited_pricing_or_demo),
            "cross_channel_engaged": float(cross_channel_engaged),
            "dm_x_pricing_or_demo": float(is_decision_maker and visited_pricing_or_demo),
            "pricing_page_visits_decayed": pricing_page_visits_decayed,
            "content_engagement_decayed": content_engagement_decayed,
            "smb_fit_x_intent": float(is_smb_fit) * pricing_page_visits_decayed,
            "good_source_x_content": float(lead_source_tier) * content_engagement_decayed,
            "demo_page_visits_decayed": demo_page_visits_decayed,
            "open_rate": open_rate,
            "email_clicks_decayed": email_clicks_decayed,
            "email_opens_decayed": email_opens_decayed,
            "industry_tier": float(industry_tier),
            "lead_source_tier": float(lead_source_tier),
            "click_to_open_rate": open_rate * 0.5,
            "is_decision_maker": float(is_decision_maker),
            "case_study_views_decayed": case_study_views_decayed,
            "company_size_score": float(company_size_score),
            "trial_depth_score": trial_depth_score,
            "is_engaged_trial": float(is_engaged_trial),
            "is_smb_fit": float(is_smb_fit),
            "senior_x_trial": float(seniority_rank) * trial_depth_score,
            "seniority_rank": float(seniority_rank),
            "visited_last_7_days": float(visited_last_7_days),
            "log_annual_revenue": 10.0,
            "tech_stack_score": float(tech_stack_score),
            "pages_per_session": pages_per_session,
            "roi_calculator_uses_decayed": roi_calculator_uses_decayed,
            "blog_views_decayed": blog_views_decayed,
            "is_unsubscribed": 0.0,
            "careers_page_visits_decayed": 0.0,
            "is_hard_bounced": 0.0,
            "is_bot_suspect": 0.0,
        }
        
        # Score
        with st.spinner("Scoring lead..."):
            result = scorer.score_single(features, lead_id)
        
        # Display results
        st.markdown("---")
        
        col_res1, col_res2, col_res3 = st.columns([1, 1, 2])
        
        with col_res1:
            st.metric("Lead ID", result.lead_id)
        
        with col_res2:
            tier_color = TIER_COLORS.get(result.tier, "gray")
            st.markdown(f"""
            <div style="background-color:{tier_color};padding:10px;border-radius:5px;text-align:center;color:white;font-weight:bold;">
                {result.tier} Tier
            </div>
            """, unsafe_allow_html=True)
        
        with col_res3:
            st.metric("Conversion Probability", f"{result.lr_prob:.1%}")
        
        # Score gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result.score_0_100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Lead Score (0-100)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': tier_color},
                'steps': [
                    {'range': [0, 24], 'color': TIER_COLORS["Suppress"]},
                    {'range': [25, 41], 'color': TIER_COLORS["Nurture"]},
                    {'range': [42, 59], 'color': TIER_COLORS["Warm"]},
                    {'range': [60, 100], 'color': TIER_COLORS["Hot"]},
                ],
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Explanations
        st.subheader("🧠 Why This Score?")
        
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            st.markdown("**✅ Positive Signals:**")
            st.markdown(f"1. {result.reason_1}")
            if result.reason_2 != "—":
                st.markdown(f"2. {result.reason_2}")
            if result.reason_3 != "—":
                st.markdown(f"3. {result.reason_3}")
        
        with exp_col2:
            if result.top_negative != "—":
                st.markdown("**⚠️ Risk Factor:**")
                st.markdown(f"{result.top_negative}")
        
        # Action recommendation
        st.info(f"📋 **Recommended Action:** {result.sla}")

elif page == "📁 Batch Score":
    st.title("Batch Score Leads")
    
    if not model_ready:
        st.stop()
    
    st.markdown("Upload a CSV file with lead features to score multiple leads at once.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        try:
            # Preview
            df_preview = pd.read_csv(uploaded_file)
            st.subheader("Preview")
            st.dataframe(df_preview.head(), use_container_width=True)
            
            # Check required columns
            missing = set(scorer.feature_cols) - set(df_preview.columns)
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                if st.button("🚀 Score All Leads", use_container_width=True):
                    with st.spinner("Scoring leads..."):
                        uploaded_file.seek(0)
                        results = scorer.score_from_csv(uploaded_file.read())
                    
                    st.success(f"✅ Scored {len(results)} leads")
                    
                    # Results
                    st.subheader("Results")
                    st.dataframe(results, use_container_width=True)
                    
                    # Tier distribution
                    tier_counts = results["tier"].value_counts()
                    fig = px.pie(
                        values=tier_counts.values,
                        names=tier_counts.index,
                        color=tier_counts.index,
                        color_discrete_map=TIER_COLORS,
                        title="Tier Distribution",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download button
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="scored_leads.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")

elif page == "🔍 Model Insights":
    st.title("Model Insights")
    
    if not model_ready:
        st.stop()
    
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Features:** {len(scorer.feature_cols)}")
        st.markdown(f"**Scoring Tiers:**")
        for tier, (lo, hi) in scorer.score_tiers.items():
            st.markdown(f"- {tier}: {lo}-{hi} → {scorer.tier_sla[tier]}")
    
    with col2:
        st.markdown("**Top Feature Categories:**")
        st.markdown("- Behavioral signals (visits, engagement)")
        st.markdown("- Firmographics (industry, company size)")
        st.markdown("- Email engagement (opens, clicks)")
        st.markdown("- Trial activity depth")
    
    # Feature descriptions
    st.subheader("Feature Descriptions")
    feature_df = pd.DataFrame([
        {"Feature": k, "Description": v}
        for k, v in FEATURE_LABELS.items()
    ])
    st.dataframe(feature_df, use_container_width=True)

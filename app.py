import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Logistic Growth Function
# -------------------------------
def logistic_growth(t, K, P0, r):
    return K / (1 + ((K - P0) / P0) * np.exp(-r * t))

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(layout="wide")
st.title("📈 YouTube Subscriber Growth Predictor")
st.markdown("### 🔹 Model: Logistic Growth + Milestones + Visualization")

# -------------------------------
# Inputs
# -------------------------------
st.subheader("🎛 Input Parameters")

col1, col2, col3, col4 = st.columns(4)

with col1:
    P0 = st.number_input("Initial Subscribers", 1, 1000000, 1000)

with col2:
    K = st.number_input("Max Audience (Carrying Capacity)", 1000, 10000000, 1000000)

with col3:
    r = st.slider("Growth Rate", 0.01, 1.0, 0.2)

with col4:
    days = st.slider("Prediction Days", 30, 1000, 180)

# Milestones input
milestone_text = st.text_input(
    "Enter Milestones (comma separated)",
    "1000,10000,50000,100000"
)
milestones = [int(x) for x in milestone_text.split(",") if x.strip().isdigit()]

# Buttons
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    run_model = st.button("🚀 Predict Growth")

with col_btn2:
    clear = st.button("🧹 Clear")

# -------------------------------
# Run Prediction
# -------------------------------
if run_model:

    # Generate data
    t = np.linspace(0, days, days)
    subs = logistic_growth(t, K, P0, r)
    df = pd.DataFrame({
        "Day": t.astype(int),
        "Subscribers": subs.astype(int)
    })

    # -------------------------------
    # Graphs Layout (2 columns for first 2 graphs)
    # -------------------------------
    colA, colB = st.columns(2)

    # Graph 1: Growth Curve
    with colA:
        st.subheader("📊 Subscriber Growth (Logistic Model)")

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(df["Day"], df["Subscribers"], label="Subscribers", color="blue")
        ax.axhline(y=K, color='green', linestyle='--', label="Max Audience")
        for m in milestones:
            ax.axhline(y=m, linestyle=':', alpha=0.5)
        ax.set_xlabel("Days")
        ax.set_ylabel("Subscribers")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

    # Graph 2: Milestone Table
    with colB:
        st.subheader("🏆 Milestone Tracking")

        milestone_data = []
        for m in milestones:
            reached = df[df["Subscribers"] >= m]
            if not reached.empty:
                milestone_data.append((m, int(reached.iloc[0]["Day"])))
            else:
                milestone_data.append((m, "Not reached"))

        milestone_df = pd.DataFrame(
            milestone_data, columns=["Milestone", "Day Reached"]
        )
        st.dataframe(milestone_df)

    # -------------------------------
    # Graph 3: Milestone Scatter Chart
    # -------------------------------
    st.subheader("📍 Milestone Chart")
    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.plot(df["Day"], df["Subscribers"], color="blue")
    for m, d in milestone_data:
        if isinstance(d, int):
            ax2.scatter(d, m, color='red')
            ax2.text(d, m, f"{m}", fontsize=8)
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Subscribers")
    ax2.grid()
    st.pyplot(fig2)

    # -------------------------------
    # Graphs 4 & 5: Advanced Visualizations (Growth Rate + Saturation)
    # -------------------------------
    st.subheader("📊 Advanced Visualizations")
    col3, col4 = st.columns(2)

    # Graph 4: Growth Rate Curve
    with col3:
        st.subheader("⚡ Growth Rate Over Time")
        growth_rate = np.gradient(df["Subscribers"])
        fig3, ax3 = plt.subplots()
        ax3.plot(df["Day"], growth_rate, color="orange")
        ax3.set_xlabel("Days")
        ax3.set_ylabel("Growth Rate")
        ax3.grid()
        st.pyplot(fig3)

    # Graph 5: % of Max Audience
    with col4:
        st.subheader("🎯 Audience Saturation")
        percent = (df["Subscribers"] / K) * 100
        fig4, ax4 = plt.subplots()
        ax4.plot(df["Day"], percent, color="green")
        ax4.set_xlabel("Days")
        ax4.set_ylabel("Percentage (%)")
        ax4.grid()
        st.pyplot(fig4)

    # -------------------------------
    # Data Table + Download
    # -------------------------------
    st.subheader("📋 Prediction Data")
    st.dataframe(df)

    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "youtube_growth.csv", "text/csv")

    # -------------------------------
    # Insights
    # -------------------------------
    st.subheader("📌 Insights")
    st.write(f"""
    - Initial Subscribers: {P0}  
    - Maximum Audience: {K}  
    - Growth Rate: {r}  

    **Logistic Growth Model**:
    - Fast growth initially  
    - Slows near maximum audience  
    - Milestones help track progress
    """)

# -------------------------------
# Clear Button Action
# -------------------------------
if clear:
    st.experimental_rerun()
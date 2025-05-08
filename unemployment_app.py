import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Set up the Streamlit page
st.set_page_config(
    page_title="Kenya Youth Unemployment Risk Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Load models ---
model_25 = joblib.load('quantile_model_25.pkl')
model_50 = joblib.load('quantile_model_50.pkl')
model_75 = joblib.load('quantile_model_75.pkl')

# --- Load dataset for visualization ---
data = pd.read_csv('youth_unemployment_kenya_simulated.csv')

# --- HEADER ---
st.title("🇰🇪 Youth Unemployment Risk Predictor")
st.markdown("""
Welcome to the **Youth Unemployment Risk Predictor** for Kenya.

This tool uses quantile regression to estimate the likelihood that a young person may be unemployed, based on:
- Education level
- Gender
- Region
- Monthly income
- Government program access

📊 You can also view trends and compare your risk to other groups.

*Note: This tool is for academic demonstration purposes only.*
""")

# --- INPUT FORM ---
st.header("📝 Enter Your Details")
education = st.selectbox(
    "Education Level",
    ['Primary', 'Secondary', 'Tertiary'],
    help="Select the highest level of education completed."
)

region = st.selectbox(
    "Region",
    ['Nairobi', 'Central', 'Western', 'Coast', 'Rift Valley'],
    help="Select the region you live in."
)

gender = st.selectbox(
    "Gender",
    ['Female', 'Male'],
    help="Select your gender."
)

income = st.number_input(
    "Monthly Income (KES)",
    min_value=0,
    help="Enter your average monthly income."
)

ajira = st.selectbox(
    "Have you participated in Ajira Digital?",
    ['No', 'Yes'],
    help="Select 'Yes' if you've participated in the Ajira Digital Program."
)

yedf = st.selectbox(
    "Have you accessed the Youth Fund (YEDF)?",
    ['No', 'Yes'],
    help="Select 'Yes' if you've received funding through YEDF."
)

# Quantile selection
quantile_choice = st.radio(
    "Choose the risk group you want to estimate for:",
    ['25th Percentile (Low-Risk)', '50th Percentile (Median)', '75th Percentile (High-Risk)']
)

# Convert to numeric values
ajira_val = 1 if ajira == 'Yes' else 0
yedf_val = 1 if yedf == 'Yes' else 0

# --- PREDICTION ---
if st.button("🔮 Predict Unemployment Risk"):
    user_input = pd.DataFrame({
        'Education': [education],
        'Region': [region],
        'Gender': [gender],
        'Income': [income],
        'Ajira_Digital': [ajira_val],
        'YEDF_Access': [yedf_val]
    })

    # Select model
    if "25th" in quantile_choice:
        prediction = model_25.predict(user_input)[0]
    elif "75th" in quantile_choice:
        prediction = model_75.predict(user_input)[0]
    else:
        prediction = model_50.predict(user_input)[0]

    st.success(f"Your predicted unemployment risk is: **{prediction:.2%}**")

    # --- Risk Explanation Chart ---
    st.subheader("📊 How Your Risk Compares Across Groups")
    risk_data = pd.DataFrame({
        'Quantile': ['25th (Low-Risk)', '50th (Median)', '75th (High-Risk)'],
        'Risk': [
            model_25.predict(user_input)[0],
            model_50.predict(user_input)[0],
            model_75.predict(user_input)[0]
        ]
    })

    chart = sns.barplot(x='Quantile', y='Risk', data=risk_data, palette='viridis')
    chart.set_ylabel("Predicted Unemployment Risk")
    chart.set_ylim(0, 1)
    st.pyplot(plt.gcf())
    plt.clf()

# --- UNEMPLOYMENT TRENDS ---
st.header("📈 Trends in Youth Unemployment (Simulated Data)")

st.subheader("By Education Level")
edu_plot = sns.countplot(x='Education', hue='Unemployed', data=data, palette='Set2')
edu_plot.set_title("Unemployment by Education")
edu_plot.set_ylabel("Number of Youth")
st.pyplot(plt.gcf())
plt.clf()

st.subheader("By Region")
region_plot = sns.countplot(x='Region', hue='Unemployed', data=data, palette='Set1')
region_plot.set_title("Unemployment by Region")
region_plot.set_ylabel("Number of Youth")
plt.xticks(rotation=45)
st.pyplot(plt.gcf())
plt.clf()

# --- FOOTER ---
st.markdown("---")
st.caption("Developed by Natania Kimilu | Youth Unemployment Project | 2025")

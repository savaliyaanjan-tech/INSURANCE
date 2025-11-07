
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import io

st.set_page_config(page_title="Insurance Insights Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Insurance.csv")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)
    return df

df = load_data()

encoder = LabelEncoder()
encoded_df = df.copy()
for col in encoded_df.select_dtypes(include=['object']).columns:
    encoded_df[col] = encoder.fit_transform(encoded_df[col])

st.sidebar.title("🔍 Filters")
columns = list(df.columns)
job_col = st.sidebar.selectbox("Select Job/Role column", options=columns)
satisfaction_col = st.sidebar.selectbox("Select Satisfaction or Numeric column", options=columns)

unique_roles = df[job_col].unique()
selected_roles = st.sidebar.multiselect("Select Roles", unique_roles, default=unique_roles)
slider_min, slider_max = int(df[satisfaction_col].min()), int(df[satisfaction_col].max())
selected_value = st.sidebar.slider("Filter by Satisfaction/Policy Metric", slider_min, slider_max, (slider_min, slider_max))

filtered_df = df[(df[job_col].isin(selected_roles)) & (df[satisfaction_col].between(selected_value[0], selected_value[1]))]

st.title("📊 Insurance Insights Dashboard")
st.markdown("### Explore policy patterns, customer satisfaction, and model predictions to improve insurance strategies.")

tab1, tab2, tab3 = st.tabs(["📈 Data Insights", "🧠 Model Performance", "📤 Predict New Data"])

with tab1:
    st.subheader("1️⃣ Visual Insights")

    fig1, ax1 = plt.subplots()
    sns.countplot(x='Policy_Status', data=filtered_df, ax=ax1)
    plt.title("Policy Status Distribution")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Policy_Status', y=satisfaction_col, data=filtered_df, ax=ax2)
    plt.title("Satisfaction / Policy Metric by Status")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.barplot(x=job_col, y=satisfaction_col, data=filtered_df, estimator=np.mean, ax=ax3)
    plt.title("Average Satisfaction by Role/Policy Type")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    sns.scatterplot(x=satisfaction_col, y='Premium' if 'Premium' in filtered_df.columns else df.columns[2],
                    hue='Policy_Status', data=filtered_df, ax=ax4)
    plt.title("Satisfaction vs Premium (colored by Policy Status)")
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots()
    filtered_df.groupby('Policy_Status').size().plot(kind='pie', autopct='%1.1f%%', ax=ax5)
    plt.title("Policy Status Percentage Share")
    st.pyplot(fig5)

with tab2:
    st.subheader("🧠 Model Training and Evaluation")

    if st.button("Run All Models"):
        X = encoded_df.drop('Policy_Status', axis=1)
        y = encoded_df['Policy_Status']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        models = {
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosted": GradientBoostingClassifier()
        }

        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results.append([name, acc])
            st.write(f"### {name} Results")
            st.text(classification_report(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"{name} - Confusion Matrix")
            st.pyplot(fig)

        results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
        st.write("### Model Comparison")
        st.dataframe(results_df)

with tab3:
    st.subheader("📤 Upload New Data for Prediction")

    uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])
    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(new_df.head())

        for col in new_df.columns:
            if new_df[col].dtype == 'object':
                new_df[col].fillna(new_df[col].mode()[0], inplace=True)
            else:
                new_df[col].fillna(new_df[col].mean(), inplace=True)

        for col in new_df.select_dtypes(include=['object']).columns:
            new_df[col] = encoder.fit_transform(new_df[col])

        model = RandomForestClassifier()
        X = encoded_df.drop('Policy_Status', axis=1)
        y = encoded_df['Policy_Status']
        model.fit(X, y)

        predictions = model.predict(new_df)
        new_df['Predicted_Policy_Status'] = predictions

        st.success("✅ Predictions completed!")
        st.dataframe(new_df.head())

        buffer = io.BytesIO()
        new_df.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=buffer,
            file_name="Predicted_Insurance.csv",
            mime="text/csv"
        )

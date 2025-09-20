import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="🚚 Last Mile Delivery Dashboard", layout="wide")
st.title("🚚 Last Mile Delivery Dashboard")
st.markdown("Explore, filter, and predict delivery performance interactively!")

# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader("📥 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # -------------------------
    # Data Cleaning
    # -------------------------
    df.dropna(subset=['Order_ID', 'Delivery_Time', 'Vehicle'], inplace=True)
    categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Age grouping
    if 'Agent_Age' in df.columns:
        df['AgentAgeGroup'] = pd.cut(df['Agent_Age'], bins=[0,25,40,100], labels=['<25','25–40','40+'])

    # Late delivery flag
    threshold = df['Delivery_Time'].mean() + df['Delivery_Time'].std()
    df['LateDeliveryFlag'] = np.where(df['Delivery_Time'] > threshold, 1, 0)

    # -------------------------
    # Sidebar Filters
    # -------------------------
    st.sidebar.header("🔍 Filters")
    filter_weather = st.sidebar.multiselect("Weather", options=df['Weather'].unique(), default=df['Weather'].unique())
    filter_traffic = st.sidebar.multiselect("Traffic", options=df['Traffic'].unique(), default=df['Traffic'].unique())
    filter_vehicle = st.sidebar.multiselect("Vehicle", options=df['Vehicle'].unique(), default=df['Vehicle'].unique())
    filter_area = st.sidebar.multiselect("Area", options=df['Area'].unique(), default=df['Area'].unique())
    filter_category = st.sidebar.multiselect("Category", options=df['Category'].unique(), default=df['Category'].unique())

    filtered_df = df[
        (df['Weather'].isin(filter_weather)) &
        (df['Traffic'].isin(filter_traffic)) &
        (df['Vehicle'].isin(filter_vehicle)) &
        (df['Area'].isin(filter_area)) &
        (df['Category'].isin(filter_category))
    ]

    # -------------------------
    # KPIs
    # -------------------------
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("📊 Avg Delivery Time", f"{filtered_df['Delivery_Time'].mean():.2f} mins")
    kpi2.metric("⏰ Late Deliveries", f"{filtered_df['LateDeliveryFlag'].mean()*100:.2f}%")
    kpi3.metric("📦 Total Deliveries", f"{len(filtered_df)}")

    st.download_button(
        label="📥 Download Filtered Data",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_data.csv",
        mime="text/csv"
    )

    # -------------------------
    # Visual Tabs
    # -------------------------
    tabs = st.tabs(["📊 Trends & Insights", "🤖 Predictions", "📈 Extra Analysis"])

    # -------- Trends & Insights --------
    with tabs[0]:
        st.subheader("Delivery Time by Weather & Traffic")
        plt.figure(figsize=(10,5))
        sns.barplot(data=filtered_df, x="Weather", y="Delivery_Time", hue="Traffic", ci="sd")
        st.pyplot(plt.gcf())
        plt.clf()

        st.subheader("Avg Delivery Time by Vehicle")
        plt.figure(figsize=(8,5))
        sns.barplot(data=filtered_df, x="Vehicle", y="Delivery_Time", ci="sd")
        st.pyplot(plt.gcf())
        plt.clf()

        if 'Agent_Rating' in df.columns and 'Agent_Age' in df.columns:
            st.subheader("Agent Rating vs Delivery Time")
            plt.figure(figsize=(10,5))
            sns.scatterplot(data=filtered_df, x="Agent_Rating", y="Delivery_Time", hue="AgentAgeGroup", palette="deep")
            st.pyplot(plt.gcf())
            plt.clf()

        st.subheader("Category-wise Delivery Time Distribution")
        plt.figure(figsize=(10,5))
        sns.boxplot(data=filtered_df, x="Category", y="Delivery_Time")
        st.pyplot(plt.gcf())
        plt.clf()

    # -------- Predictions --------
    with tabs[1]:
        st.subheader("🤖 Predict Late Deliveries")

        # Prepare features
        feature_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
        df_enc = pd.get_dummies(df[feature_cols], drop_first=True)
        X = df_enc
        y = df['LateDeliveryFlag']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # User input for prediction
        st.markdown("### Try 'What-if' Analysis")
        user_weather = st.selectbox("Weather", df['Weather'].unique())
        user_traffic = st.selectbox("Traffic", df['Traffic'].unique())
        user_vehicle = st.selectbox("Vehicle", df['Vehicle'].unique())
        user_area = st.selectbox("Area", df['Area'].unique())
        user_category = st.selectbox("Category", df['Category'].unique())

        user_input = pd.DataFrame([[user_weather, user_traffic, user_vehicle, user_area, user_category]],
                                  columns=feature_cols)
        user_input_enc = pd.get_dummies(user_input)
        user_input_enc = user_input_enc.reindex(columns=X.columns, fill_value=0)

        pred = model.predict(user_input_enc)[0]
        prob = model.predict_proba(user_input_enc)[0][1]

        st.write(f"🔮 Prediction: {'Late Delivery' if pred==1 else 'On-time Delivery'}")
        st.write(f"📈 Probability of being late: {prob*100:.2f}%")

    # -------- Extra Analysis --------
    with tabs[2]:
        st.subheader("Delivery Time Distribution")
        plt.figure(figsize=(8,4))
        sns.histplot(filtered_df['Delivery_Time'], kde=True, bins=30)
        st.pyplot(plt.gcf())
        plt.clf()

        st.subheader("% Late Deliveries by Traffic")
        traffic_summary = filtered_df.groupby("Traffic")['LateDeliveryFlag'].mean().reset_index()
        traffic_summary['LateDeliveryFlag'] *= 100
        plt.figure(figsize=(8,4))
        sns.barplot(data=traffic_summary, x="Traffic", y="LateDeliveryFlag")
        plt.ylabel("Late Deliveries (%)")
        st.pyplot(plt.gcf())
        plt.clf()

else:
    st.info("Please upload a CSV file to start exploring 🚀")

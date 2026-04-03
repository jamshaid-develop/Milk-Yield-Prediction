import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Load Model ----------
with open("milk_yield_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle['model']
scaler = bundle['scaler']
features = bundle['features']
model_name = bundle['model_name']

# ---------- Sidebar Navigation ----------
st.sidebar.title("🐄 Menu")

option = st.sidebar.radio("Select Option", [
    "🏠 Home",
    "🔮 Predict",
    "📊 Visualizations",
    "ℹ️ About"
])

# ---------- HOME ----------
if option == "🏠 Home":
    st.title("🥛 Cow Milk Yield Prediction App")
    st.write("This app predicts milk yield using Machine Learning.")

    st.markdown("### 📌 Features Used:")
    for f in features:
        st.write(f"✔ {f}")

# ---------- PREDICT ----------
elif option == "🔮 Predict":
    st.title("🔮 Predict Milk Yield")

    cow_id = st.number_input("Cow ID", value=17012)
    img_number = st.number_input("Image Number", value=1)
    has_multiple = st.selectbox("Has Multiple Images?", [0, 1])

    yield_mean = st.number_input("Yield Mean", value=7000.0)
    yield_std = st.number_input("Yield Std", value=500.0)
    img_count = st.number_input("Image Count", value=3)

    yield_min = st.number_input("Yield Min", value=6000.0)
    yield_max = st.number_input("Yield Max", value=8000.0)
    yield_range = st.number_input("Yield Range", value=2000.0)

    if st.button("Predict"):
        input_data = np.array([[
            cow_id, img_number, has_multiple,
            yield_mean, yield_std, img_count,
            yield_min, yield_max, yield_range
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        st.success(f"🥛 Predicted Milk Yield: {prediction[0]:.2f} grams")

# ---------- VISUALIZATIONS ----------
elif option == "📊 Visualizations":
    st.title("📊 Model Insights & Visualizations")

    # Dummy sample input (for demo graph)
    sample = np.array([[17012,1,1,7000,500,3,6000,8000,2000]])
    sample_scaled = scaler.transform(sample)

    # Feature Importance
    if hasattr(model, "coef_"):
        st.subheader("📈 Feature Importance (Lasso)")

        coef_df = pd.DataFrame({
            "Feature": features,
            "Coefficient": model.coef_
        }).sort_values(by="Coefficient")

        fig, ax = plt.subplots()
        ax.barh(coef_df["Feature"], coef_df["Coefficient"])
        st.pyplot(fig)

    # Sample Prediction Comparison
    pred = model.predict(sample_scaled)

    st.subheader("📉 Sample Prediction vs Mean")

    comp_df = pd.DataFrame({
        "Type": ["Mean Yield", "Predicted Yield"],
        "Value": [7000, pred[0]]
    })

    st.bar_chart(comp_df.set_index("Type"))

# ---------- ABOUT ----------
elif option == "ℹ️ About":
    st.title("ℹ️ About Project")

    st.write("📊 Dataset: Kaggle Cow Milk Yield Prediction")
    st.write(f"🤖 Model Used: {model_name}")
    st.write("🎯 Task: Regression (Predict Milk Yield)")

    st.markdown("### 📌 Features:")
    for f in features:
        st.write(f"✔ {f}")

    st.markdown("### 👨‍💻 Developer")
    st.write("ML Streamlit App \n Project developed by M Jamshaid Nawaz 🚀")
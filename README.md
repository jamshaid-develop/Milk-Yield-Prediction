🥛 Cow Milk Yield Prediction App

📌 Project Overview:
This is a Machine Learning project that predicts milk yield (in grams) of cows based on engineered features extracted from image data.

📊 Dataset:
- Source: Kaggle
- Train Samples: 999
- Test Samples: 237
- Target: Milk Yield (grams)

⚙️ Features Used (9):
1. cow_id
2. img_number
3. has_multiple
4. yield_mean
5. yield_std
6. img_count
7. yield_min
8. yield_max
9. yield_range

🤖 Model:
- Best Model: Lasso Regression
- Test R² Score: 0.9316

📦 Files Included:
- milk_yield_model.pkl → Trained model + scaler + features
- Streamlit App → Cow_Images_for_Milk_Yield_Prediction.py

🚀 How to Run:
1. Open terminal in project folder
2. Activate virtual environment
3. Run command:

   streamlit run Cow_Images_for_Milk_Yield_Prediction.py

📌 Notes:
- Make sure milk_yield_model.pkl is in the same folder as the app
- All 9 features must be provided for correct prediction

👨‍💻 Author:
BSCS Student Project

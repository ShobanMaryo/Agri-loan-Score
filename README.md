🌱 AgriLoanScore – Farmer Loan Eligibility Predictor
AgriLoanScore is a machine learning-powered application designed to help financial institutions and farmers assess loan eligibility. By analyzing socio-economic factors, agricultural productivity, and credit history, the system provides real-time eligibility predictions with a user-friendly, multilingual dashboard.

🚀 Features
AI-Powered Predictions: Uses a Random Forest Classifier to predict eligibility based on historical data.
Multilingual Interface: Full support for both English and Tamil languages.
Real-time Confidence: Provides a confidence score for every prediction.
Interactive Dashboard: Built with Streamlit for a smooth, responsive user experience.
Data Analytics: Includes visual insights into key factors like CIBIL scores and their impact on eligibility.
🛠️ Tech Stack
Language: Python
Machine Learning: Scikit-Learn (Random Forest)
Data Processing: Pandas, NumPy
Frontend/Dashboard: Streamlit
Visualization: Matplotlib, Seaborn
Model Serialization: Joblib
📁 Project Structure
train_model.py: Script to preprocess data and train the Random Forest model.
dashboard.py: The Streamlit web application code.
agriloanscore_dataset_tn.csv: Dataset containing agricultural and financial records.
agriloanscore_model.pkl: The trained and serialized model ready for inference.
⚙️ Installation & Usage
Clone the repository:

bash
git clone https://github.com/your-username/AgriLoanScore.git
cd AgriLoanScore
Install dependencies:

bash
pip install pandas scikit-learn joblib streamlit matplotlib seaborn
Train the model (Optional):

bash
python train_model.py
Run the Dashboard:

bash
streamlit run dashboard.py
📊 Key Indicators for Eligibility
The model considers the following factors:

Financials: Annual Income, CIBIL Score, Credit History.
Agricultural Details: Land Size, Crop Type, Yield per Acre, Market Price.
Loan Details: Requested Amount, Purpose, Repayment Capacity.

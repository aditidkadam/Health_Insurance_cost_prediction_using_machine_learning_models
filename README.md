# Health Insurance Payment Prediction Using Machine Learning

### **1. Description**  
This project analyzes health insurance customer data to identify factors that influence payment amounts and builds a machine learning model capable of predicting insurance charges based on demographic, medical, and lifestyle characteristics. The project includes exploratory data analysis, data preprocessing, model training, and deployment through an interactive Streamlit web application. The outcome demonstrates how predictive analytics can support cost estimation, risk assessment, and pricing strategies.

---

## **2. Data Information**

### **Data Sources**  
- Dataset included locally as `insurance.csv`  
- Commonly derived from public insurance cost datasets used for health expense modeling  

### **Data Description**  
| File | Format | Description |
|------|---------|-------------|
| `insurance.csv` | CSV | Contains individual-level records with demographic and health-related attributes used to model insurance payment amounts |

### **Data Dictionary**
| Column Name | Description |
|-------------|-------------|
| age | Age of individual |
| gender | Biological sex |
| bmi | Body Mass Index |
| bloodpressure | Blood pressure reading |
| children | Number of dependents |
| diabetic | Whether the individual is diabetic |
| smoker | Smoking status |
| charges | Actual insurance payment cost |

### **Data Collection**
- Provided as a static dataset  
- No live scraping or API retrieval  
- Used directly for exploratory analysis and modeling  

---

## **3. Project Structure and Code**

### **File Structure**
```
project/
├── app.py
├── analysis_model.ipynb
├── insurance.csv
├── best_model.pkl
├── scaler.pkl
├── gender_label_encoder.pkl
├── diabetic_label_encoder.pkl
├── smoker_label_encoder.pkl
└── README.md
```

### **Key Files**
- `analysis_model.ipynb` — EDA, preprocessing, modeling, evaluation  
- `insurance.csv` — Dataset used for training and analysis  
- `app.py` — Streamlit UI for collecting user inputs and returning predictions  
- `best_model.pkl` — Saved final model  
- `scaler.pkl` — Feature scaling transformer  
- `*_label_encoder.pkl` — Encoders for categorical variables  

### **Technologies / Libraries**
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Joblib  
- Matplotlib / Seaborn (used in notebook)

### **Environment Setup / Installation**
```
conda install -n <env_name> 
conda activate <env_name>
conda install <package1> <package2> <package3>


```

---

## **4. Analysis and Methodology**

### **Problem Statement / Research Questions**
- Can insurance payment amounts be predicted reliably from lifestyle and demographic factors?
- Which variables most strongly influence insurance cost?
- How do smoking, diabetes, age, and BMI shift premium expectations?

### **Methodology**
- Exploratory Data Analysis  
- Feature encoding and scaling  
- Model comparison (regression and tree-based approaches)  
- Model selection based on accuracy and error metrics  
- Export of deployment-ready prediction pipeline  

### **Data Cleaning and Preprocessing**
- Categorical encoding using label encoders  
- Standardization of numerical fields  
- Outlier inspection  
- Train-test splitting  

### **Key Findings / Results**
- Smoking was the strongest driver of payment cost  
- Higher BMI and age increased charges significantly  
- Selected model delivered strong predictive accuracy (R² / RMSE suitable for regression tasks)  
- Model generalized well to unseen data  

---

## **5. Usage and Reproducibility**

### **How to Run / Use**
1. Ensure required `.pkl` files are in the same directory as `app.py`  
2. Run the app:
```
streamlit run app.py
```
3. Enter the requested input values in the interface  
4. View the predicted insurance payment estimate  

### **Limitations / Assumptions**
- Dataset size limits broad generalization  
- Self-reported medical indicators may introduce bias  
- Does not include geographical pricing factors  
- Predictions are estimates, not real insurance premiums  


### ** Contact Information **
- email_id : aditikadam2460@gmail.com
- linkedin: https://www.linkedin.com/in/aditi-kadam24/


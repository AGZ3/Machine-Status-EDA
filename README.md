# Machine Status Analysis - Predictive Maintenance

## Overview  
This project focuses on predictive maintenance for industrial machines by analyzing sensor data to forecast equipment failures. The system processes raw sensor readings, performs exploratory data analysis, engineers relevant features, and builds machine learning models to predict machine status (`NORMAL`, `BROKEN`, or `RECOVERING`).

---

## Key Features  
- **Data Preprocessing**: Handles missing values, outliers, and standardizes sensor data.  
- **Exploratory Analysis**: Visualizes sensor correlations, temporal patterns, and status distributions.  
- **Feature Engineering**: Creates rolling statistics (mean, std) and lagged features for predictive modeling.  
- **Machine Learning**: Implements and evaluates three classification models:  
  - Random Forest Classifier  
  - Logistic Regression  
  - Decision Tree Classifier  

---

## Data Processing Pipeline  
1. **Preprocessing**:  
   - Missing value imputation (mean filling)  
   - Outlier capping using IQR method  
   - Z-score standardization  
   - Machine status encoding (`BROKEN→0`, `NORMAL→1`, `RECOVERING→2`)  

2. **Feature Engineering**:  
   - Rolling means (5-period window)  
   - Rolling standard deviations  
   - 1-period lagged features  
   - Focused on 11 relevant sensors (e.g., `sensor_00`, `sensor_50`)  

---

## Model Performance  
| Model              | Accuracy | Precision | Recall | F1 Score |  
|--------------------|----------|-----------|--------|----------|  
| Random Forest      | 0.98     | 0.98      | 0.98   | 0.98     |  
| Decision Tree      | 0.98     | 0.98      | 0.98   | 0.98     |  
| Logistic Regression| 0.93     | 0.87      | 0.93   | 0.90     |  

**Best Model**: Random Forest (high accuracy and robustness).  

For detailed methodology, refer to the Machine Status Analysis Report.

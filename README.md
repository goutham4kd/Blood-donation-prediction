Blood Donation Prediction using AutoML and Logistic Regression

This repository contains the code and resources for a machine learning project developed during my Data Analytics Traineeship at **MedTourEasy**. The goal is to predict whether a donor is likely to give blood again, helping improve blood bank management and outreach.

 Project Overview

- Problem: Predict potential future blood donors based on donation history.
- Approach: Logistic Regression with TPOT AutoML for automated model pipeline selection.
- Dataset: Historical donation data with features like frequency, recency, and volume.
- Accuracy: Achieved 85% prediction accuracy on validation data.

 Key Features

- Data cleaning and preprocessing with Pandas and NumPy
- Automated model building using TPOT
- Model evaluation with classification metrics
- Exploratory data analysis and visualization
- Well-structured pipeline for reproducibility

🗂️ Project Structure

```
blood-donation-prediction/
├── data/                  # Dataset (anonymized or link)
├── notebooks/             # Jupyter Notebooks for EDA and modeling
├── src/                   # Source code (functions, scripts)
├── README.md              # Project overview
├── requirements.txt       # Python dependencies
├── LICENSE                # License file (MIT recommended)
└── .gitignore             # Ignore unnecessary files
```Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/blood-donation-prediction.git
cd blood-donation-prediction
pip install -r requirements.txt
```

 Usage

Run the main notebook to preprocess data and train the model:

```bash
jupyter notebook notebooks/blood_donation_model.ipynb
```

 Results

- Logistic Regression with TPOT-optimized pipeline
- ~85% accuracy with improved generalizability
- Useful for donor targeting and blood stock optimization

 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

 Acknowledgements

- MedTourEasy – Traineeship support and problem statement
- TPOT – AutoML library used for pipeline optimization
- Scikit-learn – Core machine learning algorithms

 Author

Goutham Kumar Das 
📧 gouthamkumar471@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/goutham-kumar-53aaba319/)

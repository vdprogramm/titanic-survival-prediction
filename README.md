# 🚢 Titanic Survival Prediction V2

**Machine Learning Classification | Python • Scikit-learn • Pandas • Streamlit**

A machine learning project that predicts Titanic passenger survival based on demographic and travel information. The project covers exploratory data analysis, preprocessing, feature engineering, model training, evaluation, and deployment as an interactive web application.

## 🌐 Live Demo

**[Try the Titanic Survival Prediction App](https://titanic-survival-prediction-jogdkappjyf55fudllbimew.streamlit.app/)**

Users can enter passenger information and receive a survival prediction from a trained Random Forest model.

## 📊 Project Results

| Metric | Result |
|---|---|
| Machine Learning Task | Binary Classification |
| Dataset | Kaggle Titanic |
| Algorithm | Random Forest Classifier |
| Baseline Kaggle Public Score | 0.75598 |
| Improved V2 Kaggle Public Score | **0.78708** |
| Improvement | +3.11 percentage points |
| Deployed Application | Streamlit Community Cloud |

The V2 model improved the Kaggle public score through feature engineering and Random Forest hyperparameter tuning.

*Note: Kaggle scores refer to submitted predictions. The deployed model is retrained and its predictions may differ from the original V2 submission.*

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Algorithm | Random Forest |
| Model Serialization | Joblib |
| Web Application | Streamlit |
| Development Environment | Jupyter Notebook, VS Code |
| Deployment | Streamlit Community Cloud |
| Version Control | Git, GitHub |

## 🎯 Project Objectives

- Analyze the Titanic passenger dataset.
- Clean and preprocess missing or categorical data.
- Identify factors associated with passenger survival.
- Train and evaluate classification models.
- Improve predictive performance through feature engineering.
- Build an interactive web interface for predictions.
- Deploy the application for public access.

## 📁 Project Structure

```text
titanic-survival-ml/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── titanic_analysis.ipynb
├── src/
│   └── train.py
├── models/
│   └── titanic_model.pkl
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── submission.csv
└── submission_v2.csv
```

## 📈 Exploratory Data Analysis

The analysis explores how passenger characteristics relate to survival, including:

- Gender and survival rate.
- Passenger class and survival rate.
- Age distribution.
- Ticket fare.
- Family relationships.
- Missing data and feature distributions.

Exploratory analysis and visualizations are available in `notebooks/titanic_analysis.ipynb`.

## 🧠 Machine Learning Workflow

### 1. Data Preprocessing

The dataset is prepared by:

- Handling missing values in `Age`, `Fare`, and `Embarked`.
- Encoding categorical variables.
- Selecting relevant numerical and categorical features.
- Splitting labeled data into training and validation sets.

### 2. Feature Engineering

The improved notebook model includes additional features:

**FamilySize**

```python
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
```

**IsAlone**

```python
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
```

**Title**

Extracted from passenger names and grouped into common and rare titles.

These features help the model capture additional patterns in the passenger data.

### 3. Model Training

The project experiments with classification algorithms, including Logistic Regression and Random Forest.

The improved Random Forest configuration uses:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

### 4. Model Evaluation

Model performance is examined using:

- Validation accuracy.
- Classification report.
- Confusion matrix.
- Feature importance.
- Kaggle public leaderboard score.

## 💻 Streamlit Web Application

The deployed application allows users to enter:

- Passenger class.
- Gender.
- Age.
- Ticket fare.
- Number of siblings or spouses.
- Number of parents or children.
- Port of embarkation.
- Passenger title.

The application loads a saved machine learning model and generates a passenger survival prediction.

The output is a statistical prediction for educational purposes, not a historical determination of an individual passenger's fate.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/vdprogramm/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train and save the model

```bash
python src/train.py
```

### 5. Launch the Streamlit application

```bash
python -m streamlit run app.py
```

Open:

`http://localhost:8501`

## ☁️ Deployment

The web application is deployed using **Streamlit Community Cloud** and connected to the GitHub repository.

Deployment configuration:

```text
Repository: vdprogramm/titanic-survival-prediction
Branch: main
Main file path: app.py
```

The deployment installs Python dependencies from `requirements.txt` and launches the Streamlit application.

## 📚 Key Learnings

Through this project, I gained practical experience in:

- Data cleaning and exploratory analysis.
- Machine learning classification workflows.
- Feature engineering.
- Model training and evaluation.
- Comparing baseline and improved models.
- Saving and loading trained models.
- Building Python web applications with Streamlit.
- Deploying machine learning applications through GitHub.

## 🔮 Future Improvements

- Improve preprocessing consistency using Scikit-learn Pipelines.
- Compare additional classification algorithms.
- Perform cross-validation and hyperparameter optimization.
- Add model explainability visualizations.
- Improve the Streamlit user interface.
- Add automated tests for the prediction pipeline.

## 👨‍💻 Author

**Dinh Thanh Vinh**

GitHub: [vdprogramm](https://github.com/vdprogramm)

---

⭐ If you find this project useful, consider giving the repository a star.

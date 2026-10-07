# Titanic Survival Prediction 🚢

A Machine Learning classification project that predicts whether a passenger survived the Titanic disaster based on passenger information.

This project covers the complete basic Machine Learning workflow, including data exploration, preprocessing, feature engineering, model training, evaluation, and Kaggle submission.

## Project Overview

The objective of this project is to predict passenger survival using the Titanic dataset from Kaggle.

The project started with a baseline Random Forest model and was later improved through feature engineering.

### Kaggle Results

| Model | Kaggle Score |
|---|---:|
| Random Forest Baseline | 0.75598 |
| Random Forest + Feature Engineering | **0.78708** |

Feature engineering improved the Kaggle score from **75.60% to 78.71%**, an improvement of approximately **3.11 percentage points**.

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Dataset

The project uses the Titanic dataset from the Kaggle competition:

**Titanic - Machine Learning from Disaster**

Main files:

- `train.csv` - training data containing the `Survived` target
- `test.csv` - test data used for Kaggle predictions

The target variable is:

- `0` - Did not survive
- `1` - Survived

Important input features include:

- `Pclass` - passenger class
- `Sex` - gender
- `Age` - passenger age
- `SibSp` - siblings/spouses aboard
- `Parch` - parents/children aboard
- `Fare` - ticket fare
- `Embarked` - port of embarkation

## Data Preprocessing

The preprocessing pipeline includes:

- Inspecting missing values
- Filling missing `Age` values using the median
- Filling missing `Embarked` values using the mode
- Filling missing `Fare` values using the median
- Encoding categorical variables using one-hot encoding
- Removing unused/high-missing features from the baseline model

The `Cabin` feature was not used in the baseline model because a large portion of its values were missing.

## Exploratory Data Analysis

Basic exploratory data analysis was performed to understand relationships between passenger attributes and survival.

One notable observation was that female passengers had a significantly higher survival rate than male passengers.

## Feature Engineering

The second version introduced additional features.

### FamilySize

```python
FamilySize = SibSp + Parch + 1
```

Represents the total number of family members travelling together, including the passenger.

### IsAlone

```python
IsAlone = 1 if FamilySize == 1 else 0
```

Identifies passengers travelling alone.

### Title

Passenger titles were extracted from the `Name` column.

Examples:

- Mr
- Mrs
- Miss
- Master

Less common titles were grouped into the `Rare` category.

These engineered features improved the Kaggle score from **0.75598 to 0.78708**.

## Machine Learning Models

Two classification algorithms were explored:

### Logistic Regression

Used as a simple classification baseline for model comparison.

### Random Forest Classifier

Random Forest was used as the main prediction model.

The improved version used parameters such as:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

A local validation set was created using an 80/20 train-validation split.

Final performance was also evaluated through Kaggle submissions.

## Results

The baseline submission achieved:

```text
Kaggle Score: 0.75598
```

After adding `FamilySize`, `IsAlone`, and `Title`:

```text
Kaggle Score: 0.78708
```

This demonstrates how feature engineering can improve a Machine Learning model without simply increasing model complexity.

## Project Structure

```text
titanic-survival-ml/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── titanic_analysis.ipynb
│
├── submission.csv
├── submission_v2.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd titanic-survival-ml
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/titanic_analysis.ipynb
```

and run the cells from top to bottom.

## What I Learned

Through this project, I practiced:

- Data cleaning with Pandas
- Handling missing values
- Exploratory Data Analysis
- Feature engineering
- Categorical feature encoding
- Train/validation splitting
- Classification using Logistic Regression and Random Forest
- Model evaluation
- Feature importance analysis
- Generating predictions for unseen data
- Creating and submitting predictions to Kaggle

## Future Improvements

Possible improvements include:

- Cross-validation
- Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Additional feature engineering
- Comparing Gradient Boosting models
- Building a reusable Scikit-learn Pipeline

## Author

**Dinh Thanh Vinh**

Backend Developer / Machine Learning Learner
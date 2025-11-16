# 🎬 Movie Analytics - Revenue Prediction Project

## 📋 Project Description

This project applies Data Mining and Machine Learning techniques to analyze and predict movie revenue based on the TMDB (The Movie Database) dataset. The model uses **Random Forest Regressor** algorithm with features such as movie genres, budget, popularity, and user ratings.

## 🎯 Objectives

- Predict movie revenue based on input features
- Analyze correlations between factors affecting movie revenue
- Evaluate Machine Learning model performance on real-world dataset
- Identify the most important factors influencing movie success

## 📊 Data

### Data Source
- **Dataset**: TMDB 5000 Movies
- **File**: `tmdb_5000_movies.csv`
- **Path**: `data movie/tmdb_5000_movies.csv`

### Features Used

#### Numeric Features:
- `popularity`: Movie popularity score
- `vote_average`: Average user rating
- `vote_count`: Number of votes
- `budget`: Production budget

#### Categorical Features:
- `genre_ids`: Movie genres (encoded using MultiLabelBinarizer)
- `original_language`: Original language (en or other)

#### Target Variable:
- `revenue`: Movie revenue

## 🛠️ Technologies and Libraries

### Python Libraries
```python
- pandas: Data processing and analysis
- numpy: Numerical computing
- scikit-learn: Machine Learning
- matplotlib: Data visualization
- seaborn: Advanced visualization
```

### Machine Learning Techniques
- **Model**: Random Forest Regressor
- **Hyperparameter Optimization**: RandomizedSearchCV
- **Data Transformation**: Log transformation (log1p)
- **Encoding**: MultiLabelBinarizer, OneHotEncoder
- **Normalization**: Missing value imputation with mean

## 📁 Project Structure

```
Movie-Analytics-CO3029/
│
├── data movie/
│   └── tmdb_5000_movies.csv      # TMDB movie data
│
├── images/                         # Images/charts directory
│
├── data_mining_nhom_8.ipynb       # Main notebook
│
└── README.md                       # Project documentation
```

## 🚀 Usage Guide

### 1. Environment Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Run the Notebook

1. Open `data_mining_nhom_8.ipynb` in Jupyter Notebook or VS Code
2. Run cells sequentially from top to bottom
3. Ensure the CSV file path matches your system

### 3. Data Processing Steps

#### Step 1: Load and Preprocess Data
```python
csv_path = "data movie/tmdb_5000_movies.csv"
df = load_tmdb_movies_df(csv_path)
```

#### Step 2: Feature Encoding
- Convert movie genres from JSON to ID list
- Encode language (en/other)
- Handle missing values and zeros

#### Step 3: Prepare Dataset
```python
X, y, feature_cols = prepare_dataset_for_sklearn(df, target="revenue")
```

#### Step 4: Split Train/Test Sets
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

#### Step 5: Model Optimization
Using RandomizedSearchCV to find best hyperparameters:
- `n_estimators`: [100, 300, 500, 800]
- `max_depth`: [None, 10, 20, 30, 40, 50]
- `min_samples_split`: [2, 5, 10, 20]
- `min_samples_leaf`: [1, 2, 5, 10]
- `max_features`: ["sqrt", "log2", 0.3, 0.5, None]
- `bootstrap`: [True, False]

## 📈 Results and Evaluation

### Evaluation Metrics

- **MSE (Mean Squared Error)**: Average squared error
- **RMSE (Root Mean Squared Error)**: Square root of MSE
- **R² Score**: Coefficient of determination (model fit quality)

### Results Analysis

1. **Feature Importance**: 
   - Top 20 most important features chart
   - Budget is typically the most important factor

2. **Actual vs Predicted Revenue**:
   - Scatter plot comparing actual and predicted revenue
   - Model performs better on low/medium revenue movies

3. **Residual Analysis**:
   - Analysis of prediction errors
   - High-revenue movies tend to have larger prediction errors

4. **Learning Curve**:
   - Evaluates model learning capability by training set size
   - Detects overfitting/underfitting

### Visualizations

- ✅ Correlation Heatmap: Feature correlation matrix
- ✅ Revenue Distribution: Revenue distribution plot
- ✅ Feature vs Revenue: Relationship between each feature and revenue
- ✅ Residual Plot: Error analysis
- ✅ Max Depth Distribution: Tree depth distribution in Random Forest

## 🔍 Observations and Conclusions

### Strengths
- Model predicts fairly well for low and medium revenue movies
- Budget is the strongest factor affecting revenue
- Log transformation improves model performance

### Limitations
- Model struggles to predict blockbuster (high revenue) movie revenue
- Shows signs of overfitting (RMSE test > RMSE train)
- Important factors like marketing and lead actors not considered

### Future Improvements
- Collect additional features: actors, directors, marketing campaigns
- Experiment with other models: XGBoost, Neural Networks
- Better outlier handling for extremely high-revenue movies
- Apply advanced Feature Engineering techniques

## 👥 Team Information

- **Course**: CO3029 - Data Mining
- **Group**: 8
- **University**: Ho Chi Minh City University of Technology

## 📝 Notes

- This project is for educational and research purposes
- Data is from the public TMDB dataset
- Prediction results are for reference only

## 📞 Contact

For any questions about this project, please contact via GitHub repository.

---

**Last Updated**: November, 2025

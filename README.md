# Software-testing-using-machine-learning-algorithms
This project aims to predict software quality using various machine learning algorithms such as Decision Trees, Random Forests, Logistic Regression, and CNNs. It analyzes software metrics to improve prediction accuracy and help developers identify quality issues early.

[Project Objective]
To build a predictive model that estimates software quality based on static code attributes (complexity, cohesion, coupling, etc.) using machine learning algorithms and ensemble techniques.

# Technologies Used
Python
Scikit-learn
Keras (for CNN)
Pandas, NumPy, Matplotlib
CODEMR dataset
Spring Framework (for backend processing and integration)

# Dataset
The dataset includes class-level, method-level, and package-level quality metrics.
Data source: Kaggle Dataset
Columns: QualifiedName, Complexity, Coupling, LOC, NOC, RFC, WMC, etc.
Many columns include missing values and non-numeric types that are handled in preprocessing.

# Steps to Reproduce Results

# 1. Run the Application
Double-click on the run.bat file to launch the main application interface.

# 2. Upload Dataset
Click the Upload Dataset button and select the dataset file (e.g., 2015-6.csv).

# 3. Visualize Initial Data
The app shows plots of unique values for each feature to give a statistical overview.

# 4. Preprocess Dataset
Click on Preprocess Dataset
Handles missing values and non-numeric fields by replacing them with appropriate counts.
Displays a bar graph of missing values per column.

# 5. Feature Selection
PCA is applied to reduce features from 39 to 30 most important ones.
Dataset split: ~29,542 records for training, ~7,386 for testing.

# 6. Train Machine Learning Models
Click Run Machine Learning Algorithms to train:
Naive Bayes
Decision Tree
Random Forest
Logistic Regression
Gradient Boosting
Bagging Classifiers

# 7. Train Deep Learning Model (CNN)
Click Run CNN Algorithm:
Model runs for 10 epochs, improving accuracy and reducing loss in each iteration.

# 8. View Results
Metrics displayed: Accuracy, Precision, Recall, F1-Score.
Click Comparison Graph to visualize performance across models.

# Results
Achieved up to 98.8% accuracy using ensemble methods and CNN.
CNN model demonstrated improved performance with each epoch.

# Future Enhancements
Use larger datasets for deeper training.
Apply meta-heuristic optimization techniques.
Transform the classification into multi-class problems.
Integrate more advanced deep learning architectures.

# References
[International Research Journal of Modernization in Engineering Technology and Science, 2022]
ISBSG Repositor
CODEMR Tool — dataset license provider

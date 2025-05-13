import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def load_data(filepath):
    """Load and prepare the Iris dataset"""
    df = pd.read_csv(filepath)
    
    
    df['Species'] = df['Species'].apply(lambda x: x.replace('Iris-', ''))
    
    
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        
    return df


iris_df = load_data('Iris.csv')


print("Dataset Info:")
print(f"Shape: {iris_df.shape}")
print("\nFirst 5 rows:")
print(iris_df.head())
print("\nDataset Summary:")
print(iris_df.describe())
print("\nClass Distribution:")
print(iris_df['Species'].value_counts())
print("\nMissing Values:")
print(iris_df.isnull().sum())


def visualize_data(df):
    """Create visualizations for the Iris dataset"""
    
    
    plt.figure(figsize=(15, 10))
    
    
    print("\nCreating pair plot...")
    sns.pairplot(df, hue='Species', height=2.5)
    plt.suptitle('Pair Plot of Iris Features by Species', y=1.02)
    plt.savefig('iris_pairplot.png')
    
    
    print("Creating correlation matrix...")
    plt.figure(figsize=(10, 8))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Iris Features')
    plt.savefig('iris_correlation.png')
    
    
    print("Creating box plots...")
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']):
        plt.subplot(2, 2, i+1)
        sns.boxplot(x='Species', y=feature, data=df)
        plt.title(f'{feature} by Species')
    plt.tight_layout()
    plt.savefig('iris_boxplots.png')
    
    
    print("Creating violin plots...")
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']):
        plt.subplot(2, 2, i+1)
        sns.violinplot(x='Species', y=feature, data=df)
        plt.title(f'{feature} Distribution by Species')
    plt.tight_layout()
    plt.savefig('iris_violinplots.png')
    
    print("Visualizations completed and saved.")


visualize_data(iris_df)


def preprocess_data(df):
    """Split the dataset into features and target and perform train-test split"""
    
    X = df.drop('Species', axis=1)
    y = df['Species']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_data(iris_df)

print(f"\nTraining set shape: {X_train.shape}, {y_train.shape}")
print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

def build_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models on the Iris dataset"""
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=sorted(y_test.unique()), 
                    yticklabels=sorted(y_test.unique()))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {name}')
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
    
    return results

model_results = build_and_evaluate_models(X_train, X_test, y_train, y_test)

def tune_best_model(X_train, y_train, X_test, y_test, results):
    """Perform hyperparameter tuning for the best performing model"""
    
    best_model_name = max(results, key=results.get)
    print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]:.4f}")
    
    param_grids = {
        'Logistic Regression': {
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__solver': ['liblinear', 'lbfgs']
        },
        'Decision Tree': {
            'model__max_depth': [None, 3, 5, 7, 10],
            'model__min_samples_split': [2, 5, 10]
        },
        'Random Forest': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 5, 10],
            'model__min_samples_split': [2, 5, 10]
        },
        'SVM': {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['rbf', 'linear', 'poly']
        },
        'KNN': {
            'model__n_neighbors': [3, 5, 7, 9, 11],
            'model__weights': ['uniform', 'distance']
        }
    }
    
    if best_model_name == 'Logistic Regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif best_model_name == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=42)
    elif best_model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
    elif best_model_name == 'SVM':
        model = SVC(random_state=42)
    else:  
        model = KNeighborsClassifier()
    
    param_grid = param_grids[best_model_name]
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    print(f"\nPerforming grid search for {best_model_name}...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print(f"\nTuned {best_model_name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    if best_model_name in ['Decision Tree', 'Random Forest']:
        feature_importances = best_model.named_steps['model'].feature_importances_
        features = X_train.columns
        
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Feature Importance - {best_model_name}')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{best_model_name.replace(" ", "_").lower()}.png')
    
    return best_model

best_model = tune_best_model(X_train, y_train, X_test, y_test, model_results)

def predict_iris_species(model, sepal_length, sepal_width, petal_length, petal_width):
    """Predict Iris species given the flower measurements"""
    input_data = pd.DataFrame({
        'SepalLengthCm': [sepal_length],
        'SepalWidthCm': [sepal_width],
        'PetalLengthCm': [petal_length],
        'PetalWidthCm': [petal_width]
    })
    
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    
    classes = model.classes_
    proba_dict = {class_name: prob for class_name, prob in zip(classes, probabilities[0])}
    
    return prediction[0], proba_dict

print("\nExample prediction:")
sepal_length = 5.1
sepal_width = 3.5
petal_length = 1.4
petal_width = 0.2

prediction, probabilities = predict_iris_species(
    best_model, sepal_length, sepal_width, petal_length, petal_width
)

print(f"Input: Sepal Length = {sepal_length}cm, Sepal Width = {sepal_width}cm, "
      f"Petal Length = {petal_length}cm, Petal Width = {petal_width}cm")
print(f"Predicted species: {prediction}")
print("Prediction probabilities:")
for species, probability in probabilities.items():
    print(f"  {species}: {probability:.4f}")

def perform_cross_validation(X, y):
    """Perform cross-validation on all models to ensure robust evaluation"""
    print("\nPerforming 5-fold cross-validation on all models...")
    
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Decision Tree': Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier(random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(random_state=42))
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(random_state=42))
        ]),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier())
        ])
    }
    
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"{name}: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")
    
    plt.figure(figsize=(12, 6))
    means = [result['mean'] for result in cv_results.values()]
    stds = [result['std'] for result in cv_results.values()]
    model_names = list(cv_results.keys())
    
    bars = plt.bar(model_names, means, yerr=stds, capsize=10, alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Models')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('5-Fold Cross-Validation Results')
    plt.ylim(0.8, 1.05)  
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('cross_validation_results.png')
    
    return cv_results

cv_results = perform_cross_validation(iris_df.drop('Species', axis=1), iris_df['Species'])

print("\nIris Classification Project Completed!")
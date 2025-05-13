import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from joblib import dump

def load_data(filepath):
    return pd.read_csv(filepath)

def prepare_data(df, target_column):
    ''' Drop the non-encoded 'machine_status' column '''
    if 'machine_status' in df.columns:
        df = df.drop('machine_status', axis=1)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

def convert_timestamp(df, column_name='timestamp'):
    """ Convert datetime column to numerical format (Unix timestamp) """
    df[column_name] = pd.to_datetime(df[column_name])
    df[column_name] = (df[column_name] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    return df

def cross_validate_model(X, y, model, cv_folds=5):
    """ Perform cross-validation and return the average of each metric """
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='weighted', zero_division=0), # Needed to add zero_division parameter to handle some data errors. 
        'recall': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1': make_scorer(f1_score, average='weighted', zero_division=0)
    }
    cv_results = cross_validate(model, X, y, cv=cv_folds, scoring=scoring, return_train_score=True)
    return cv_results

def main():

    # df = load_data('data/processed_sensor_data.csv')
    df = load_data('data/engineered_sensor_data.csv')

    df = convert_timestamp(df, 'timestamp')

    X, y = prepare_data(df, 'machine_status_encoded')

    models = { # Models which are going to be trained and evaluated
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42)
    }

    for name, model in models.items():
        
        print(f"Training {name}...")
        trained_model = model.fit(X, y)  # Train model
        dump(trained_model, f'models/{name}_model.joblib')  # Save model
        
        print(f"Evaluating {name} with cross-validation...") # Terminal debugging
        cv_results = cross_validate_model(X, y, model, cv_folds=5) # Utilize sklearn cross-validate from function
        print(f"Results for {name}:")
        for metric in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']: # Print mean and std for each test conducted in model evaluation
            mean_score = cv_results[metric].mean()
            std_score = cv_results[metric].std()
            print(f"{metric.split('_')[1].capitalize()}: Mean={mean_score:.2f}, Std={std_score:.2f}")

if __name__ == "__main__":
    main()

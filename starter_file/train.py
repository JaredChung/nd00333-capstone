from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# Retrive current run's information

def clean_data(data):
    
    column_fillna = ['city', 'gender', 'relevent_experience',
       'enrolled_university', 'education_level', 'major_discipline',
       'experience', 'company_size', 'company_type', 'last_new_job']
    
    x_df = data.to_pandas_dataframe()
    x_df[column_fillna] = x_df[column_fillna].fillna('N_A')
    y_df = x_df.pop("target").astype('int')
    
    x_df = pd.get_dummies(x_df)

    return x_df, y_df

def main():

    run = Run.get_context()

    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()
    run.log("Regularization Strength: ", np.float(args.C))
    run.log("Max iterations: ", np.int(args.max_iter))

    ws = run.experiment.workspace
    found = False
    key = "traindata"

    if key in ws.datasets.keys(): 
        found = True
        dataset = ws.datasets[key] 

    X, y = clean_data(dataset)
    # Split data into train and test set
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
#     os.makedirs('outputs', exist_ok=True)
#     joblib.dump(model, 'outputs/model.joblib')


if __name__ == '__main__':
    main()
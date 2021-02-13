import numpy as np
import pandas as pd
import joblib
import os
import argparse
from sklearn.ensemble import RandomForestClassifier


if __name__ == "__main__":
    """
    Code to train the model when the script is called
    
    Hyperparameters that can be passed in:
    - class_weight
    - criterion
    - max_depth
    - max_features
    
    """
    parser = argparse.ArgumentParser()
    
    # Hyperparameters: list out the hyperparameters we will be adding in
    parser.add_argument('--class_weight', type=str, default='balanced')
    parser.add_argument('--criterion', type=str, default='entropy')
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--max_features', type=str, default='auto')

    # SageMaker specific arguments. Defaults are set in the environment vars
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    args = parser.parse_args()
    
    # read in the training data
    # take the files in the training data location and read into a dataframe
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)
                  if ".csv" in file]
    if len(input_files) == 0:
        raise ValueError(
            ("There are no files in {}.\n"
             + "This typicall means the channel ({}) was incorrectly specified,\n"
             + "the data specification in S3 was incorrectly specified, or the role\n"
             + "specified does not have permission to access the data"
            ).format(args.train, "train")
        )
    raw_data = [pd.read_csv(file, engine="python") for file in input_files]
    train_data = pd.concat(raw_data)
    
    # labels are in the first column, following SageMaker convention
    y_train = train_data.iloc[:, 0]
    X_train = train_data.iloc[:, 1:]
    
    # Instantiate the hyperparameters with the passed values
    class_weight = args.class_weight
    criterion = args.criterion
    max_depth = args.max_depth
    max_features = args.max_features
    
    # now use scikit-learn's decision tree classifier to train the model
    # (or could have chosen any other model, just would need appropriate hyperparameters)
    clf = RandomForestClassifier(
        class_weight=class_weight
        , criterion=criterion
        , max_depth=max_depth
        , max_features=max_features
    )
    clf = clf.fit(X_train, y_train)
    
    # print the coefficients of the trained classifier, and save coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))
    
    
    
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return clf
    

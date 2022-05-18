from logging import getLogger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pickle
import warnings
import mlflow
from mlflow.sklearn import save_model  # , log_model
import os.path
from sklearn.preprocessing import StandardScaler
import numpy as np
RSEED=42
from sklearn.metrics import mean_absolute_error, mean_squared_error

from feature_engineering import (
    fill_missing_values,
    drop_column,
    transform_altitude,
    altitude_high_meters_mean,
    altitude_mean_log_mean,
    altitude_low_meters_mean,
)

from config import TRACKING_URI, EXPERIMENT_NAME

warnings.filterwarnings("ignore")
logger = getLogger(__name__)


def __get_data():
    logger.info("Getting the data")
    #accessing the directory 
    os.path.dirname('C:/Users/haritha_retnakaran/Documents/Neufische/CapStone')

    #train data
    df = pd.read_csv('./data/train.csv')

    #more features:linear interpolation
    more_var = pd.read_csv('./data/utms.csv')

    #feature engineering
    logger.info("Feature engineering on data")
    df['vol'] = df['time_step'] * df['u_in']
    df['vol'] = df.query('u_out==0').groupby('breath_id')['vol'].cumsum()
    df['rtime']=df['time_step'].apply(lambda x: round(x,3))
    df.query('id%80==2')
    df['step_id']=df.id.apply(lambda x: x%80)
    df=df.query('u_out==0')
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['minus_one']=-1.0
    df['plus_one']=1.0
    df['exponent']=(df['minus_one']*df['time_step'])/(df['R']*df['C'])
    df['factor']=np.exp(df['exponent'])
    df['vf']=(df['u_in_cumsum']*df['R'])/df['factor']
    df['vt']=0
    df.loc[df['time_step'] != 0, 'vt']=df['vol']/(df['C']*(df['minus_one']*df['factor']+df['plus_one']))
    df['v']=df['vf']+df['vt']

    #merge 2 data frame
    new_df = df.merge(more_var, how='inner', on='id')

    #columns to be scaled
    cols=['v','u_in_cumsum','vol','pressure', 'u_in', 'R', 'C', 'time_step', 'step_id', 'R*C', '1/R*C', 'asc', 'mean',
       'std', 'utm1', 'utm2', 'utm3']
    red_df=new_df[cols]

    #train test split
    target="pressure"
    X = red_df.drop(target, axis=1)
    y = red_df.loc[:,target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=RSEED)
    
    #scale the data
    scaler= StandardScaler()
    #variables to be scaled
    col_names = ['v', 'u_in_cumsum', 'vol', 'u_in',	'R', 'C', 'R*C', 'mean', 'std', 'utm1', 'utm2', 'utm3']
    X_train[col_names] = scaler.fit_transform(X_train[col_names])
    X_test[col_names] = scaler.transform(X_test[col_names])
    return X_train, X_test, y_train, y_test

def __compute_and_log_metrics(
    y_true: pd.Series, y_pred: pd.Series, prefix: str = "train"
):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    logger.info(
        "Linear Regression performance on "
        + str(prefix)
        + " set: MSE = {:.1f}, set: MAE = {:.1f} ,R2 = {:.1%},".format(mse, mae, r2)
    )
    mlflow.log_metric(prefix + "-" + "MSE", mse)
    mlflow.log_metric(prefix + "-" + "MAE", mae)
    mlflow.log_metric(prefix + "-" + "R2", r2)
    return mse, mae, r2

def run_training():
    logger.info(f"Getting the data")
    X_train, X_test, y_train, y_test = __get_data()

    logger.info("Training simple model and tracking with MLFlow")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    # model
    logger.info("Training a simple linear regression")
    with mlflow.start_run():
        reg = LinearRegression().fit(X_train, y_train)
        y_train_pred = reg.predict(X_train)

        __compute_and_log_metrics(y_train, y_train_pred)

        y_test_pred = reg.predict(X_test)
        __compute_and_log_metrics(y_test, y_test_pred, "test")

        logger.info("prediction on test data finished")
        # saving the model
        # logger.info("Saving model in the model folder")
        # path = "models/linear"
        # save_model(sk_model=reg, path=path)
        # logging the model to mlflow will not work without a AWS Connection setup.. too complex for now
        
if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
    logger.setLevel(logging.INFO)

    run_training()
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
RSEED = 42

#create a main method
if __name__ == "__main__":
    #give experiment name
    mlflow.set_experiment(experiment_name="mlflow_demo")
    #read the data
    df = pd.read_csv('data/train.csv')
    print('Loaded training model')

    #feature engineering
    df['vol'] = df['time_step'] * df['u_in']
    df['vol'] = df.query('u_out==0').groupby('breath_id')['vol'].cumsum()
    df['rtime']=df['time_step'].apply(lambda x: round(x,3))
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

    #useful features
    cols=['v','u_in_cumsum','vol','pressure', 'u_in', 'R', 'C', 'time_step', 'step_id']
    red_df=df[cols]

    #train test split
    target="pressure"
    X = red_df.drop(target, axis=1)
    y = red_df.loc[:,target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=RSEED)

    #scaling the data
    scaler= StandardScaler()
    col_names = ['v', 'u_in_cumsum', 'vol', 'u_in',	'R', 'C']
    X_train_num=X_train[col_names].copy()
    X_train[col_names] = scaler.fit_transform(X_train[col_names])
    X_test[col_names] = scaler.transform(X_test[col_names])
    print('completed feature scaling')

    #create a linear regressor
    # Initalizing and training the model 
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    print('model trained')

    # Making predictions 
    y_pred = lin_reg.predict(X_test)
    # Evaluting model
    mae =  mean_absolute_error(y_test, y_pred).round(2)
    mse = mean_squared_error(y_test, y_pred).round(2)
    print('MAE:', mae)
    print('MSE', mse)

    #what to track everytime we run the model
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    #track model for each run
    mlflow.sklearn.log_model(lin_reg, "model")


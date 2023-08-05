import pandas as pd
from sklearn.preprocessing import  StandardScaler, MaxAbsScaler, MinMaxScaler,RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
                            roc_auc_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


scaler = {
"standard" : StandardScaler(),
"max_abs" : MaxAbsScaler(),
"min_max" : MinMaxScaler(),
"robust" : RobustScaler()
}

 # List of available model classes
model_classes = {
    'linearregression': LinearRegression,
    'logisticregression': LogisticRegression,
    'ridge': Ridge,
    'lasso': Lasso,
    'elasticnet': ElasticNet,
    'sgdregressor': SGDRegressor,
    'decisiontreeclassifier': DecisionTreeClassifier,
    'decisiontreeregressor': DecisionTreeRegressor,
    'randomforestclassifier': RandomForestClassifier,
    'randomforestregressor': RandomForestRegressor,
    'gradientboostingclassifier': GradientBoostingClassifier,
    'gradientboostingregressor': GradientBoostingRegressor,
    'adaboostclassifier': AdaBoostClassifier,
    'adaboostregressor': AdaBoostRegressor,
    'svc': SVC,
    'svr': SVR,
    'mlpclassifier': MLPClassifier,
    'mlpregressor': MLPRegressor,
    'kneighborsclassifier': KNeighborsClassifier,
    'kneighborsregressor': KNeighborsRegressor,
}
    

def proc_data(df):
    # Remove object columns having more than 10 unique values
    object_columns = df.select_dtypes(include='object').columns
    columns_to_drop = [col for col in object_columns if df[col].nunique() > 10]
    df.drop(columns_to_drop, axis=1, inplace=True)
    # Remove 50% NaN values columns
    nan_percent = df.isnull().mean() * 100
    valid_columns = nan_percent[nan_percent <= 50].index.tolist()
    df = df[valid_columns]
    # Fill_na with modes of columns
    modes = df.mode().iloc[0]
    df = df.fillna(modes)
    # Transform columns to onehot encoding
    object_columns = object_columns.difference(columns_to_drop)
    df_encoded = pd.get_dummies(df[object_columns], drop_first=True)
    df = pd.concat([df.drop(object_columns, axis=1), df_encoded], axis=1)

    return df

def normalize(df,option="standard"):
    norm_df =  scaler[option].fit_transform(df)
    norm_df = pd.DataFrame(norm_df,columns=df.columns)
    return norm_df

def transform_stats(df):
    desc = df.describe()
    desc=desc.drop([desc.index[4]  , desc.index[6]])
    na_percent = df.isnull().mean() * 100
    desc.loc["NaN %"] = na_percent
    return desc.rename(index={'50%': 'median'})

def get_X_y(df,target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X,y

def is_classification(target):
    unique_values = set(target)
    if len(unique_values) <= 10: 
        return True
    return False

def classification_report(y_test,y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }

def regression_report(y_test,y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        'Mean Squared Error': mse,
        'R-squared': r2
    }

def report(y_test,y_pred):
    if is_classification(y_test):
        return classification_report(y_test,y_pred)
    return regression_report(y_test,y_pred)
        



def get_model(model_name:str):
    model_name = model_name.lower()

    model_class = model_classes[model_name] 
    model = model_class()
    return model

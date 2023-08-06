import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler, MaxAbsScaler, MinMaxScaler,RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
                            roc_auc_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
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
classification_models = {
    'ada_boost': AdaBoostClassifier,
    'decision_tree': DecisionTreeClassifier,
    'gradient_boosting': GradientBoostingClassifier,
    'kneighbors': KNeighborsClassifier,
    'linear': LogisticRegression,
    'mlp': MLPClassifier,
    'random_forest': RandomForestClassifier,
    'support_vector': SVC
}
regression_models={
    'ada_boost': AdaBoostRegressor,
    'decision_tree': DecisionTreeRegressor,
    'gradient_boosting': GradientBoostingRegressor,
    'kneighbors': KNeighborsRegressor,
    'linear': LinearRegression,
    'mlp': MLPRegressor,
    'random_forest': RandomForestRegressor,
    'support_vector': SVR,
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


class MLDataPipeline:
    def __init__(self, X,y, test_size=0.3, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.is_classification = len(set(y)) <= 10 
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data(X,y)

    def _prepare_data(self,X,y):
        return train_test_split(X, y, test_size=self.test_size, \
                                random_state=self.random_state)

    def train_model(self, model):
        model.fit(self.X_train, self.y_train)

    def evaluate_model(self, model):
        y_pred = model.predict(self.X_test)
        if self.is_classification:
            return classification_report(self.y_test, y_pred)
        return regression_report(self.y_test, y_pred)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

class MLDataPipeline:
    def __init__(self, data, target_column, test_size=0.3, random_state=42):
        self.data = data
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self._prepare_data()

    def _prepare_data(self):
        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=self.random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess_data(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)

    def select_features(self, num_features=10):
        feature_selector = SelectKBest(f_classif, k=num_features)
        self.X_train = feature_selector.fit_transform(self.X_train, self.y_train)
        self.X_val = feature_selector.transform(self.X_val)
        self.X_test = feature_selector.transform(self.X_test)

    def train_model(self, model):
        model.fit(self.X_train, self.y_train)

    def evaluate_model(self, model):
        y_pred = model.predict(self.X_val)
        report = classification_report(self.y_val, y_pred)
        return report

    def hyperparameter_tuning(self, model, param_grid, cv=3):
        grid_search = GridSearchCV(model, param_grid, cv=cv)
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        return best_model

# Example usage
data = pd.read_csv('your_data.csv')
target_column = 'target'
pipeline = MLDataPipeline(data, target_column)
pipeline.preprocess_data()
pipeline.select_features()
model = RandomForestClassifier(random_state=42)
pipeline.train_model(model)
evaluation_report = pipeline.evaluate_model(model)
print("Validation Classification Report:\n", evaluation_report)

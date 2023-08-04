from sklearn.model_selection import train_test_split
from utils.preprocessing import report

class MLDataPipeline:
    def __init__(self, X,y, test_size=0.3, random_state=42):
        self.X, self.y = X, y
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_val, self.y_train, self.y_val = self._prepare_data()

    def _prepare_data(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_val,  y_train, y_val
    def train_model(self, model):
        model.fit(self.X_train, self.y_train)

    def evaluate_model(self, model):
        y_pred = model.predict(self.X_val)
        return report(self.y_val, y_pred)

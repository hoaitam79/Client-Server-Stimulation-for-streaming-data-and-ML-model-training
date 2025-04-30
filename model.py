import os
import pickle
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List
from joblibspark import register_spark
import warnings
from sklearn.utils import parallel_backend
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from pyspark.ml.classification import LinearSVC, RandomForestClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.ml.linalg import DenseVector, Vector, Vectors
from pyspark.sql.dataframe import DataFrame

warnings.filterwarnings('ignore')
register_spark()

class BaseModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.best_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
        # self.models_dir = "saved_models"
        # os.makedirs(self.models_dir, exist_ok=True)

    @abstractmethod
    def train(self, df):
        pass

    @abstractmethod
    def predict(self, df):
        pass

    def evaluate(self, y_true, y_pred) -> List:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return y_pred, accuracy, precision, recall, f1


class SVM(BaseModel):
    def __init__(self, loss="hinge", penalty="l2", max_iter=100, tol=1e-4):
        super().__init__("SVM")
        self.loss = loss
        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol
        self.model = LinearSVC(loss=loss, penalty=penalty, random_state=0)

    def train(self, df: DataFrame) -> List:
        X = np.array(df.select("image").collect()).reshape(-1, 3072)
        y = np.array(df.select("label").collect()).reshape(-1)

        with parallel_backend("spark", n_jobs=4):
            self.model.fit(X, y)

        predictions = self.model.predict(X)
        predictions = np.array(predictions)
        return self.evaluate(y, predictions)

    def predict(self,  df: DataFrame) -> List:
        X = np.array(df.select("image").collect()).reshape(-1, 3072)
        y = np.array(df.select("label").collect()).reshape(-1)

        with parallel_backend("spark", n_jobs=4):
            self.model.fit(X, y)

        predictions = self.model.predict(X)
        predictions = np.array(predictions)
        return self.evaluate(y, predictions)


class RandomForest(BaseModel):
    def __init__(self, num_trees=100, max_depth=10, seed=42):
        super().__init__("RandomForest")
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.seed = seed
        self.model = RandomForestClassifier(n_estimators=self.num_trees,
                                            max_depth=self.max_depth,
                                            random_state=self.seed)

    def train(self, df: DataFrame) -> List:
        X = np.array(df.select("image").collect()).reshape(-1, 3072)
        y = np.array(df.select("label").collect()).reshape(-1)

        with parallel_backend("spark", n_jobs=4):
            self.model.fit(X, y)

        predictions = self.model.predict(X)
        return self.evaluate(y, predictions)

    def predict(self, df: DataFrame) -> List:
        if self.model is None:
            raise ValueError("Model not trained yet!")

        X = np.array(df.select("image").collect()).reshape(-1, 3072)
        y = np.array(df.select("label").collect()).reshape(-1)

        predictions = self.model.predict(X)
        return self.evaluate(y, predictions)



class LogisticRegressionModel(BaseModel):
    def __init__(self, max_iter=100, reg_param=0.3, elastic_net_param=0.8):
        super().__init__("LogisticRegression")
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.elastic_net_param = elastic_net_param
        self.model = LogisticRegression(max_iter=self.max_iter,
                                        C=1 / self.reg_param if self.reg_param != 0 else 1e12,
                                        penalty='elasticnet',
                                        l1_ratio=self.elastic_net_param,
                                        solver='saga')

    def train(self, df: DataFrame) -> List:
        X = np.array(df.select("image").collect()).reshape(-1, 3072)
        y = np.array(df.select("label").collect()).reshape(-1)

        with parallel_backend("spark", n_jobs=4):
            self.model.fit(X, y)

        predictions = self.model.predict(X)
        return self.evaluate(y, predictions)

    def predict(self, df: DataFrame) -> List:
        if self.model is None:
            raise ValueError("Model not trained yet!")

        X = np.array(df.select("image").collect()).reshape(-1, 3072)
        y = np.array(df.select("label").collect()).reshape(-1)

        predictions = self.model.predict(X)
        return self.evaluate(y, predictions)
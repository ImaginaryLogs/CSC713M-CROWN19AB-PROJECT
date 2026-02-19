import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from enum import Enum
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

class rf_antibody:
    def __init__(self, target: str, df: pd.DataFrame) -> None:
        self.exclude = [f'is_binding_{target}', f'is_neutral_{target}', 'name']
        self.rf_classifier_neutral: RandomForestClassifier
        self.rf_classifier_binding: RandomForestClassifier
        self.target = target
        self.df = df

    def train(self, X:pd.DataFrame, y: pd.Series):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        rf_antibody = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_antibody.fit(X_train, y_train)
        y_pred = rf_antibody.predict(X_test)
        return rf_antibody, classification_report(y_test, y_pred)
        
    def get_x_y(self, label: str):
        bool_not_unknown = self.df[label] != 2
        X = self.df[bool_not_unknown].drop(columns=self.exclude)
        y = self.df[bool_not_unknown][label]
        return X, y
        
    def train_neutral(self):
        label = f'is_neutral_{self.target}'
        X, y = self.get_x_y(label)
        self.rf_classifier_neutral, report = self.train(X, y)
        return self.rf_classifier_neutral, report
    
    def train_binding(self):
        label = f'is_binding_{self.target}'
        X, y = self.get_x_y(label)
        self.rf_classifier_binding, report = self.train(X, y)
        return self.rf_classifier_binding, report
        
        
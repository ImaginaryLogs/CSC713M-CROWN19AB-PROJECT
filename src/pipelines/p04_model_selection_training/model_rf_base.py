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
        # Define what to exclude once. 
        # Add any other string columns here (e.g., 'CDRH3', 'VHorVHH')
        self.target = target
        self.exclude = [f'is_binding_{target}', f'is_neutral_{target}', 'name', 'CDRH3', 'CDRL3']
        self.df = df
        self.rf_classifier_neutral = None
        self.rf_classifier_binding = None
        self.rf_neutral_X_cols = None
        self.rf_binding_X_cols = None

    def get_x_y(self, label: str):
        # 1. Filter out unknowns
        bool_not_unknown = self.df[label] != 2
        filtered_df = self.df[bool_not_unknown]
        
        # 2. Separate Features (X) and Target (y)
        # errors='ignore' prevents crashes if a column was already dropped
        X = filtered_df.drop(columns=self.exclude, errors='ignore')
        y = filtered_df[label]
        
        # 3. CRITICAL: Select only numeric features
        # This removes any lingering sequences or IDs that would break Scikit-Learn
        X = X.select_dtypes(include=[np.number])
        
        return X, y

    def train_binding(self):
        label = f'is_binding_{self.target}'
        X, y = self.get_x_y(label)
        self.rf_classifier_binding, report, self.rf_binding_X_cols = self.train(X, y)
        return self.rf_classifier_binding, report

    def train(self, X:pd.DataFrame, y: pd.Series):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        rf_antibody = RandomForestClassifier(n_estimators=200, max_features='sqrt', min_samples_leaf=5, random_state=42, class_weight='balanced')
        rf_antibody.fit(X_train, y_train)
        y_pred = rf_antibody.predict(X_test)
        return rf_antibody, classification_report(y_test, y_pred), X_train.columns
        
    def train_neutral(self):
        label = f'is_neutral_{self.target}'
        X, y = self.get_x_y(label)
        self.rf_classifier_neutral, report, self.rf_neutral_X_cols = self.train(X, y)
        return self.rf_classifier_neutral, report
    
    def plot_feature_importance(self, model, feature_names, prediction_label, top_n=20):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        names_list = list(feature_names)
        
        plt.figure(figsize=(10, 8))
        plt.title(f"Top {top_n} Features Influencing {prediction_label} Prediction")
        
        sorted_names = [names_list[i] for i in indices]
        
        plt.barh(range(len(indices)), importances[indices], align='center', color='teal')
        plt.yticks(range(len(indices)), sorted_names)
        plt.xlabel('Relative Importance (Gini Impurity Reduction)')
        plt.tight_layout()
        plt.show()
    
    def plot_importance_binding(self):
        return self.plot_feature_importance(self.rf_classifier_binding, self.rf_binding_X_cols, "Binding")
    
    def plot_importance_neutral(self):
        return self.plot_feature_importance(self.rf_classifier_neutral, self.rf_neutral_X_cols, "Neutral")
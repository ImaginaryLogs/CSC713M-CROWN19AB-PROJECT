from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression as ScikitLR
from sklearn.naive_bayes import GaussianNB
from typing import Union
import numpy as np
import types
import etc.config

from src.utils.logging_module import get_logging

logger = get_logging(__name__)

class ClassicMlClassifer:
    def __init__(self, classifier_model: type[ClassifierMixin], **kwargs: object) -> None:
        """Initialize the preprocessing steps for a Classifier ML"

        Args:
            classifier_model (type[ClassifierMixin]): A scikit-learn model.

        Raises:
            TypeError: when `classifier_model` is not from a scikit-learn library.
        """
        self.support_sample_weight = True
        self.scaler_cls: type = StandardScaler
        
        if not issubclass(classifier_model, ClassifierMixin):
            raise TypeError(f"{classifier_model} must be a Scikit-Model.")
        
        steps: list[tuple] = []
        
        if self.scaler_cls is not None:
            steps.append(("scaler", self.scaler_cls()))
        steps.append(("cls", classifier_model(**kwargs)))
        
        self.model = Pipeline(steps)
        
    def __str__(self) -> str:
        return f"Classical ML: {self.__class__.__name__}"

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> None:
        fit_kwargs = {}
        if sample_weight is not None:
            if self.support_sample_weight:
                fit_kwargs["clf__sample_weight"] = sample_weight
            
        self.model.fit(X, y, **fit_kwargs)
    
    def predict(self, X: np.ndarray) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        return self.model.predict(X) 
    
    def save(self, name: str | None = None) -> None:
        
        path = str()
        print(f"Model \"{name}\" saved to {path}")
    
    def load(self, name: str | None = None) -> None:
        pass

class KNearestNeighbors(ClassicMlClassifer):
    def __init__(self, random_state: int | None = None, **kwargs: object) -> None:
        super().__init__(KNeighborsClassifier, **kwargs)
        self.support_sample_weight = False
        
class SupportVectorMachine(ClassicMlClassifer):
    def __init__(self, random_state: int | None = None, **kwargs: object) -> None:
        super().__init__(SVC, **kwargs)
        
class RandomForest(ClassicMlClassifer):
    def __init__(self, random_state: int | None = None, **kwargs: object) -> None:
        super().__init__(RandomForestClassifier, **kwargs)

class LogisticRegression(ClassicMlClassifer):
    def __init__(self, random_state: int | None = None, **kwargs: object) -> None:
        super().__init__(ScikitLR, **kwargs)

class NaiveBayes(ClassicMlClassifer):
    def __init__(self, random_state: int | None = None, **kwargs: object) -> None:
        super().__init__(GaussianNB, **kwargs)


CLASSICAL_ML_CLASSIFIER: dict[str, type[ClassicMlClassifer]] = {
    "knn" : KNearestNeighbors,
    "lr" : LogisticRegression,
    "rf" : RandomForest,
    "svm" : SupportVectorMachine,
    "nb" : NaiveBayes
}
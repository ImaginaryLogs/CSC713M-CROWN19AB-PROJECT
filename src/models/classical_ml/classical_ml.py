from sklearn.base import ClassifierMixin, BaseEstimator 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression as ScikitLR
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from typing import Any, Callable, Type, Union, cast
from sklearn.decomposition import PCA
import numpy as np
import types
import joblib
from etc import constants_training
from pathlib import Path
from src.utils.logging_module import get_logging

logger = get_logging(__name__)

ClassifierType = Union[Type[ClassifierMixin], CalibratedClassifierCV]

class ClassicMlClassifer:
    def __init__(self, classifier_model: ClassifierType, has_pca: bool = False, **kwargs: Any) -> None:
        """Initialize the preprocessing steps for a Classifier ML"

        Args:
            classifier_model (type[ClassifierMixin]): A scikit-learn model.

        Raises:
            TypeError: when `classifier_model` is not from a scikit-learn library.
        """
        self.support_sample_weight = True
        self.scaler_cls: type = StandardScaler
        
        steps: list[tuple] = []
        
        if self.scaler_cls is not None:
            steps.append(("scaler", self.scaler_cls()))

        if has_pca:
            steps.append(("pca", PCA(n_components=constants_training.PCA_N_COMPONENTS, random_state=42))) # Reduce 32k to n uncorrelated features    
        
        if isinstance(classifier_model, type):
            if not issubclass(classifier_model, BaseEstimator):
                raise TypeError(f"{classifier_model} must be a Scikit-Learn Class.")
            
            model_instance = cast(Any, classifier_model)(**kwargs)
            steps.append(("cls", model_instance))
        else:
            # It's an instance (like CalibratedClassifierCV), use it directly
            steps.append(("cls", classifier_model))
        
        self.model = Pipeline(steps)
        
    def __str__(self) -> str:
        return f"Classical ML: {self.model.named_steps['cls'].__class__.__name__}"

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        fit_kwargs = {}
        if sample_weight is not None and self.support_sample_weight:
            if self.support_sample_weight:
                fit_kwargs["cls__sample_weight"] = sample_weight
            
        self.model.fit(X, y, **fit_kwargs)
    
    def predict(self, X: np.ndarray):
        return self.model.predict(X) 
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def save(self, directory: str | Path | None = None , name: str | None = None) -> None:
        save_dir = Path(directory) if directory else constants_training.ART_MODELS_DIR
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        ml_name = name or self.__class__.__name__
        path = str(save_dir / f"{ml_name}.joblib")
        joblib.dump(self.model, path)
        logger.info(f"Model saved as \"{name}\" saved to {path}")
    
    @classmethod   
    def load(cls, name: str | None = None) -> None:
        model_name = name or cls.__name__.lower()
        path = constants_training.ART_MODELS_DIR / f"{model_name}.joblib"
        if not path.exists(): return None
        instance = cls.__new__(cls)
        instance.model = joblib.load(str(path))
        logger.info(f"Model loaded from {path}")

class KNearestNeighbors(ClassicMlClassifer):
    def __init__(self, random_state: int | None = 42, has_pca:bool=False, **kwargs: object) -> None:
        super().__init__(KNeighborsClassifier, has_pca=has_pca, **kwargs)
        self.support_sample_weight = False
        
class SupportVectorMachine(ClassicMlClassifer):
    def __init__(self, random_state: int | None = 42, has_pca:bool=False, **kwargs: Any) -> None:
        # 1. Clean up kwargs for LinearSVC
        kwargs.pop('probability', None) 
        
        # 2. Build the calibrated fast SVM
        base_model = LinearSVC(random_state=random_state, **kwargs)
        # CalibratedClassifierCV is an instance here
        calibrated_model = CalibratedClassifierCV(base_model, cv=3) 
        
        # 3. Pass the instance to super
        super().__init__(classifier_model=calibrated_model, max_iter=constants_training.SVM_MAX_ITER, has_pca=has_pca)
        
class RandomForest(ClassicMlClassifer):
    def __init__(self, random_state: int | None = 42, has_pca:bool=False, **kwargs: object) -> None:
        super().__init__(RandomForestClassifier, random_state=random_state, has_pca=has_pca,**kwargs)

class LogisticRegression(ClassicMlClassifer):
    def __init__(self, random_state: int | None = 42, has_pca:bool=False, **kwargs: object) -> None:
        super().__init__(ScikitLR, random_state=random_state, has_pca=has_pca, **kwargs)

class NaiveBayes(ClassicMlClassifer):
    def __init__(self, random_state: int | None = 42, has_pca: bool = False, **kwargs: object) -> None:
        super().__init__(GaussianNB, has_pca=has_pca, **kwargs)


CLASSICAL_ML_CLASSIFIER: dict[str, Callable[...,ClassicMlClassifer]] = {
    "knn" : KNearestNeighbors,
    "lr" : LogisticRegression,
    "rf" : RandomForest,
    "svm" : SupportVectorMachine,
    "nb" : NaiveBayes
}
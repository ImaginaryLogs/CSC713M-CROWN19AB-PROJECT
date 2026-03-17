from src.pipelines.p03_preprocessing.p01_label_preprocessor import Label_Preprocess
from src.pipelines.p03_preprocessing.p02_naive_preprocessor import Naive_Genetics_Preprocess
from src.pipelines.p03_preprocessing.p03_motif_3kmer_preprocessor import Motif_3kmer_Preprocess
from src.pipelines.p03_preprocessing.p04_motif_conjoint_preprocessor import Motif_Conjoint_Preprocess
from src.pipelines.p03_preprocessing.p05_biochemical_preprocessor import Biochemical_Preprocess
from etc import config
from sklearn.model_selection import train_test_split


class PreprocessorOrchestrator:
    def __init__(self):
        self.p01_label_worker = Label_Preprocess()
        self.p02_naive_worker = Naive_Genetics_Preprocess()
        self.p03_motif_3kmer_worker = Motif_3kmer_Preprocess()
        self.p04_motif_conjoint_worker = Motif_Conjoint_Preprocess()
        self.p05_biochemical_worker = Biochemical_Preprocess()
    
    def run(self):
        INPUT_PATH = config.RAWDATA_FILE
        OUTPUT_PATH = config.FEATURES_DIR
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        



def main():
    INPUT_PATH = config.RAWDATA_FILE
    OUTPUT_PATH = config.FEATURES_DIR 
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE1 = OUTPUT_PATH / "test_label.csv"
    OUTPUT_FILE2 = OUTPUT_PATH / "test_genetics.csv"
    worker1 = Label_Preprocess()
    worker1.run_pipeline(INPUT_PATH, OUTPUT_FILE1)
    worker2 = Naive_Genetics_Preprocess()
    worker2.run_pipeline(INPUT_PATH, OUTPUT_FILE2)

if __name__ == "__main__":
    main()
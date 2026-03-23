import wandb
import pandas as pd
from src.features.feature_factory import FeatureType, FeatureFactory
from sklearn.pipeline import Pipeline
from src.data_module.data_module import DataModule
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import shap
import extraction
from concurrent.futures import ThreadPoolExecutor
import shutil
import joblib
all_audit_data = []

def find_error_drivers(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, feature_names: list, n_features=10):
    """
    Identifies features that differ most between Correct and Error cases.
    """
    tp_idx = (y_true == 1) & (y_pred == 1)
    fp_idx = (y_true == 0) & (y_pred == 1) # The 'Liar' features
    fn_idx = (y_true == 1) & (y_pred == 0) # The 'Missing' features
    
    insights = {}

    # Analyze False Positives (FP) vs True Positives (TP)
    # We want to know: "What feature is present in FPs that makes them look like TPs?"
    if np.any(fp_idx) and np.any(tp_idx):
        fp_means = X[fp_idx].mean(axis=0)
        tp_means = X[tp_idx].mean(axis=0)
        
        # Calculate the Delta/Difference
        delta_fp = fp_means - tp_means
        # Get indices of features where FP is much higher than TP
        top_fp_drivers = np.argsort(np.abs(delta_fp))[-n_features:]
        
        insights['FP_Drivers'] = [
            {"feature": feature_names[i], "delta": delta_fp[i]} 
            for i in reversed(top_fp_drivers)
        ]

    # Analyze False Negatives (FN) vs True Positives (TP)
    if np.any(fn_idx) and np.any(tp_idx):
        fn_means = X[fn_idx].mean(axis=0)
        # Delta: What is missing in FN that is present in TP?
        delta_fn = tp_means - fn_means
        top_fn_drivers = np.argsort(np.abs(delta_fn))[-n_features:]
        
        insights['FN_Drivers'] = [
            {"feature": feature_names[i], "delta": delta_fn[i]} 
            for i in reversed(top_fn_drivers)
        ]

    return insights

def analyze_specific_errors(run_df, feature_lookup, feature_names):
    """Helper to slice the lookup and run driver analysis."""
    # Filter categories
    tps = run_df[run_df['Category'] == 'TP']
    fps = run_df[run_df['Category'] == 'FP']
    fns = run_df[run_df['Category'] == 'FN']
    
    # Extract feature vectors from lookup
    tp_feats = np.array([feature_lookup[n] for n in tps['Antibody_Name'] if n in feature_lookup])
    fp_feats = np.array([feature_lookup[n] for n in fps['Antibody_Name'] if n in feature_lookup])
    fn_feats = np.array([feature_lookup[n] for n in fns['Antibody_Name'] if n in feature_lookup])
    
    # Prepare data for find_error_drivers
    # We stack them to create a mini-X matrix for just this run
    if len(fp_feats) > 0 and len(tp_feats) > 0:
        # Create a temporary matrix: [TPs then FPs then FNs]
        X_sub = np.vstack([tp_feats, fp_feats])
        y_true = np.array([1]*len(tp_feats) + [0]*len(fp_feats))
        y_pred = np.ones(len(tp_feats) + len(fp_feats))
        
        # If we have FNs, add them to the stack
        if len(fn_feats) > 0:
            X_sub = np.vstack([X_sub, fn_feats])
            y_true = np.append(y_true, [1]*len(fn_feats))
            y_pred = np.append(y_pred, [0]*len(fn_feats))
            
        return find_error_drivers(X_sub, y_true, y_pred, feature_names)
    return None

def summarize_all_errors(master_df: pd.DataFrame, feature_names: list, n_features: int = 10):
    """
    Automates error driver analysis across all runs in the master_df.
    Groups by data configuration to minimize DataModule re-loading.
    """
    summary_results = []
    
    # 1. Group by unique data configurations (Task + Oversampling + Synthetic)
    # We do this because all runs in this group share the same feature_lookup
    config_groups = master_df.groupby(['task', 'oversample', 'has_synthetic'])
    
    for (task, oversample, has_synthetic), group_df in config_groups:
        
        # 2. Load the features for this specific data configuration
        task = str(group_df['task'].iloc[0])
        oversample = int(group_df['oversample'].iloc[0])
        has_synthetic = bool(group_df['has_synthetic'].iloc[0])
        
        try:
            # Now passing explicitly typed variables
            lookup = extraction.search(task, oversample, has_synthetic)
        except Exception as e:
            print(f"Error loading features for {task}: {e}")
            continue
        
        # 3. Analyze every run (model) within this configuration
        for run_id in group_df['run_id'].unique():
            run_subset = group_df[group_df['run_id'] == run_id]
            model_type = run_subset['model'].iloc[0]
            run_name = run_subset['name'].iloc[0]
            
            # Use the provided analysis logic to get drivers
            drivers = analyze_specific_errors(run_subset, lookup, feature_names)

            if drivers:
                for category, driver_list in drivers.items():
                    for d in driver_list:
                        summary_results.append({
                            "run_name": run_name,
                            "model": model_type,
                            "category": category,
                            "feature": d['feature'],
                            "delta": d['delta'],
                            "task": task
                        })

    return pd.DataFrame(summary_results)

k = FeatureFactory()
types = [
    FeatureType.NAIVE,
    FeatureType.MOTIF_CONJOINT,
    FeatureType.BIOCHEMICAL
]
feature_names = FeatureFactory().get_all_feature_names(types)
print(len(feature_names))
# Define cache paths in your project root
CACHE_DIR = Path("artifacts/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STATS_CACHE = CACHE_DIR / "master_stats.csv"
AUDIT_CACHE = CACHE_DIR / "master_audit.csv"

def get_master_data(force_refresh=False):
    if STATS_CACHE.exists() and AUDIT_CACHE.exists() and not force_refresh:
        print("Loading data from Local Cache...")
        master_stats_df = pd.read_csv(STATS_CACHE)
        master_audit_df = pd.read_csv(AUDIT_CACHE)
        return master_stats_df, master_audit_df
    
    api = wandb.Api()
    entity, project = "logarithmicpresence-de-la-salle-university", "CSC713M_MSINTSY"
    runs = api.runs(f"{entity}/{project}")

    print("Cache not found or Refresh triggered. Downloading from WandB...")
    
    metadata_list = []
    for run in runs:
        if run.state == "finished":
            s = run.summary
            
            # Helper to find the best available metric key
            def get_metric(keys):
                for k in keys:
                    val = s.get(k)
                    if val is not None and not pd.isna(val):
                        return val
                return 0

            metadata_list.append({
                "run_id": run.id,
                "name": run.name,
                "model": run.config.get("model"),
                "task": run.config.get("task"),
                "has_pca": run.config.get("has_pca"),
                "oversample": run.config.get("oversampling_ratio"),
                "has_synthetic": run.config.get("has_synthetic_cdr"),
                "class_weight": run.config.get("class_weight"),
                
                # Broaden the search for F1
                "val_f1": get_metric(["val_f1", "val/f1", "epoch/val_f1", "val_f1_epoch"]),
                "val_recall": get_metric(["val_recall", "val/recall", "epoch/val_recall"]),
                "val_precision": get_metric(["val_precision", "val/precision", "epoch/val_precision"]),
                "val_acc": get_metric(["val_acc", "val/acc", "epoch/val_acc"]),
                "val_mcc": get_metric(["val_mcc", "val/mcc", "epoch/val_mcc", "val/mcc"])
            })
    # Final Master Audit DataFrame
    master_stats_df = pd.DataFrame(metadata_list)
    
    def fetch_run_audit(run):
        """Worker function to handle a single run's artifact download"""
        if run.state != "finished":
            return None
            
        try:
            model_name = run.config.get("model")
            task = run.config.get("task", "unknown_task")
            artifacts = run.logged_artifacts()
            run_id = run.id
            
            # 1. Define your organized destination
            # Structure: runs/neutralization/deepNn/dk18knxm/
            output_dir = Path("runs") / task / str(model_name) / run_id
            output_dir.mkdir(parents=True, exist_ok=True)
            dest_path = output_dir / f"audit_{run_id}.json"

            # If we already moved it, just load it from our organized folder
            if dest_path.exists():
                with open(dest_path, 'r') as f:
                    table_json = json.load(f)
            else:
                # 2. Download to temp WandB location
                artifacts = run.logged_artifacts()
                audit_art = next(a for a in artifacts if "audit_confusion_cases_" in a.name)
                tmp_dir = Path(audit_art.download())
                source_file = list(tmp_dir.glob("*.table.json"))[0]
                shutil.move(str(source_file), str(dest_path))
                shutil.rmtree(tmp_dir)
                
                with open(dest_path, 'r') as f:
                    table_json = json.load(f)
                
            temp_df = pd.DataFrame(data=table_json["data"], columns=table_json["columns"])
            temp_df["run_id"] = run_id
            return temp_df
        except Exception:
            return None

    print("Starting parallel download of audit tables...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_run_audit, runs))

    all_audit_data = [df for df in results if df is not None]
    master_audit_df = pd.concat(all_audit_data, ignore_index=True)
    print(f"Finished! Processed {len(all_audit_data)} tables.")
    master_audit_df = pd.concat(all_audit_data, ignore_index=True)

    master_stats_df.to_csv(STATS_CACHE, index=False)
    master_audit_df.to_csv(AUDIT_CACHE, index=False)
    print(f"Data cached to {CACHE_DIR}")
    
    return master_stats_df, master_audit_df

master_stats_df, master_audit_df = get_master_data()

master_combined_df = master_audit_df.merge(
    master_stats_df, 
    on='run_id', 
    how='left', 
    suffixes=('', '_stats') # Handles overlapping columns like 'name', 'model', 'task'
)

# Create unified columns to handle the 'val/' vs 'val_' discrepancy
metrics_to_fix = ['acc', 'f1', 'precision', 'recall']

for m in ['acc', 'f1', 'recall', 'precision']:
    master_combined_df[f'val_{m}'] = master_combined_df.get(f'val/{m}', pd.Series(np.nan)) \
                                     .fillna(master_combined_df.get(f'val_{m}', pd.Series(np.nan))) \
                                     .fillna(0)
    
print(master_combined_df['val_f1'])
if 'val/mcc' in master_combined_df.columns:
    master_combined_df['val_mcc'] = master_combined_df['val/mcc'].fillna(0)
else:
    master_combined_df['val_mcc'] = 0.0

def extract_weight(w_obj, class_label):
    if pd.isna(w_obj) or w_obj is None:
        return 1.0
    
    if isinstance(w_obj, str):
        try:
            w_obj = json.loads(w_obj.replace("'", '"'))
        except:
            return 1.0
            
    if isinstance(w_obj, dict):
        return w_obj.get(str(class_label), w_obj.get(class_label, 1.0))
    
    return 1.0

master_stats_df['w0'] = master_stats_df['class_weight'].apply(lambda x: extract_weight(x, 0))
master_stats_df['w1'] = master_stats_df['class_weight'].apply(lambda x: extract_weight(x, 1))

pa = Path("notebooks") / "pipelines" / "p06_model_evaluation" / "diagnosis_results.txt"
pa.parent.mkdir(parents=True, exist_ok=True)

run_level_df = master_stats_df.drop_duplicates(subset=['run_id']).copy()
run_level_df['model_label'] = run_level_df['model'].fillna('unknown').str.lower()

# Standardize names
mapping = {
    'knn': 'KNN',
    'rf': 'RF',
    'xgb': 'XGB',
    'svm': 'SVM',
    'nb': 'NB',
    'lr': 'LR',
    'deep_nn': 'DEEP_NN',
    'deepnn': 'DEEP_NN',
    'deepneuralnetwork': 'DEEP_NN',   # add whatever the debug print shows
    'deep-nn': 'DEEP_NN',
}
run_level_df['model_label'] = run_level_df['model_label'].map(mapping).fillna(run_level_df['model_label'].str.upper())

print(run_level_df[run_level_df['model_label'] == 'DEEP_NN']['val_f1'])
with open(pa, "w") as out:
    out.write(f"{master_stats_df.head(10)}")
    out.write(f"{master_stats_df.columns}")

    summary_error = summarize_all_errors(master_combined_df, feature_names)
    out.write(f"{summary_error.head(30)}")

    # Aggregate by category and feature to find common 'drivers'
    general_drivers = summary_error.groupby(['task', 'category', 'feature']).agg({
        'delta': ['mean', 'std', 'count']
    }).reset_index()

    # Flatten columns
    general_drivers.columns = ['task', 'category', 'feature', 'avg_delta', 'delta_std', 'run_count']
    threshold = 264 * 0.2
    top_culprits = general_drivers[general_drivers['run_count'] > threshold].sort_values('avg_delta', ascending=False)
    out.write(f"{top_culprits.head(10)}")

    # 1. Global Top Models using the unified 'val_f1'
    # We sort by F1 first, then Recall to prioritize models that actually find neutralizers
    top_models = master_combined_df.groupby(['model', 'task']).agg({
        'val_acc': 'max',
        'val_f1': 'max',
        'val_recall': 'max',
        'oversample': 'first'
    }).sort_values(by=['val_f1', 'val_recall'], ascending=False).head(5)

    out.write("=== MSINTSY MODEL DIAGNOSIS REPORT ===\n\n")

    # --- 1. GLOBAL TOP 10 UNIQUE RUNS ---
    out.write("--- TOP UNIQUE RUNS (BY PERFORMANCE) ---\n")
    top_runs = run_level_df.sort_values(by=['val_f1', 'val_recall'], ascending=False).head(10)
    out.write(top_runs[['name', 'model', 'task', 'val_f1', 'val_recall', 'w0', 'w1', 'oversample']].to_string(index=False))
    out.write("\n\n")

    # --- 2. TASK-SPECIFIC ANALYSIS ---
    for task_name in run_level_df['task'].unique():
        out.write(f"{'='*15} TASK: {str(task_name).upper()} {'='*15}\n")
        task_df = run_level_df[run_level_df['task'] == task_name]
        
        for model_type in task_df['model_label'].unique():
            model_df = task_df[task_df['model_label'] == model_type]
            print(model_df)
            print(model_df.columns)
            # Use the parameters actually present in your data
            # 1. Define a broader set of potential hyperparams
            potential_params = ['oversample', 'has_pca', 'class_weight', 'has_synthetic', 'w0', 'w1']

            # 2. Filter for columns that actually exist in the DataFrame
            existing_cols = [p for p in potential_params if p in model_df.columns]

            # 3. CRITICAL: Drop columns that are entirely NaN for THIS specific model type
            # This prevents w0/w1 from breaking the DEEP_NN analysis
            valid_params = [p for p in existing_cols if model_df[p].notna().any()]

            try:
                if not valid_params:
                    # If no params exist, we just group by the model label itself 
                    # to at least get the mean performance of all runs
                    stats = model_df.groupby(['model_label'])['val_f1'].agg(['mean', 'count']).sort_values('mean', ascending=False)
                else:
                    # Group by the parameters that actually have data
                    stats = model_df.groupby(valid_params, dropna=False)['val_f1'].agg(['mean', 'count']).sort_values('mean', ascending=False)
                
                if stats.empty:
                    out.write(f"Model: {str(model_type).upper():12} | No valid runs found.\n")
                    continue

                best_config_vals = stats.index[0]
                
                # Clean up numpy types for the text file
                clean_params = {}
                for k, v in zip(valid_params, best_config_vals):
                    clean_params[k] = v.item() if hasattr(v, 'item') else v

                # Get the best individual run
                best_run = model_df.loc[model_df['val_f1'].idxmax()]
                
                out.write(f"Model: {str(model_type).upper():12}\n")
                out.write(f"  > Best Run: {best_run['name']}\n")
                out.write(f"  > Metrics:  F1: {best_run['val_f1']:.4f} | Recall: {best_run['val_recall']:.4f} | Prec: {best_run.get('val_precision', 0):.4f}\n")
                out.write(f"  > Setup:    {clean_params}\n")
                out.write(f"  > Consistency: Mean F1 of {stats.iloc[0]['mean']:.4f} over {int(stats.iloc[0]['count'])} runs\n")
            
            except Exception as e:
                out.write(f"Model: {str(model_type).upper():12} | Analysis failed: {type(e).__name__}\n")
        out.write("\n")
        
# --- PATH SETUP ---
ARTIFACTS_DIR = Path("artifacts")
PLOTS_DIR = ARTIFACTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def run_rf_shap_analysis(master_stats_df, task_name="neutralization"):
    """
    Finds the best RF run, resolves the Joblib filename vs Run ID folder, and plots SHAP.
    """
    df = master_stats_df.copy()

    # Ensure model_label exists for filtering
    if 'model_label' not in df.columns:
        mapping = {'rf': 'RF', 'deep_nn': 'DEEP_NN', 'xgb': 'XGB', 'knn': 'KNN', 'lr': 'LR'}
        df['model_label'] = df['model'].fillna('unknown').str.lower().map(mapping).fillna('OTHER')

    # 1. Find the best Random Forest run
    rf_stats = df[
        (df['task'] == task_name) & 
        (df['model_label'] == 'RF')
    ].sort_values(by='val_f1', ascending=False)

    if rf_stats.empty:
        print(f"No RF runs found. Models available: {df['model_label'].unique()}")
        return

    best_run = rf_stats.iloc[0]
    run_id = best_run['run_id']      # e.g., 'dk18knxm' (The folder)
    run_name = best_run['name']      # e.g., 'neutral-rf-014321-Pom-Pom' (The file)
    raw_model_type = str(best_run['model']).lower() # e.g., 'rf'

    # 2. Pathing Logic: runs/{task}/{model_type}/{run_id}/{run_name}.pkl
    model_dir = Path("artifacts") / "models" 
    
    # Try exact match first, then fall back to globbing if naming differs slightly
    model_path = model_dir / f"{run_name}.joblib"
    
    if not model_path.exists():
        print(f"Direct path {model_path} not found. Searching directory...")
        try:
            model_path = list(model_dir.glob("*.pkl"))[0]
        except IndexError:
            print(f"Critical: No .pkl found in {model_dir}")
            return

    print(f"Loading Model: {model_path.name}")

    # 3. Get Feature Names & Extraction
    factory = FeatureFactory()
    types = [FeatureType.NAIVE, FeatureType.MOTIF_CONJOINT, FeatureType.BIOCHEMICAL]
    feature_names = factory.get_all_feature_names(types)

    pipeline = joblib.load(model_path)
    
    if isinstance(pipeline, Pipeline):
        # We assume the model is the last step in your pipeline
        model = pipeline.steps[-1][1]
        print(f"Extracted {type(model).__name__} from Pipeline.")
    else:
        model = pipeline
    
    feature_lookup = extraction.search(
        str(best_run['task']), 
        int(best_run['oversample']), 
        bool(best_run['has_synthetic'])
    )
    
    # Sample 300 for SHAP stability
    sample_names = list(feature_lookup.keys())[:300]
    X_samples = np.array([feature_lookup[name] for name in sample_names])
    
    # Safety check on feature dimensions
    if X_samples.shape[1] != len(feature_names):
        print(f"Dim Mismatch: Data({X_samples.shape[1]}) != Names({len(feature_names)})")
        feature_names = feature_names[:X_samples.shape[1]]

    X_df = pd.DataFrame(X_samples, columns=feature_names)

    # 4. SHAP Calculation
    print(f"Generating SHAP for {best_run['name']}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    # --- CRITICAL FIX: Dimension Handling ---
    # Scikit-Learn RF returns a list of 2 arrays: [neg_class_shap, pos_class_shap]
    # We want the second one (index 1) for 'Neutralization'
    if isinstance(shap_values, list):
        # Verify if it's binary classification
        if len(shap_values) == 2:
            target_shap = shap_values[1]
        else:
            target_shap = shap_values[0]
    else:
        target_shap = shap_values

    # 1. Calculate the Global Importance (Mean Absolute SHAP)
    # This mimics Gini importance: higher = more influential
    # 1. Force target_shap to be a 2D array (Samples x Features)
    # If it's 3D (Classes x Samples x Features), we take index 1 (Positive Class)
    if len(target_shap.shape) == 3:
        target_shap = target_shap[1] 

    # 2. Calculate Global Importance (Mean Absolute SHAP)
    # This result MUST be 1D (Length = Number of Features)
    global_importances = np.abs(target_shap).mean(axis=0)

    # 3. Handle Feature Name Mismatch (if extraction returned different counts)
    num_features_in_data = global_importances.shape[0]
    if len(feature_names) != num_features_in_data:
        print(f"Warning: Re-aligning {len(feature_names)} names to {num_features_in_data} features.")
        # Slice or pad feature_names to match the actual data shape
        current_names = feature_names[:num_features_in_data]
    else:
        current_names = feature_names

    # 4. Create DataFrame (Ensuring 1D input)
    importance_df = pd.DataFrame({
        'Feature': [str(f) for f in current_names],
        'Importance': global_importances.flatten() # .flatten() ensures 1D
    }).sort_values(by='Importance', ascending=False).head(20)

    # 5. Professional Seaborn Plot
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="white") # Cleaner look for defense
    
    # Horizontal Bar Plot
    sns.barplot(
        data=importance_df, 
        x='Importance', 
        y='Feature', 
        palette="mako"
    )

    plt.title(f"SHAP Feature Importance: {run_name}", fontsize=14, fontweight='bold')
    plt.xlabel("mean(|SHAP value|) - Average Impact on Prediction", fontsize=12)
    plt.ylabel("") 

    # Add a border to the plot to prevent the "floating" look
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    save_path = PLOTS_DIR / f"rf_shap_final_{run_id}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Success! Clean plot saved: {save_path}")

# Run analysis
run_rf_shap_analysis(master_stats_df)
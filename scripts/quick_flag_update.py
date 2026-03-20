import wandb
import re

api = wandb.Api()
entity, project = "logarithmicpresence-de-la-salle-university", "CSC713M_MSINTSY" # Update these
runs = api.runs(f"{entity}/{project}")

print(f"Checking {len(runs)} runs...")
for run in runs:
    # Pattern to find '-F' followed by the bitmask integer
    match = re.search(r'-F(\d+)-', run.name)
    
    if match:
        bitmask = int(match.group(1))
        
        # Mapping based on ModelFlags class:
        # bit 0 (1): PCA
        # bit 1 (2): Synthetic
        # bit 2 (4): Oversampling
        # bit 3 (8): Balanced
        
        updates = {
            "has_pca": bool(bitmask & 1),
            "has_synthetic_cdr": bool(bitmask & 2),
            "oversampling_ratio": 2 if bool(bitmask & 4) else 1,
            "is_balanced": bool(bitmask & 8)
        }
        
        # Update and Push
        run.config.update(updates)
        run.update()
        
        print(f"Updated {run.name} (F{bitmask}) -> {updates}")

print("Metadata repair complete.")
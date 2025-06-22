import numpy as np
import os
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

# --- Configuration ---
SAVE_DIR = "./McNemar"
os.makedirs(SAVE_DIR, exist_ok=True)
alpha = 0.05
scenarios = ['overall', 'flat', 'indoor', 'outdoor']
split_type = "Disjoint"  # Change to "Non-Disjoint" if needed

# --- McNemar test function ---
def run_mcnemar(y_true, pred1, pred2, alpha=0.05, label="Model A vs Model B", scenario="overall", split="Disjoint"):
    both_correct = (pred1 == y_true) & (pred2 == y_true)
    a_only_correct = (pred1 == y_true) & (pred2 != y_true)
    b_only_correct = (pred1 != y_true) & (pred2 == y_true)
    both_wrong = (pred1 != y_true) & (pred2 != y_true)

    table = [[both_correct.sum(), b_only_correct.sum()],
             [a_only_correct.sum(), both_wrong.sum()]]

    result = mcnemar(table, exact=False, correction=True)
    stat = result.statistic
    pval = result.pvalue
    sig = "Yes" if pval < alpha else "No"

    print(f"\n--- McNemar's Test: {label} ({scenario}, {split} Split) ---")
    print(f"Contingency Table: {table}")
    print(f"Chi2 = {stat:.4f}, p = {pval:.6f}")
    print(f"Significant at α={alpha}? {sig}")

    return {
        "Split Type": split,
        "Scenario": scenario.capitalize(),
        "Comparison": label,
        "Chi2": round(stat, 4),
        "p-value": round(pval, 6),
        f"Significant (α < {alpha})": sig
    }

# --- Handle missing overall scenario by concatenating flat, indoor, and outdoor ---
def generate_overall_if_missing():
    overall_required = [
        "y.npy", "cbam_preds.npy", "resnet_preds.npy", "grad_resnet_preds.npy"
    ]
    missing = [f for f in overall_required if not os.path.exists(os.path.join(SAVE_DIR, f))]
    if not missing:
        return  # All files exist

    print("🔧 'overall' scenario files missing — generating by concatenating flat, indoor, and outdoor data.")
    try:
        y_all = np.concatenate([
            np.load(os.path.join(SAVE_DIR, "y_flat.npy")),
            np.load(os.path.join(SAVE_DIR, "y_indoor.npy")),
            np.load(os.path.join(SAVE_DIR, "y_outdoor.npy"))
        ])
        cbam_all = np.concatenate([
            np.load(os.path.join(SAVE_DIR, "cbam_preds_flat.npy")),
            np.load(os.path.join(SAVE_DIR, "cbam_preds_indoor.npy")),
            np.load(os.path.join(SAVE_DIR, "cbam_preds_outdoor.npy"))
        ])
        resnet_all = np.concatenate([
            np.load(os.path.join(SAVE_DIR, "resnet_preds_flat.npy")),
            np.load(os.path.join(SAVE_DIR, "resnet_preds_indoor.npy")),
            np.load(os.path.join(SAVE_DIR, "resnet_preds_outdoor.npy"))
        ])
        grad_all = np.concatenate([
            np.load(os.path.join(SAVE_DIR, "grad_resnet_preds_flat.npy")),
            np.load(os.path.join(SAVE_DIR, "grad_resnet_preds_indoor.npy")),
            np.load(os.path.join(SAVE_DIR, "grad_resnet_preds_outdoor.npy"))
        ])

        np.save(os.path.join(SAVE_DIR, "y.npy"), y_all)
        np.save(os.path.join(SAVE_DIR, "cbam_preds.npy"), cbam_all)
        np.save(os.path.join(SAVE_DIR, "resnet_preds.npy"), resnet_all)
        np.save(os.path.join(SAVE_DIR, "grad_resnet_preds.npy"), grad_all)
        print("✅ Overall scenario files generated and saved.")
    except Exception as e:
        print(f"❌ Failed to generate 'overall' scenario files: {e}")

# --- Run everything ---
generate_overall_if_missing()

results = []
for scene in scenarios:
    suffix = "" if scene == "overall" else f"_{scene}"
    try:
        y_true = np.load(os.path.join(SAVE_DIR, f"y{suffix}.npy"))
        cbam_preds = np.load(os.path.join(SAVE_DIR, f"cbam_preds{suffix}.npy"))
        resnet_preds = np.load(os.path.join(SAVE_DIR, f"resnet_preds{suffix}.npy"))
        gradcam_preds = np.load(os.path.join(SAVE_DIR, f"grad_resnet_preds{suffix}.npy"))
    except FileNotFoundError as e:
        print(f"⚠️ Missing files for scenario '{scene}': {e}")
        continue

    results.append(run_mcnemar(y_true, cbam_preds, resnet_preds,
                               alpha=alpha, label="CBAM-ResNet vs ResNet", scenario=scene, split=split_type))
    results.append(run_mcnemar(y_true, cbam_preds, gradcam_preds,
                               alpha=alpha, label="CBAM-ResNet vs Grad-CAM ResNet", scenario=scene, split=split_type))

# --- Save to CSV ---
df = pd.DataFrame(results)
csv_path = os.path.join(SAVE_DIR, "mcnemar_results_all_scenarios.csv")
df.to_csv(csv_path, index=False)
print(f"\n✅ Saved McNemar test results to {csv_path}")

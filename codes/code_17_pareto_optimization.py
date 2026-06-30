import pandas as pd
import numpy as np
import joblib
import random
import os
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# --- CONFIGURATION ---
MODEL_PATH = "../files/ansys_ensemble_pipeline.pkl"
TARGET_CAPACITANCE = 200.0  # pF/m
NUM_CANDIDATES = 25000

# Base Material Data
BACKBONES = {
    "PE": {"backbone": "CCCCC", "eps": 2.2, "tg": -120, "density": 0.92},
    "PP": {"backbone": "CC(C)CC(C)", "eps": 2.2, "tg": -10, "density": 0.90},
    "PVC": {"backbone": "CC(Cl)CC(Cl)", "eps": 3.5, "tg": 80, "density": 1.40},
    "PTFE": {"backbone": "C(F)(F)C(F)(F)", "eps": 2.1, "tg": 115, "density": 2.20},
    "PVDF": {"backbone": "CC(F)(F)CC(F)(F)", "eps": 12.0, "tg": -35, "density": 1.78},
    "Polystyrene": {"backbone": "CC(c1ccccc1)CC(c1ccccc1)", "eps": 2.5, "tg": 100, "density": 1.05},
    "PMMA": {"backbone": "CC(C)(C(=O)OC)CC(C)(C(=O)OC)", "eps": 3.0, "tg": 105, "density": 1.18},
    "Polycarbonate": {"backbone": "Oc1ccc(cc1)C(C)(C)c2ccc(cc2)O", "eps": 2.9, "tg": 145, "density": 1.20},
    "PET": {"backbone": "C(=O)c1ccc(cc1)C(=O)OCCO", "eps": 3.3, "tg": 75, "density": 1.38},
    "Nylon-6": {"backbone": "CCCCCC(=O)N", "eps": 3.5, "tg": 47, "density": 1.14}
}
LEFT_GROUPS = ["C", "CC", "CCC", "FC(F)(F)", "Cl", "CO", "CC(=O)", "c1ccccc1", "N"]
RIGHT_GROUPS = ["C", "CC", "CCC", "C(F)(F)F", "Cl", "OC", "C(=O)O", "C#N"]

def smiles_to_morgan(smi, n_bits=512):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(n_bits)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp)

# --- PARETO FRONT ALGORITHM ---
def identify_pareto(scores):
    """
    Find the pareto-efficient points
    scores is a numpy array of shape (N, 2). 
    We want to MINIMIZE column 0 (Capacitance Error) and MAXIMIZE column 1 (Tg).
    Since standard pareto algorithms minimize both, we will invert column 1 for the math.
    """
    # Create a copy where objective 2 is inverted
    scores_inv = scores.copy()
    scores_inv[:, 1] = -scores_inv[:, 1] 
    
    population_size = scores_inv.shape[0]
    pareto_front = np.ones(population_size, dtype=bool)
    for i in range(population_size):
        for j in range(population_size):
            if all(scores_inv[j] <= scores_inv[i]) and any(scores_inv[j] < scores_inv[i]):
                pareto_front[i] = 0
                break
    return pareto_front

def main():
    print(f"============================================================")
    print(f"  Phase 5.1 — Multi-Objective Pareto Optimization")
    print(f"  Objective 1: Minimize Error from {TARGET_CAPACITANCE} pF/m")
    print(f"  Objective 2: Maximize Thermal Stability (Tg Celsius)")
    print(f"============================================================")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model {MODEL_PATH} not found.")
        return

    print(f"\n[1/4] Loading trained Ensemble Model...")
    model = joblib.load(MODEL_PATH)

    print(f"[2/4] Generating Virtual Library ({NUM_CANDIDATES:,} random configurations)...")
    candidates = []
    fps = []
    for _ in range(NUM_CANDIDATES):
        name = random.choice(list(BACKBONES.keys()))
        meta = BACKBONES[name]
        smi = f"{random.choice(LEFT_GROUPS)}{meta['backbone']}{random.choice(RIGHT_GROUPS)}"
        
        tg = meta["tg"] + random.uniform(-5, 5)
        candidates.append({
            "smiles": smi,
            "base_material": name,
            "dielectric_constant": meta["eps"] + random.uniform(-0.1, 0.1),
            "tg_celsius": tg,
            "density_g_cm3": meta["density"] + random.uniform(-0.05, 0.05),
            "thickness_nm": random.uniform(500, 5000),
            "applied_voltage": random.uniform(10, 500),
            "temp_c": 25
        })
        fps.append(smiles_to_morgan(smi))

    df_cand = pd.DataFrame(candidates)
    
    print("[3/4] Running ultra-fast Machine Learning predictions...")
    fp_cols = [f"Morgan_{i}" for i in range(512)]
    df_fps = pd.DataFrame(np.vstack(fps), columns=fp_cols)
    physical_cols = ['dielectric_constant', 'tg_celsius', 'density_g_cm3', 'thickness_nm', 'applied_voltage', 'temp_c']
    
    X = pd.concat([df_fps, df_cand[physical_cols]], axis=1)
    preds = model.predict(X)
    df_cand['predicted_capacitance'] = preds
    df_cand['cap_error'] = abs(df_cand['predicted_capacitance'] - TARGET_CAPACITANCE)
    
    print("[4/4] Calculating Pareto-Optimal Front...")
    # Extract the two objectives as a numpy array
    scores = df_cand[['cap_error', 'tg_celsius']].values
    
    # Run Pareto Algorithm
    pareto_mask = identify_pareto(scores)
    pareto_df = df_cand[pareto_mask].sort_values(by='tg_celsius', ascending=False)
    
    # Save Pareto results
    pareto_df.to_csv("../results/pareto_optimal_polymers.csv", index=False)
    print(f"\n-> Identified {len(pareto_df)} Pareto-Optimal polymers out of {NUM_CANDIDATES}.")
    print(f"-> Saved to pareto_optimal_polymers.csv")
    
    # --- PLOTTING ---
    plt.figure(figsize=(10, 7))
    # Plot all candidates in grey
    plt.scatter(df_cand['tg_celsius'], df_cand['cap_error'], c='lightgrey', alpha=0.5, label='Sub-optimal Candidates', s=10)
    # Plot Pareto Front in red
    plt.scatter(pareto_df['tg_celsius'], pareto_df['cap_error'], c='red', label='Pareto Front', s=30, edgecolor='black')
    
    plt.title(f'Multi-Objective Pareto Optimization\nTarget: {TARGET_CAPACITANCE} pF/m vs Thermal Stability (Tg)')
    plt.xlabel('Thermal Stability (Tg Celsius) $\\rightarrow$ Maximize')
    plt.ylabel(f'Absolute Capacitance Error from {TARGET_CAPACITANCE} pF/m $\\rightarrow$ Minimize')
    plt.axhline(0, color='black', linestyle='--')
    plt.ylim(-10, 150) # Zoom in on the relevant error region
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../graphs/pareto_front.png", dpi=300)
    print(f"-> Generated plot: pareto_front.png")
    
if __name__ == "__main__":
    main()

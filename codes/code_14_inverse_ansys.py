import pandas as pd
import numpy as np
import joblib
import random
import os

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# --- CONFIGURATION ---
MODEL_PATH = "../files/ansys_ensemble_pipeline.pkl"
TARGET_CAPACITANCE = 200.0  # pF/m (The industrial EV target we discussed)
NUM_CANDIDATES = 25000      # Size of the Virtual High-Throughput Screening library

# --- MATERIAL BACKBONES (Must match Phase 1) ---
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
    if mol is None:
        return np.zeros(n_bits)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp)

def run_inverse_design():
    print(f"============================================================")
    print(f"  Phase C — Inverse Design (Target: {TARGET_CAPACITANCE} pF/m)")
    print(f"============================================================")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model {MODEL_PATH} not found. Run code_13 first.")
        return

    print("\n[1/4] Loading trained 3-Way Ensemble Model...")
    model = joblib.load(MODEL_PATH)

    print(f"[2/4] Generating Virtual Library ({NUM_CANDIDATES:,} random configurations)...")
    candidates = []
    fps = []
    
    for _ in range(NUM_CANDIDATES):
        name = random.choice(list(BACKBONES.keys()))
        meta = BACKBONES[name]
        lg = random.choice(LEFT_GROUPS)
        rg = random.choice(RIGHT_GROUPS)
        
        smi = f"{lg}{meta['backbone']}{rg}"
        
        # Add realistic physical variations
        eps = meta["eps"] + random.uniform(-0.1, 0.1)
        tg = meta["tg"] + random.uniform(-5, 5)
        dens = meta["density"] + random.uniform(-0.05, 0.05)
        
        # Inverse Design searches for the optimal manufacturing parameters
        thick = random.uniform(500, 5000)
        volt = random.uniform(10, 500)
        
        candidates.append({
            "smiles": smi,
            "base_material": name,
            "dielectric_constant": eps,
            "tg_celsius": tg,
            "density_g_cm3": dens,
            "thickness_nm": thick,
            "applied_voltage": volt,
            "temp_c": 25  # standard
        })
        fps.append(smiles_to_morgan(smi))

    df_cand = pd.DataFrame(candidates)
    
    # Reconstruct the exact 518-feature matrix used in Phase B training
    print("[3/4] Running ultra-fast Machine Learning predictions...")
    fp_cols = [f"Morgan_{i}" for i in range(512)]
    df_fps = pd.DataFrame(np.vstack(fps), columns=fp_cols)
    physical_cols = ['dielectric_constant', 'tg_celsius', 'density_g_cm3', 'thickness_nm', 'applied_voltage', 'temp_c']
    
    X = pd.concat([df_fps, df_cand[physical_cols]], axis=1)
    
    # Predict all 25,000 samples instantly
    preds = model.predict(X)
    df_cand['predicted_capacitance'] = preds
    
    print(f"[4/4] Scoring candidates against target ({TARGET_CAPACITANCE} pF/m)...")
    # Objective function: Minimize Absolute Error
    df_cand['error'] = abs(df_cand['predicted_capacitance'] - TARGET_CAPACITANCE)
    
    # Filter out anything physically impossible (thickness < 500) just to be safe
    df_valid = df_cand[df_cand['thickness_nm'] >= 500]
    ranked = df_valid.sort_values(by='error').head(10).reset_index(drop=True)
    
    print(f"\n--- TOP 10 RECOMMENDATIONS FOR {TARGET_CAPACITANCE} pF/m ---")
    print("-" * 115)
    print(f"{'Rank':<5} | {'Base Material':<15} | {'Predicted Cap':<15} | {'Thickness':<12} | {'Voltage':<10} | {'SMILES (Truncated)'}")
    print("-" * 115)
    
    for i, row in ranked.iterrows():
        cap = f"{row['predicted_capacitance']:.2f} pF/m"
        thick = f"{row['thickness_nm']:.1f} nm"
        volt = f"{row['applied_voltage']:.1f} V"
        smiles_trunc = row['smiles'][:20] + "..." if len(row['smiles']) > 20 else row['smiles']
        print(f"#{i+1:<4} | {row['base_material']:<15} | {cap:<15} | {thick:<12} | {volt:<10} | {smiles_trunc}")
    print("-" * 115)
    
    ranked.to_csv("../results/inverse_design_recommendations.csv", index=False)
    print("\n-> Full recommendation specs saved to: inverse_design_recommendations.csv")

if __name__ == "__main__":
    run_inverse_design()

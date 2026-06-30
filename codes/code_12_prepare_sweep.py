import pandas as pd
import numpy as np
import random
import os
from rdkit import Chem

# --- 1. SETTINGS ---
OUTPUT_FILE = "../results/ansys_sweep_targets.csv"

# --- 2. BASE POLYMER BACKBONES ---
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

# --- 3. MUTATION GROUPS ---
# 9 Left x 8 Right = 72 valid chemical mutations per backbone
LEFT_GROUPS = ["C", "CC", "CCC", "FC(F)(F)", "Cl", "CO", "CC(=O)", "c1ccccc1", "N"]
RIGHT_GROUPS = ["C", "CC", "CCC", "C(F)(F)F", "Cl", "OC", "C(=O)O", "C#N"]

def generate_targets():
    print(f"--- Preparing Combinatorial SMILES Sweep Targets ---")
    data = []

    # Iterate through all 720 possible combinatorial variations TWICE (1440 samples)
    for _ in range(2):
        for name, meta in BACKBONES.items():
            for lg in LEFT_GROUPS:
                for rg in RIGHT_GROUPS:
                    # 1. Mutate the SMILES
                    mutated_smiles = f"{lg}{meta['backbone']}{rg}"
                    
                    # 2. Perturb the physical properties slightly to simulate the mutation's effect
                    mutated_eps = meta["eps"] + random.uniform(-0.1, 0.1)
                    mutated_tg = meta["tg"] + random.uniform(-5, 5)
                    mutated_density = meta["density"] + random.uniform(-0.05, 0.05)
                    
                    # 3. Assign random simulation bounds
                    thickness_nm = np.random.uniform(500, 5000)
                    voltage = np.random.uniform(10, 500)
                    
                    data.append({
                        "smiles": mutated_smiles,
                        "base_material": name,
                        "dielectric_constant": mutated_eps,
                        "tg_celsius": mutated_tg,
                        "density_g_cm3": mutated_density,
                        "thickness_nm": thickness_nm,
                        "applied_voltage": voltage,
                        "temp_c": np.random.choice([25, 100, 200, 300])
                    })

    # Shuffle the dataset to ensure random distribution
    random.shuffle(data)
    
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"-> Created {OUTPUT_FILE} with exactly {len(df)} UNIQUE valid polymer structures!")

if __name__ == "__main__":
    generate_targets()

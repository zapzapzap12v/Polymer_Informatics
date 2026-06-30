"""
Code 18: Phase 5.2 -- Graph Neural Network (GNN) Feature Extractor + Retraining.

Instead of binary Morgan fingerprints, this script uses a manual 2-hop Message
Passing Neural Network (MPNN) philosophy to extract rich, continuous atom-level
features from each polymer's molecular graph using RDKit. These richer features
are then used to retrain the 3-Way Ensemble for improved R2.
"""
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INPUT_CSV  = "../results/ansys_simulation_results.csv"
OUTPUT_MODEL = "../files/ansys_gnn_pipeline.pkl"
RANDOM_SEED  = 42

# --- ATOM FEATURE EXTRACTOR (Manual Message Passing) ---
ATOM_FEATURES = ['AtomicNum', 'Degree', 'NumHs', 'Charge', 'IsArom', 'IsInRing',
                 'Hybridization_SP', 'Hybridization_SP2', 'Hybridization_SP3']
N_ATOM_FEATS = len(ATOM_FEATURES)

def atom_features(atom):
    hyb = atom.GetHybridization()
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetTotalNumHs(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        int(hyb == Chem.rdchem.HybridizationType.SP),
        int(hyb == Chem.rdchem.HybridizationType.SP2),
        int(hyb == Chem.rdchem.HybridizationType.SP3),
    ]

def mpnn_fingerprint(smi, max_atoms=30):
    """
    2-hop Message Passing: for each atom, aggregate its own features
    plus the mean features of its 1-hop and 2-hop neighbors.
    Output: a fixed-length vector representing the whole molecule.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(N_ATOM_FEATS * 3)  # own + 1hop + 2hop aggregates

    n_atoms = mol.GetNumAtoms()
    atom_feat_matrix = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=float)

    # Build adjacency list
    adj = {i: [] for i in range(n_atoms)}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)

    # 1-hop aggregation (mean of direct neighbors)
    hop1 = np.zeros_like(atom_feat_matrix)
    for i in range(n_atoms):
        nbrs = adj[i]
        if nbrs:
            hop1[i] = atom_feat_matrix[nbrs].mean(axis=0)

    # 2-hop aggregation (mean of neighbors-of-neighbors)
    hop2 = np.zeros_like(atom_feat_matrix)
    for i in range(n_atoms):
        nbrs2 = set()
        for j in adj[i]:
            nbrs2.update(adj[j])
        nbrs2.discard(i)
        if nbrs2:
            hop2[i] = atom_feat_matrix[list(nbrs2)].mean(axis=0)

    # Global-pool (mean over all atoms) to get molecule-level descriptor
    own_global  = atom_feat_matrix.mean(axis=0)
    hop1_global = hop1.mean(axis=0)
    hop2_global = hop2.mean(axis=0)

    return np.concatenate([own_global, hop1_global, hop2_global])

def rdkit_2d_descriptors(smi):
    """Additional 2D molecular descriptors from RDKit for extra signal."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(6)
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
    ])

def build_gnn_features(df):
    mpnn_feats = np.vstack([mpnn_fingerprint(s) for s in df['smiles']])
    rdkit_feats = np.vstack([rdkit_2d_descriptors(s) for s in df['smiles']])
    phys = df[['dielectric_constant','tg_celsius','density_g_cm3',
               'thickness_nm','applied_voltage','temp_c']].values
    return np.hstack([mpnn_feats, rdkit_feats, phys])


def main():
    print("=" * 60)
    print("  Phase 5.2 -- GNN Message Passing Feature Retraining")
    print("=" * 60)

    print("\n[1/5] Loading ANSYS ground truth data...")
    df = pd.read_csv(INPUT_CSV)
    df_ok = df[df['sim_status'] == 'Success'].dropna(subset=['sim_capacitance_F']).copy()
    print(f"     -> {len(df_ok)} verified ANSYS samples loaded.")

    print("[2/5] Building MPNN molecular graph features (2-hop message passing)...")
    X = build_gnn_features(df_ok)
    y = df_ok['sim_capacitance_F'].values
    print(f"     -> Feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED)

    print("[3/5] Training 3-Way Ensemble on GNN features...")
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=2000,
                       early_stopping=True, n_iter_no_change=200,
                       random_state=RANDOM_SEED, alpha=0.01, learning_rate='adaptive')
    gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                    max_depth=4, subsample=0.8,
                                    random_state=RANDOM_SEED)
    xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                       colsample_bytree=0.8, subsample=0.8,
                       random_state=RANDOM_SEED, verbosity=0)

    ensemble = VotingRegressor([('mlp', mlp), ('gbr', gbr), ('xgb', xgb)],
                               weights=[0.35, 0.35, 0.30])

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', ensemble)
    ])
    pipe.fit(X_train, y_train)

    print("[4/5] Evaluating GNN Ensemble...")
    r2_test = pipe.score(X_test, y_test)
    y_pred  = pipe.predict(X_test)
    mae = np.mean(np.abs(y_pred - y_test))
    rmse = np.sqrt(np.mean((y_pred - y_test)**2))

    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
    print(f"\n  Test R2 : {r2_test:.4f}")
    print(f"  MAE     : {mae:.2f} pF/m")
    print(f"  RMSE    : {rmse:.2f} pF/m")
    print(f"  5-Fold CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Parity Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue', edgecolors='k', lw=0.3)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Capacitance (pF/m)')
    plt.ylabel('GNN Ensemble Prediction (pF/m)')
    plt.title(f'GNN (MPNN) Surrogate Model Parity Plot\nR2 = {r2_test:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../graphs/gnn_parity_plot.png', dpi=300)
    print("  -> Saved: gnn_parity_plot.png")

    print(f"\n[5/5] Saving GNN ensemble model -> {OUTPUT_MODEL}")
    joblib.dump(pipe, OUTPUT_MODEL)
    print("\n" + "=" * 60)
    print("  Phase 5.2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import os
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from io import BytesIO
import base64

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="Polymer Inverse Design",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit default styling
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white;}
    </style>
    """, unsafe_allow_html=True)

# --- BACKEND DATA ---
MODEL_PATH = "../files/ansys_ensemble_pipeline.pkl"

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

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def smiles_to_morgan(smi, n_bits=512):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.zeros(n_bits)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp)

def render_molecule(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        img = Draw.MolToImage(mol, size=(250, 100), kekulize=True)
        return img
    return None

# --- SIDEBAR ---
st.sidebar.title("⚡ Settings")
st.sidebar.markdown("Configure the Inverse Design constraints:")

target_cap = st.sidebar.number_input("Target Capacitance (pF/m)", min_value=10.0, max_value=2000.0, value=200.0, step=10.0)
library_size = st.sidebar.slider("Virtual Library Size", min_value=1000, max_value=25000, value=10000, step=1000, help="Larger libraries take longer but find better matches.")
thickness_range = st.sidebar.slider("Thickness Constraint (nm)", min_value=100, max_value=5000, value=(500, 3000), step=100)

st.sidebar.markdown("---")
st.sidebar.markdown("**Powered by:** ANSYS Maxwell 2D & XGBoost")

# --- MAIN PAGE ---
st.title("🔬 Polymer Informatics Inverse Design")
st.markdown("Enter a target capacitance and the Physics-Informed ML Ensemble will recommend the exact polymer structure and manufacturing parameters to achieve it.")

model = load_model()

if model is None:
    st.error(f"Model `{MODEL_PATH}` not found. Please wait for your training script to finish.")
else:
    if st.button("🚀 Run AI Inverse Design Discovery"):
        with st.spinner(f"Generating and screening {library_size:,} candidates..."):
            
            # 1. Generate Virtual Library
            candidates = []
            fps = []
            
            for _ in range(library_size):
                name = random.choice(list(BACKBONES.keys()))
                meta = BACKBONES[name]
                smi = f"{random.choice(LEFT_GROUPS)}{meta['backbone']}{random.choice(RIGHT_GROUPS)}"
                
                thick = random.uniform(thickness_range[0], thickness_range[1])
                volt = random.uniform(10, 500)
                
                candidates.append({
                    "smiles": smi,
                    "base_material": name,
                    "dielectric_constant": meta["eps"] + random.uniform(-0.1, 0.1),
                    "tg_celsius": meta["tg"] + random.uniform(-5, 5),
                    "density_g_cm3": meta["density"] + random.uniform(-0.05, 0.05),
                    "thickness_nm": thick,
                    "applied_voltage": volt,
                    "temp_c": 25
                })
                fps.append(smiles_to_morgan(smi))

            # 2. Predict using ultra-fast ML
            df_cand = pd.DataFrame(candidates)
            fp_cols = [f"Morgan_{i}" for i in range(512)]
            df_fps = pd.DataFrame(np.vstack(fps), columns=fp_cols)
            physical_cols = ['dielectric_constant', 'tg_celsius', 'density_g_cm3', 'thickness_nm', 'applied_voltage', 'temp_c']
            
            X = pd.concat([df_fps, df_cand[physical_cols]], axis=1)
            preds = model.predict(X)
            
            # 3. Score against target
            df_cand['predicted_capacitance'] = preds
            df_cand['error'] = abs(df_cand['predicted_capacitance'] - target_cap)
            
            # Get Top 6
            ranked = df_cand.sort_values(by='error').head(6).reset_index(drop=True)
            
            st.success(f"Screening Complete! Maximum deviation from target: {ranked['error'].iloc[0]:.3f} pF/m")
            
            # 4. Display Results in a beautiful grid
            cols = st.columns(3)
            
            for i, row in ranked.iterrows():
                col = cols[i % 3]
                with col:
                    st.markdown(f"### Rank #{i+1}")
                    st.markdown(f"**Predicted:** `{row['predicted_capacitance']:.2f} pF/m`")
                    st.markdown(f"**Base:** {row['base_material']}")
                    st.markdown(f"**Thickness:** {row['thickness_nm']:.1f} nm")
                    
                    img = render_molecule(row['smiles'])
                    if img:
                        st.image(img, use_container_width=True)
                    st.caption(f"SMILES: `{row['smiles']}`")
                    st.markdown("---")

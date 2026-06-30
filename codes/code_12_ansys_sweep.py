import os
import sys
import pandas as pd
import time
from ansys.aedt.core import Desktop, Maxwell2d
from ansys.aedt.core.modules.boundary.maxwell_boundary import MatrixElectric
import config

# --- 1. CONFIGURATION ---
INPUT_FILE  = "../results/ansys_sweep_targets.csv"
OUTPUT_FILE = "../results/ansys_simulation_results.csv"
ANSYS_VERSION = "2025.2"
SAMPLES_PER_SESSION = config.ANSYS_SESSION_RESTART_INTERVAL
MAX_MESH_RETRIES = 3          # 5.5: Max retries per sample before logging as failed
THICKNESS_BACKOFF_FACTOR = 0.1  # 5.5: Back off by 10% of thickness on each retry


def run_single_sample(m2d, row, thickness_override_nm=None):
    """
    5.5 Adaptive Error Handling: Attempt to run one sample.
    Returns (cap_value, status_string).
    Raises RuntimeError on irrecoverable mesh failure.
    """
    eps        = row['dielectric_constant']
    thickness_nm = thickness_override_nm if thickness_override_nm else row['thickness_nm']
    thickness_um = thickness_nm / 1000.0
    voltage    = row['applied_voltage']

    m2d["FilmThickness"] = f"{thickness_um}um"
    m2d["AppVoltage"]    = f"{voltage}V"
    m2d.materials["vacuum"].permittivity = eps
    m2d.analyze_setup("MySetup", cores=config.ANSYS_CORE_COUNT, gpus=0, use_auto_settings=False)

    sol = m2d.post.get_solution_data("Matrix1.C(TopPlate, TopPlate)")
    cap = sol.data_real()[0] if sol else 0.0
    return cap


def run_simulation_sweep():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        sys.exit()

    df_targets = pd.read_csv(INPUT_FILE)
    print(f"--- Starting ANSYS Sweep ({len(df_targets)} samples) ---")

    results = []
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        start_index = len(df_existing)
        results     = df_existing.to_dict("records")
        print(f"Resuming from sample {start_index}...")
    else:
        start_index = 0

    total_samples  = len(df_targets)
    current_index  = start_index

    while current_index < total_samples:
        try:
            print(f"\nSession Start: Index {current_index}")
            d   = Desktop(version=ANSYS_VERSION, non_graphical=True, student_version=True)
            m2d = Maxwell2d(project="CapacitorSweep", design="ThinFilm")
            m2d.solution_type = "Electrostatic"

            # Setup Template
            m2d["FilmThickness"] = "1um"
            m2d["AppVoltage"]    = "100V"
            m2d.modeler.create_region([20, 20, 20, 20])
            dielectric = m2d.modeler.create_rectangle(
                [0, 0, 0], ["10um", "FilmThickness", 0],
                name="Dielectric", matname="vacuum"
            )
            m2d.assign_voltage(dielectric.edges[2], amplitude="AppVoltage", name="TopPlate")
            m2d.assign_voltage(dielectric.edges[0], amplitude="0V",         name="BottomPlate")
            margs = MatrixElectric(signal_sources=["TopPlate"],
                                   ground_sources=["BottomPlate"],
                                   matrix_name="Matrix1")
            m2d.assign_matrix(margs)
            m2d.create_setup(name="MySetup")

            batch_end = min(current_index + SAMPLES_PER_SESSION, total_samples)
            for i in range(current_index, batch_end):
                row    = df_targets.iloc[i]
                smiles = row["smiles"]

                # ── 5.5 Adaptive Mesh Retry Loop ──────────────────────────
                cap    = 0.0
                status = "Failed"
                thickness_try = row["thickness_nm"]

                for attempt in range(1, MAX_MESH_RETRIES + 1):
                    try:
                        cap    = run_single_sample(m2d, row, thickness_override_nm=thickness_try)
                        status = "Success" if cap > 0 else "Failed"
                        if cap > 0:
                            break
                    except Exception as mesh_err:
                        backoff_nm = thickness_try * THICKNESS_BACKOFF_FACTOR
                        thickness_try += backoff_nm
                        if attempt < MAX_MESH_RETRIES:
                            print(f"\n  [Retry {attempt}/{MAX_MESH_RETRIES}] Mesh singularity at "
                                  f"{thickness_try - backoff_nm:.0f} nm. "
                                  f"Backing off to {thickness_try:.0f} nm...")
                        else:
                            print(f"\n  [FAILED] Sample {i} exhausted {MAX_MESH_RETRIES} "
                                  f"retries: {str(mesh_err)[:60]}")
                            status = "Failed Extraction"

                row_res = row.to_dict()
                row_res["sim_capacitance_F"] = cap
                row_res["sim_status"]        = status
                row_res["final_thickness_nm"] = thickness_try
                results.append(row_res)

                print(f"[{i+1}/{total_samples}] {smiles[:20]}... -> {cap:.2f} pF/m [{status}]",
                      end="\r")

                if (i + 1) % 10 == 0:
                    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
                current_index += 1

            d.release_desktop(True, True)

        except Exception as e:
            print(f"\nSession crashed, restarting: {str(e)[:80]}")
            os.system("taskkill /F /IM ansysedt.exe /T >nul 2>&1")
            os.system("taskkill /F /IM ansysedtsv.exe /T >nul 2>&1")
            time.sleep(5)

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print("\nSweep Complete!")
    os._exit(0)


if __name__ == "__main__":
    run_simulation_sweep()
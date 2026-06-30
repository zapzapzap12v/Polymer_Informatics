import pandas as pd
import os

INPUT_FILE = "../results/ansys_simulation_results.csv"
OUTPUT_FILE = "../results/ranked_successful_polymers.csv"

def rank_polymers():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    print(f"--- Analyzing {INPUT_FILE} ---")
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Filter only the successful simulations
    success_df = df[df['sim_status'] == 'Success'].copy()
    print(f"Found {len(success_df)} successful simulations out of {len(df)} total.")
    
    # 2. Sort from highest capacitance to lowest
    ranked_df = success_df.sort_values(by='sim_capacitance_F', ascending=False).reset_index(drop=True)
    
    # 3. Print the Top 10 to the console for quick viewing
    print("\n--- TOP 10 HIGHEST CAPACITANCE POLYMERS ---")
    print("-" * 90)
    print(f"{'Rank':<5} | {'Base Material':<15} | {'Capacitance':<15} | {'Thickness':<12} | {'Dielectric':<10} | {'SMILES (Truncated)'}")
    print("-" * 90)
    
    for i, row in ranked_df.head(10).iterrows():
        cap = f"{row['sim_capacitance_F']:.2f} pF/m"
        thick = f"{row['thickness_nm']:.1f} nm"
        eps = f"{row['dielectric_constant']:.2f}"
        smiles_trunc = row['smiles'][:20] + "..." if len(row['smiles']) > 20 else row['smiles']
        
        print(f"#{i+1:<4} | {row['base_material']:<15} | {cap:<15} | {thick:<12} | {eps:<10} | {smiles_trunc}")
    print("-" * 90)
    
    # 4. Save the full sorted list to a new CSV file
    ranked_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n-> Full ranked list containing all {len(ranked_df)} success cases saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    rank_polymers()

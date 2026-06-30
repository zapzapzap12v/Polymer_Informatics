import os
import glob

replacements = {
    'ansys_ensemble_pipeline.pkl': '../files/ansys_ensemble_pipeline.pkl',
    'ansys_gnn_pipeline.pkl': '../files/ansys_gnn_pipeline.pkl',
    'ansys_confusion_matrix.png': '../graphs/ansys_confusion_matrix.png',
    'ansys_error_dist.png': '../graphs/ansys_error_dist.png',
    'ansys_mlp_loss.png': '../graphs/ansys_mlp_loss.png',
    'ansys_predictions.png': '../graphs/ansys_predictions.png',
    'gnn_parity_plot.png': '../graphs/gnn_parity_plot.png',
    'pareto_front.png': '../graphs/pareto_front.png',
    'vae_discovery_distribution.png': '../graphs/vae_discovery_distribution.png',
    'vae_loss_curve.png': '../graphs/vae_loss_curve.png',
    'ansys_simulation_results.csv': '../results/ansys_simulation_results.csv',
    'ansys_sweep_targets.csv': '../results/ansys_sweep_targets.csv',
    'ansys_target_dataset_7200.csv': '../results/ansys_target_dataset_7200.csv',
    'inverse_design_recommendations.csv': '../results/inverse_design_recommendations.csv',
    'pareto_optimal_polymers.csv': '../results/pareto_optimal_polymers.csv',
    'ranked_successful_polymers.csv': '../results/ranked_successful_polymers.csv',
    'Final_Report_ANSYS.md': '../results/Final_Report_ANSYS.md',
    'batch.log': '../results/batch.log'
}

for py_file in glob.glob('*.py'):
    if py_file == 'fix_paths.py': continue
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    original = content
    for old, new in replacements.items():
        # Replace only if it's quoted to avoid partial matches
        content = content.replace(f"'{old}'", f"'{new}'")
        content = content.replace(f'"{old}"', f'"{new}"')
        
    if content != original:
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Updated paths in {py_file}')

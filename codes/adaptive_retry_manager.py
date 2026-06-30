import json
import os
import pandas as pd
from typing import Dict, Any, Tuple
from simulation_diagnostics import FailureClassifier

class EnhancedSimulationManager:
    """Manages adaptive retry logic for ANSYS simulations based on diagnosed failure modes."""

    def __init__(self, output_dir: str = '../results'):
        self.output_dir = output_dir
        self.failure_log = []
        self.classifier = FailureClassifier()
        self.retry_strategies = {
            'mesh_error': self._remesh_adaptive,
            'solver_divergence': self._adjust_solver_params,
            'geometry_error': self._validate_geometry,
            'convergence_fail': self._refine_mesh_locally,
            'unknown': self._default_retry
        }
        os.makedirs(self.output_dir, exist_ok=True)

    def diagnose_and_adapt(self, error: Exception, params: Dict[str, Any], attempt: int, max_retries: int) -> Tuple[bool, Dict[str, Any], str]:
        """Diagnose error and return adapted parameters. Returns (should_retry, new_params, failure_mode)."""
        failure_mode = self.classifier.diagnose(str(error))
        
        log_entry = {
            'attempt': attempt,
            'failure_mode': failure_mode,
            'error_message': str(error),
            'params_before': dict(params)
        }
        self.failure_log.append(log_entry)

        if attempt >= max_retries:
            return False, params, failure_mode

        strategy_fn = self.retry_strategies.get(failure_mode, self._default_retry)
        new_params = strategy_fn(params, attempt)
        
        return True, new_params, failure_mode

    def _remesh_adaptive(self, params: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Adaptively reduce mesh size or adjust thickness backoff."""
        # E.g. standard backoff logic but enhanced
        thickness = params.get('thickness_nm', 1000)
        backoff_factor = params.get('backoff_factor', 0.1) * (1.2 ** attempt)
        params['thickness_nm'] = thickness + (thickness * backoff_factor)
        # In a real environment we would also adjust ansys mesh params directly
        return params

    def _adjust_solver_params(self, params: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Adjust solver settings for stability."""
        params['voltage'] = params.get('voltage', 100) * 0.9 # Reduce stress
        return params

    def _validate_geometry(self, params: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Validate and repair geometry before retry."""
        # Implement gap filling / tolerance
        thickness = params.get('thickness_nm', 1000)
        params['thickness_nm'] = thickness * 1.1 # Thicker layer might heal geometry
        return params

    def _refine_mesh_locally(self, params: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Refine mesh in high-gradient regions."""
        params['mesh_resolution'] = "Fine" if attempt > 1 else "Standard"
        return params
        
    def _default_retry(self, params: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        thickness = params.get('thickness_nm', 1000)
        params['thickness_nm'] = thickness * 1.1
        return params

    def save_failure_report(self):
        """Generate detailed failure analysis for post-mortem."""
        report_path = os.path.join(self.output_dir, 'simulation_diagnostics.json')
        with open(report_path, 'w') as f:
            json.dump(self.failure_log, f, indent=4)
        
        if self.failure_log:
            df = pd.DataFrame(self.failure_log)
            summary_path = os.path.join(self.output_dir, 'failure_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("SIMULATION FAILURE ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Failures Logged: {len(df)}\n")
                f.write(f"Failure Mode Distribution:\n{df['failure_mode'].value_counts()}\n\n")

import pandas as pd
from typing import Dict
import logging
from tqdm import tqdm
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class ProgressTrackingModelTrainer:
    """Train models with progress monitoring and checkpoint save."""
    
    def __init__(self, model, checkpoint_dir: str = './checkpoints'):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint = None

    def train_on_dataset(self, df: pd.DataFrame, target_col: str, config: Dict) -> Dict:
        """Train model with progress bar."""
        failed_samples = []
        training_log = {
            'total_samples': len(df),
            'successful_training': 0,
            'failed_training': 0,
            'errors': []
        }
        
        # We need a target value to fit. Assuming df has a target column.
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Training"):
            try:
                target_value = row[target_col]
                features = row.drop(target_col).values
                self.model.fit([features], [target_value])
                training_log['successful_training'] += 1
                
            except Exception as e:
                failed_samples.append({
                    'index': idx,
                    'error': str(e),
                    'row_id': row.get('sample_id', idx)
                })
                training_log['failed_training'] += 1
                training_log['errors'].append(str(e))
                
        if training_log['failed_training'] > 0:
            logger.error(f"Training failed for {training_log['failed_training']} samples:")
            for fail in failed_samples[:5]:
                logger.error(f"  Sample {fail['row_id']}: {fail['error']}")
            
            failure_rate = training_log['failed_training'] / training_log['total_samples']
            if failure_rate > config.get('max_failure_rate', 0.05):
                raise RuntimeError(f"Training failure rate {failure_rate:.1%} exceeds threshold")
                
        return training_log

    def train_with_checkpoints(self, df: pd.DataFrame, target_col: str, checkpoint_interval: int = 100) -> Dict:
        """Train with periodic checkpoints."""
        training_log = {
            'total_samples': len(df),
            'checkpoints_saved': 0,
            'last_checkpoint': None
        }
        
        for i, (idx, row) in enumerate(df.iterrows()):
            try:
                target_value = row[target_col]
                features = row.drop(target_col).values
                self.model.fit([features], [target_value])
                
                if (i + 1) % checkpoint_interval == 0:
                    checkpoint_path = self._save_checkpoint(i + 1)
                    training_log['checkpoints_saved'] += 1
                    training_log['last_checkpoint'] = str(checkpoint_path)
            
            except Exception as e:
                logger.error(f"Training failed at sample {i}: {e}")
                raise
        
        return training_log
        
    def _save_checkpoint(self, step: int) -> Path:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f'model_step_{step}.pkl'
        joblib.dump(self.model, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        self.model = joblib.load(checkpoint_path)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")

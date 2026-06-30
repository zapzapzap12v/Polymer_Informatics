import os
import random
import numpy as np
import logging
try:
    import tensorflow as tf
except ImportError:
    tf = None

logger = logging.getLogger(__name__)

class DeterministicPipeline:
    """Ensures deterministic execution across various libraries."""
    
    @staticmethod
    def set_random_seed(seed: int = 42):
        """Sets random seeds for reproducibility."""
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        if tf is not None:
            tf.random.set_seed(seed)
            
        logger.info(f"Random seed set to {seed} for os, random, numpy, tensorflow")
        
        # Save seed to metadata
        import json
        os.makedirs('../results', exist_ok=True)
        with open('../results/metadata.json', 'w') as f:
            json.dump({'random_seed': seed}, f)

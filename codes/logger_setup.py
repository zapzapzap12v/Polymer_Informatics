import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime

class PipelineLogger:
    """Configures and provides structured logging for the pipeline."""
    
    _initialized = False
    
    @classmethod
    def setup_logging(cls, log_dir: str = '../logs', level: int = logging.INFO):
        if cls._initialized:
            return
            
        os.makedirs(log_dir, exist_ok=True)
        
        # Log file named by date
        log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")
        
        # Formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File Handler with rotation
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        cls._initialized = True
        logging.getLogger(__name__).info(f"Logging initialized. Writing to {log_file}")

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        if not cls._initialized:
            cls.setup_logging()
        return logging.getLogger(name)

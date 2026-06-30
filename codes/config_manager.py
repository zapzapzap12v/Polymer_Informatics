import os
import sys
import json
import logging
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class AnsysPathResolver:
    """Dynamically locate Ansys installation across platforms."""
    
    WINDOWS_STANDARD_PATHS = [
        r"C:\Program Files\ANSYS Inc\v{version}\ANSYS\bin\ansys{version}",
        r"C:\Program Files (x86)\ANSYS Inc\v{version}\ANSYS\bin\ansys{version}",
    ]
    
    LINUX_STANDARD_PATHS = [
        "/opt/ansys/v{version}/ansys/bin/ansys{version}",
        "/usr/local/ANSYS/v{version}/ansys/bin/ansys{version}",
        "/home/shared/ansys/v{version}/ansys/bin/ansys{version}",
    ]
    
    @classmethod
    def resolve_ansys_path(cls, version: str = None) -> Optional[str]:
        """Find Ansys executable using multiple strategies."""
        
        # Strategy 1: Explicit environment variable
        explicit_path = os.getenv('ANSYS_EXECUTABLE')
        if explicit_path and Path(explicit_path).exists():
            return explicit_path
        
        # Strategy 2: Standard paths
        platform_paths = cls.WINDOWS_STANDARD_PATHS if sys.platform == 'win32' else cls.LINUX_STANDARD_PATHS
        
        for path_template in platform_paths:
            if version:
                path = path_template.format(version=version)
                if Path(path).exists():
                    return path
            else:
                for v in ['2025', '2024', '2023', '2022']:
                    candidate = path_template.format(version=v)
                    if Path(candidate).exists():
                        return candidate
        
        # Strategy 3: Check PATH
        import shutil
        possible_names = ['ansys', 'ansys2025', 'ansys2024', 'ansys2023']
        for name in possible_names:
            found = shutil.which(name)
            if found:
                return found
        
        return None

class ConfigurationLoader:
    """Load and validate configuration from multiple sources."""
    
    @classmethod
    def load_configuration(cls) -> Dict:
        """Load configuration from environment variables."""
        
        ansys_path = AnsysPathResolver.resolve_ansys_path(os.getenv('ANSYS_VERSION'))
        
        config = {
            'ansys': {
                'executable_path': ansys_path,
                'version': os.getenv('ANSYS_VERSION', '2025.2'),
                'license_server': os.getenv('ANSYS_LICENSE_SERVER', 'localhost'),
                'license_port': int(os.getenv('ANSYS_LICENSE_PORT', '1055')),
                'core_count': int(os.getenv('ANSYS_CORE_COUNT', '4')),
                'mesh_resolution': os.getenv('ANSYS_MESH_RESOLUTION', 'Standard'),
            },
            'paths': {
                'working_directory': os.getenv('WORK_DIR', os.getcwd()),
                'temp_directory': os.getenv('TEMP_DIR', str(Path.home() / '.polymerinfo' / 'temp')),
                'output_directory': os.getenv('OUTPUT_DIR', str(Path.cwd() / 'output')),
            },
            'simulation': {
                'max_retries': int(os.getenv('MAX_RETRIES', '3')),
                'random_seed': int(os.getenv('RANDOM_SEED', '42'))
            }
        }
        
        return config

# Global config instance
pipeline_config = ConfigurationLoader.load_configuration()

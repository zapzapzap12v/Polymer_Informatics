import os
import argparse
from typing import Dict, Any, List, Tuple
from pydantic import BaseModel, Field, ValidationError

class AnsysRuntimeConfig(BaseModel):
    """Pydantic model for rigorous input validation of ANSYS configuration."""
    version: str = Field(pattern=r'^20\d{2}\.[1-2]$', description="Ansys version format like 2024.1 or 2025.2")
    core_count: int = Field(ge=1, le=128, description="Number of cores to allocate")
    license_port: int = Field(ge=1, le=65535, description="Valid port number for licensing")
    license_server: str = Field(min_length=1, description="License server hostname or IP")
    mesh_resolution: str = Field(pattern=r'^(Standard|Fine|Coarse)$', description="Mesh quality level")

class InputValidator:
    """Validates runtime inputs and environment configurations."""
    
    @staticmethod
    def validate_ansys_config(config_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate an Ansys config dictionary against the schema."""
        try:
            AnsysRuntimeConfig(**config_dict)
            return True, []
        except ValidationError as e:
            errors = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
            return False, errors
            
    @staticmethod
    def validate_file_path(path: str, require_exists: bool = True, expected_ext: str = None) -> bool:
        """Validate paths provided as CLI arguments."""
        if not path:
            return False
            
        if require_exists and not os.path.exists(path):
            return False
            
        if expected_ext and not path.endswith(expected_ext):
            return False
            
        return True

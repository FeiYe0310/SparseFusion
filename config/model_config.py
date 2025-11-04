"""
Model Configuration

Settings related to model architecture and paths.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    
    # Model paths
    model1_path: str = "models/wizardmath_7b"
    model2_path: str = "models/agentevol-7b"
    
    # Debug/testing
    debug_models: bool = False  # Use lightweight BERT models for debugging
    use_pre_trained: bool = False  # Use pre-trained models
    
    # Model generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    def __post_init__(self):
        """Override model paths if using debug models"""
        if self.debug_models:
            self.model1_path = "models/MathBERT"
            self.model2_path = "models/BERTOverflow"
    
    def to_dict(self):
        """Convert to dictionary for logging"""
        return {
            "model1_path": self.model1_path,
            "model2_path": self.model2_path,
            "debug_models": self.debug_models,
            "use_pre_trained": self.use_pre_trained,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
        }

Model Configuration

Settings related to model architecture and paths.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    
    # Model paths
    model1_path: str = "models/wizardmath_7b"
    model2_path: str = "models/agentevol-7b"
    
    # Debug/testing
    debug_models: bool = False  # Use lightweight BERT models for debugging
    use_pre_trained: bool = False  # Use pre-trained models
    
    # Model generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    def __post_init__(self):
        """Override model paths if using debug models"""
        if self.debug_models:
            self.model1_path = "models/MathBERT"
            self.model2_path = "models/BERTOverflow"
    
    def to_dict(self):
        """Convert to dictionary for logging"""
        return {
            "model1_path": self.model1_path,
            "model2_path": self.model2_path,
            "debug_models": self.debug_models,
            "use_pre_trained": self.use_pre_trained,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
        }


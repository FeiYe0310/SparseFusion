"""
Task Configuration

Settings for multi-task learning and dataset paths.
"""

from dataclasses import dataclass


@dataclass
class TaskConfig:
    """Multi-task learning configuration"""
    
    # Task enablement
    use_gsm8k: bool = True  # Always enabled by default
    use_bfcl_eval: bool = False  # BFCL function calling
    use_mbpp_eval: bool = False  # MBPP code generation
    use_mult4_eval: bool = False  # 4x4 multiplication (DoT)
    use_mult5_eval: bool = False  # 5x5 multiplication (DoT)
    use_bool_eval: bool = False  # Boolean logic (DoT)
    
    # Task weights (should sum to 1.0)
    gsm8k_weight: float = 0.5
    bfcl_weight: float = 0.5
    mbpp_weight: float = 0.33
    mult4_weight: float = 0.0
    mult5_weight: float = 0.0
    bool_weight: float = 0.0
    
    # Dataset paths
    gsm8k_data_path: str = "gsm8k"
    bfcl_data_path: str = "bfcl/data/bfcl_test_200.json"
    mbpp_data_path: str = "mbpp"
    
    def __post_init__(self):
        """Validate task configuration"""
        # Count enabled tasks
        enabled_tasks = sum([
            self.use_gsm8k,
            self.use_bfcl_eval,
            self.use_mbpp_eval,
            self.use_mult4_eval,
            self.use_mult5_eval,
            self.use_bool_eval,
        ])
        
        if enabled_tasks == 0:
            raise ValueError("At least one task must be enabled")
        
        # Check weights sum (only for enabled tasks)
        total_weight = 0.0
        if self.use_gsm8k:
            total_weight += self.gsm8k_weight
        if self.use_bfcl_eval:
            total_weight += self.bfcl_weight
        if self.use_mbpp_eval:
            total_weight += self.mbpp_weight
        if self.use_mult4_eval:
            total_weight += self.mult4_weight
        if self.use_mult5_eval:
            total_weight += self.mult5_weight
        if self.use_bool_eval:
            total_weight += self.bool_weight
        
        # Allow small tolerance for floating point
        if abs(total_weight - 1.0) > 1e-6:
            print(f"⚠️  Warning: Task weights sum to {total_weight:.4f}, not 1.0")
    
    def get_enabled_tasks(self):
        """Get list of enabled task names"""
        tasks = []
        if self.use_gsm8k:
            tasks.append("gsm8k")
        if self.use_bfcl_eval:
            tasks.append("bfcl")
        if self.use_mbpp_eval:
            tasks.append("mbpp")
        if self.use_mult4_eval:
            tasks.append("mult4")
        if self.use_mult5_eval:
            tasks.append("mult5")
        if self.use_bool_eval:
            tasks.append("bool")
        return tasks
    
    def to_dict(self):
        """Convert to dictionary for logging"""
        return {
            "enabled_tasks": self.get_enabled_tasks(),
            "gsm8k_weight": self.gsm8k_weight if self.use_gsm8k else 0.0,
            "bfcl_weight": self.bfcl_weight if self.use_bfcl_eval else 0.0,
            "mbpp_weight": self.mbpp_weight if self.use_mbpp_eval else 0.0,
            "mult4_weight": self.mult4_weight if self.use_mult4_eval else 0.0,
            "mult5_weight": self.mult5_weight if self.use_mult5_eval else 0.0,
            "bool_weight": self.bool_weight if self.use_bool_eval else 0.0,
            "gsm8k_data_path": self.gsm8k_data_path,
            "bfcl_data_path": self.bfcl_data_path,
            "mbpp_data_path": self.mbpp_data_path,
        }

Task Configuration

Settings for multi-task learning and dataset paths.
"""

from dataclasses import dataclass


@dataclass
class TaskConfig:
    """Multi-task learning configuration"""
    
    # Task enablement
    use_gsm8k: bool = True  # Always enabled by default
    use_bfcl_eval: bool = False  # BFCL function calling
    use_mbpp_eval: bool = False  # MBPP code generation
    use_mult4_eval: bool = False  # 4x4 multiplication (DoT)
    use_mult5_eval: bool = False  # 5x5 multiplication (DoT)
    use_bool_eval: bool = False  # Boolean logic (DoT)
    
    # Task weights (should sum to 1.0)
    gsm8k_weight: float = 0.5
    bfcl_weight: float = 0.5
    mbpp_weight: float = 0.33
    mult4_weight: float = 0.0
    mult5_weight: float = 0.0
    bool_weight: float = 0.0
    
    # Dataset paths
    gsm8k_data_path: str = "gsm8k"
    bfcl_data_path: str = "bfcl/data/bfcl_test_200.json"
    mbpp_data_path: str = "mbpp"
    
    def __post_init__(self):
        """Validate task configuration"""
        # Count enabled tasks
        enabled_tasks = sum([
            self.use_gsm8k,
            self.use_bfcl_eval,
            self.use_mbpp_eval,
            self.use_mult4_eval,
            self.use_mult5_eval,
            self.use_bool_eval,
        ])
        
        if enabled_tasks == 0:
            raise ValueError("At least one task must be enabled")
        
        # Check weights sum (only for enabled tasks)
        total_weight = 0.0
        if self.use_gsm8k:
            total_weight += self.gsm8k_weight
        if self.use_bfcl_eval:
            total_weight += self.bfcl_weight
        if self.use_mbpp_eval:
            total_weight += self.mbpp_weight
        if self.use_mult4_eval:
            total_weight += self.mult4_weight
        if self.use_mult5_eval:
            total_weight += self.mult5_weight
        if self.use_bool_eval:
            total_weight += self.bool_weight
        
        # Allow small tolerance for floating point
        if abs(total_weight - 1.0) > 1e-6:
            print(f"⚠️  Warning: Task weights sum to {total_weight:.4f}, not 1.0")
    
    def get_enabled_tasks(self):
        """Get list of enabled task names"""
        tasks = []
        if self.use_gsm8k:
            tasks.append("gsm8k")
        if self.use_bfcl_eval:
            tasks.append("bfcl")
        if self.use_mbpp_eval:
            tasks.append("mbpp")
        if self.use_mult4_eval:
            tasks.append("mult4")
        if self.use_mult5_eval:
            tasks.append("mult5")
        if self.use_bool_eval:
            tasks.append("bool")
        return tasks
    
    def to_dict(self):
        """Convert to dictionary for logging"""
        return {
            "enabled_tasks": self.get_enabled_tasks(),
            "gsm8k_weight": self.gsm8k_weight if self.use_gsm8k else 0.0,
            "bfcl_weight": self.bfcl_weight if self.use_bfcl_eval else 0.0,
            "mbpp_weight": self.mbpp_weight if self.use_mbpp_eval else 0.0,
            "mult4_weight": self.mult4_weight if self.use_mult4_eval else 0.0,
            "mult5_weight": self.mult5_weight if self.use_mult5_eval else 0.0,
            "bool_weight": self.bool_weight if self.use_bool_eval else 0.0,
            "gsm8k_data_path": self.gsm8k_data_path,
            "bfcl_data_path": self.bfcl_data_path,
            "mbpp_data_path": self.mbpp_data_path,
        }


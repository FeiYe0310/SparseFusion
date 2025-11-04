"""
Sparsity Configuration

Settings for sparsity-aware selection and pruning.
"""

from dataclasses import dataclass


@dataclass
class SparsityConfig:
    """Sparsity-aware configuration parameters"""
    
    # Sparsity-aware selection weights
    omega: float = 0.5  # Weight for fitness component
    beta: float = 0.5  # Weight for sparsity component
    tau: float = 1.0  # Softmax temperature for sparsity scores
    alpha: float = 1.0  # Fitness normalization exponent
    epsilon: float = 1e-10  # Threshold for considering parameters as zero
    
    # Pruning parameters
    pruning_sparsity: float = 0.0  # Target sparsity (0.0 = disabled)
    pruning_method: str = "wanda"  # Pruning method: "wanda" or "magnitude"
    
    # Dynamic sparsity scheduling
    use_dynamic_sparsity: bool = False  # Enable dynamic sparsity
    sparsity_min: float = 0.1  # Minimum sparsity value
    sparsity_max: float = 0.6  # Maximum sparsity value
    sparsity_t0: int = 100  # Iterations in first cycle
    sparsity_t_mult: int = 2  # Cycle length multiplier (1=fixed, 2=doubling)
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.omega >= 0 and self.beta >= 0, "Weights must be non-negative"
        assert 0 <= self.omega + self.beta <= 1.0 + 1e-6, "Weights should sum to ~1.0"
        assert self.pruning_method in ["wanda", "magnitude"], "Invalid pruning method"
        if self.use_dynamic_sparsity:
            assert 0 <= self.sparsity_min < self.sparsity_max <= 1.0, \
                "Invalid sparsity range"
    
    def to_dict(self):
        """Convert to dictionary for logging"""
        return {
            "omega": self.omega,
            "beta": self.beta,
            "tau": self.tau,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "pruning_sparsity": self.pruning_sparsity,
            "pruning_method": self.pruning_method,
            "use_dynamic_sparsity": self.use_dynamic_sparsity,
            "sparsity_min": self.sparsity_min,
            "sparsity_max": self.sparsity_max,
            "sparsity_t0": self.sparsity_t0,
            "sparsity_t_mult": self.sparsity_t_mult,
        }


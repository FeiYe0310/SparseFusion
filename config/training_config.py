"""
Training Configuration

Settings for evolutionary algorithm training.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    
    # Evolution parameters
    pop_size: int = 20  # Population size for the archive
    total_forward_passes: int = 50000  # Total number of forward passes
    runs: int = 10  # Number of independent runs
    
    # Evolutionary operators
    no_crossover: bool = False  # Disable crossover operation
    no_splitpoint: bool = False  # Disable splitpoint in crossover
    no_matchmaker: bool = False  # Disable matchmaker for parent selection
    
    # Distributed training
    distributed: bool = False  # Enable multi-node distributed execution
    archive_backend: str = "gpu"  # Where to place archive: "gpu" or "cpu"
    
    # Logging and saving
    store_train_results: bool = False  # Store training results
    output_dir: str = "results"  # Output directory for results
    no_save_best_model: bool = False  # Do not save final best model
    log_sparsity_stats: bool = False  # Log detailed sparsity statistics
    
    def to_dict(self):
        """Convert to dictionary for logging"""
        return {
            "pop_size": self.pop_size,
            "total_forward_passes": self.total_forward_passes,
            "runs": self.runs,
            "no_crossover": self.no_crossover,
            "no_splitpoint": self.no_splitpoint,
            "no_matchmaker": self.no_matchmaker,
            "distributed": self.distributed,
            "archive_backend": self.archive_backend,
            "store_train_results": self.store_train_results,
            "output_dir": self.output_dir,
            "no_save_best_model": self.no_save_best_model,
            "log_sparsity_stats": self.log_sparsity_stats,
        }


Training Configuration

Settings for evolutionary algorithm training.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    
    # Evolution parameters
    pop_size: int = 20  # Population size for the archive
    total_forward_passes: int = 50000  # Total number of forward passes
    runs: int = 10  # Number of independent runs
    
    # Evolutionary operators
    no_crossover: bool = False  # Disable crossover operation
    no_splitpoint: bool = False  # Disable splitpoint in crossover
    no_matchmaker: bool = False  # Disable matchmaker for parent selection
    
    # Distributed training
    distributed: bool = False  # Enable multi-node distributed execution
    archive_backend: str = "gpu"  # Where to place archive: "gpu" or "cpu"
    
    # Logging and saving
    store_train_results: bool = False  # Store training results
    output_dir: str = "results"  # Output directory for results
    no_save_best_model: bool = False  # Do not save final best model
    log_sparsity_stats: bool = False  # Log detailed sparsity statistics
    
    def to_dict(self):
        """Convert to dictionary for logging"""
        return {
            "pop_size": self.pop_size,
            "total_forward_passes": self.total_forward_passes,
            "runs": self.runs,
            "no_crossover": self.no_crossover,
            "no_splitpoint": self.no_splitpoint,
            "no_matchmaker": self.no_matchmaker,
            "distributed": self.distributed,
            "archive_backend": self.archive_backend,
            "store_train_results": self.store_train_results,
            "output_dir": self.output_dir,
            "no_save_best_model": self.no_save_best_model,
            "log_sparsity_stats": self.log_sparsity_stats,
        }


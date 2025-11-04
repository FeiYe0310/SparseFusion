"""
Evaluation Configuration

Settings for model evaluation and batch processing.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters"""
    
    # Batch sizes
    batch_size: int = 16  # Batch size for evaluation
    
    # Subset evaluation (for speed)
    eval_subset_size: int = 15  # Samples per iteration (None=all data)
    eval_on_test_subset: bool = False  # Use test split for evaluation
    test_eval_subset_size: int = 15  # Subset size for periodic evaluation
    
    # Per-task subset sizes (overrides)
    eval_subset_size_gsm8k: Optional[int] = None  # GSM8K-specific subset
    eval_subset_size_mbpp: Optional[int] = None  # MBPP-specific subset
    eval_subset_size_bfcl: Optional[int] = None  # BFCL-specific subset
    
    # Few-shot prompting for GSM8K
    gsm8k_qwen_chat: bool = False  # Use Qwen chat template for GSM8K
    gsm8k_few_shot_k: int = 3  # Number of few-shot examples
    gsm8k_few_shot_split: str = "train"  # Split to sample examples from
    
    # Few-shot prompting for MBPP
    mbpp_qwen_chat: bool = False  # Use Qwen chat template for MBPP
    mbpp_few_shot_k: int = 3  # Number of few-shot examples
    mbpp_few_shot_split: str = "train"  # Split to sample examples from
    
    # Few-shot prompting for DoT tasks
    mult4_qwen_chat: bool = False  # Use Qwen chat for 4x4 mult
    mult4_few_shot_k: int = 3
    bool_qwen_chat: bool = False  # Use Qwen chat for boolean logic
    bool_few_shot_k: int = 3
    
    def to_dict(self):
        """Convert to dictionary for logging"""
        return {
            "batch_size": self.batch_size,
            "eval_subset_size": self.eval_subset_size,
            "eval_on_test_subset": self.eval_on_test_subset,
            "test_eval_subset_size": self.test_eval_subset_size,
            "eval_subset_size_gsm8k": self.eval_subset_size_gsm8k,
            "eval_subset_size_mbpp": self.eval_subset_size_mbpp,
            "eval_subset_size_bfcl": self.eval_subset_size_bfcl,
            "gsm8k_qwen_chat": self.gsm8k_qwen_chat,
            "gsm8k_few_shot_k": self.gsm8k_few_shot_k,
            "mbpp_qwen_chat": self.mbpp_qwen_chat,
            "mbpp_few_shot_k": self.mbpp_few_shot_k,
            "mult4_qwen_chat": self.mult4_qwen_chat,
            "mult4_few_shot_k": self.mult4_few_shot_k,
            "bool_qwen_chat": self.bool_qwen_chat,
            "bool_few_shot_k": self.bool_few_shot_k,
        }


Evaluation Configuration

Settings for model evaluation and batch processing.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters"""
    
    # Batch sizes
    batch_size: int = 16  # Batch size for evaluation
    
    # Subset evaluation (for speed)
    eval_subset_size: int = 15  # Samples per iteration (None=all data)
    eval_on_test_subset: bool = False  # Use test split for evaluation
    test_eval_subset_size: int = 15  # Subset size for periodic evaluation
    
    # Per-task subset sizes (overrides)
    eval_subset_size_gsm8k: Optional[int] = None  # GSM8K-specific subset
    eval_subset_size_mbpp: Optional[int] = None  # MBPP-specific subset
    eval_subset_size_bfcl: Optional[int] = None  # BFCL-specific subset
    
    # Few-shot prompting for GSM8K
    gsm8k_qwen_chat: bool = False  # Use Qwen chat template for GSM8K
    gsm8k_few_shot_k: int = 3  # Number of few-shot examples
    gsm8k_few_shot_split: str = "train"  # Split to sample examples from
    
    # Few-shot prompting for MBPP
    mbpp_qwen_chat: bool = False  # Use Qwen chat template for MBPP
    mbpp_few_shot_k: int = 3  # Number of few-shot examples
    mbpp_few_shot_split: str = "train"  # Split to sample examples from
    
    # Few-shot prompting for DoT tasks
    mult4_qwen_chat: bool = False  # Use Qwen chat for 4x4 mult
    mult4_few_shot_k: int = 3
    bool_qwen_chat: bool = False  # Use Qwen chat for boolean logic
    bool_few_shot_k: int = 3
    
    def to_dict(self):
        """Convert to dictionary for logging"""
        return {
            "batch_size": self.batch_size,
            "eval_subset_size": self.eval_subset_size,
            "eval_on_test_subset": self.eval_on_test_subset,
            "test_eval_subset_size": self.test_eval_subset_size,
            "eval_subset_size_gsm8k": self.eval_subset_size_gsm8k,
            "eval_subset_size_mbpp": self.eval_subset_size_mbpp,
            "eval_subset_size_bfcl": self.eval_subset_size_bfcl,
            "gsm8k_qwen_chat": self.gsm8k_qwen_chat,
            "gsm8k_few_shot_k": self.gsm8k_few_shot_k,
            "mbpp_qwen_chat": self.mbpp_qwen_chat,
            "mbpp_few_shot_k": self.mbpp_few_shot_k,
            "mult4_qwen_chat": self.mult4_qwen_chat,
            "mult4_few_shot_k": self.mult4_few_shot_k,
            "bool_qwen_chat": self.bool_qwen_chat,
            "bool_few_shot_k": self.bool_few_shot_k,
        }


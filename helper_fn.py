import jax
import jax.numpy as jnp
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from typing import List, Tuple, Any, Union
from tqdm.auto import tqdm


def pytorch_to_jax_flattened(
    model: torch.nn.Module,
) -> Tuple[jnp.ndarray, List[Tuple[Any, Any]], int]:
    """
    Flattens a PyTorch model's parameters into a single JAX array
    and returns the flattened parameters, shapes, and total parameter count.
    The final JAX array is cast to bfloat16 to save memory.
    """
    params = []
    param_shapes = []
    total_params = 0
    param_list = list(model.parameters())
    for p in tqdm(param_list, desc="Flattening PyTorch parameters"):
        # Convert to float32 for numpy compatibility, then to numpy array
        param_np = p.detach().cpu().to(torch.float32).numpy()
        params.append(param_np.flatten())
        total_params += p.numel()
        # Store the original torch dtype, not the numpy dtype
        param_shapes.append((p.shape, p.dtype))

    # Create the concatenated array in float32 first
    flat_params_f32 = jnp.concatenate([jnp.asarray(p) for p in params])
    # Cast to bfloat16 to save significant memory in the JAX archive
    flat_params_bf16 = flat_params_f32.astype(jnp.bfloat16)

    return flat_params_bf16, param_shapes, total_params


def jax_flattened_to_pytorch_model(
    flat_params: jnp.ndarray,
    model_skeleton: torch.nn.Module,
    param_shapes: List[Tuple[Any, Any]],
) -> torch.nn.Module:
    """
    Loads flattened JAX parameters back into a PyTorch model skeleton
    by directly updating the model's parameter data, processing chunk by chunk
    to avoid creating a large intermediate float32 copy in RAM.
    """
    current_pos = 0

    with torch.no_grad():
        params_to_update = list(model_skeleton.parameters())
        if len(params_to_update) != len(param_shapes):
            raise ValueError(
                "Model structure mismatch: The number of parameters in the skeleton model and the shape specification do not match."
            )

        for i, (shape, dtype) in enumerate(
            tqdm(param_shapes, desc="Restoring PyTorch parameters")
        ):
            # Calculate the number of elements for the current parameter
            param_numel = np.prod(shape).item()

            # Slice the corresponding chunk from the flattened bfloat16 JAX array
            chunk_bf16 = jax.lax.dynamic_slice_in_dim(
                flat_params, current_pos, param_numel
            )

            # Convert only the small chunk to float32 for numpy conversion
            chunk_f32_np = np.array(chunk_bf16.astype(jnp.float32))

            # Ensure the sliced chunk has the correct number of elements
            if params_to_update[i].numel() != param_numel:
                raise ValueError(
                    f"Shape mismatch at parameter {i}. Expected {params_to_update[i].numel()} elements, but got {param_numel} from shapes list."
                )

            # Reshape and convert back to a float32 PyTorch tensor first
            tensor_chunk = torch.from_numpy(chunk_f32_np.reshape(shape))

            # Now, cast to the original target dtype (e.g., bfloat16)
            tensor_chunk = tensor_chunk.to(dtype)

            # Update the parameter data in place
            params_to_update[i].data.copy_(tensor_chunk)

            # Move to the next position
            current_pos += param_numel

    return model_skeleton


def slerp(val: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    # Normalize the inputs.
    norm_x = x / jnp.linalg.norm(x)
    norm_y = y / jnp.linalg.norm(y)

    # Cosine of the angle.
    dot = jnp.dot(norm_x, norm_y)
    omega = jnp.arccos(jnp.clip(dot, -1, 1))
    sin_omega = jnp.sin(omega)

    # Calculate scales for input vectors.
    scale_x = jnp.sin((1.0 - val) * omega) / sin_omega
    scale_y = jnp.sin(val * omega) / sin_omega

    # Linear interpolation weights.
    lin_scale_x = 1.0 - val
    lin_scale_y = val

    return jnp.where(
        sin_omega > 1e-6, scale_x * x + scale_y * y, lin_scale_x * x + lin_scale_y * y
    )


@jax.jit
def crossover_without_splitpoint(
    parents: tuple[jnp.ndarray, jnp.ndarray], rand_key: jnp.ndarray
) -> jnp.ndarray:
    w = jax.random.uniform(rand_key)
    return slerp(w, parents[0], parents[1])


def slerp_w_splitpoint(
    val: jnp.ndarray, split_point: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    val = jnp.ones_like(x) * val
    mask = jnp.arange(len(val)) < split_point
    val = jnp.where(mask, val, 1 - val)

    # Normalize the inputs.
    norm_x = x / jnp.linalg.norm(x)
    norm_y = y / jnp.linalg.norm(y)

    # Cosine of the angle.
    dot = jnp.dot(norm_x, norm_y)
    omega = jnp.arccos(jnp.clip(dot, -1, 1))
    sin_omega = jnp.sin(omega)

    # Calculate scales for input vectors.
    scale_x = jnp.sin((1.0 - val) * omega) / sin_omega
    scale_y = jnp.sin(val * omega) / sin_omega

    # Linear interpolation weights.
    lin_scale_x = 1.0 - val
    lin_scale_y = val

    return jnp.where(
        sin_omega > 1e-6, scale_x * x + scale_y * y, lin_scale_x * x + lin_scale_y * y
    )


@jax.jit
def crossover(
    parents: tuple[jnp.ndarray, jnp.ndarray], rand_key: jnp.ndarray
) -> jnp.ndarray:
    k1, k2 = jax.random.split(rand_key)
    # Use int64 for the split point to handle models with more than 2^31 parameters.
    split_point = jax.random.randint(
        k1, shape=(), minval=0, maxval=parents[0].shape[0], dtype=jnp.int64
    )
    w = jax.random.uniform(k2)
    return slerp_w_splitpoint(w, split_point, parents[0], parents[1])


@jax.jit
def mutate(params: jnp.ndarray, rand_key: jnp.ndarray, std: float = 0.01):
    noise = jax.random.normal(rand_key, shape=params.shape) * std
    return params + noise


def _extend_embeddings_with_zeros(model: torch.nn.Module, new_vocab_size: int) -> None:
    embedding = model.get_input_embeddings()
    if embedding is None:
        return

    old_vocab_size, _ = embedding.weight.shape
    if new_vocab_size <= old_vocab_size:
        return

    old_weight = embedding.weight.detach().clone()
    model.resize_token_embeddings(new_vocab_size)
    with torch.no_grad():
        updated_embedding = model.get_input_embeddings()
        updated_embedding.weight.zero_()
        updated_embedding.weight[:old_vocab_size] = old_weight


def _remap_embeddings_to_vocab(
    model: torch.nn.Module, original_vocab: dict[str, int], merged_vocab: dict[str, int]
) -> None:
    embedding = model.get_input_embeddings()
    if embedding is None:
        return

    old_weight = embedding.weight.detach().clone()
    new_vocab_size = len(merged_vocab)
    model.resize_token_embeddings(new_vocab_size)

    with torch.no_grad():
        updated_embedding = model.get_input_embeddings()
        updated_embedding.weight.zero_()
        for token, original_idx in tqdm(
            original_vocab.items(),
            desc="Remapping embeddings",
            total=len(original_vocab),
        ):
            new_idx = merged_vocab.get(token)
            if new_idx is not None:
                updated_embedding.weight[new_idx] = old_weight[original_idx]


def merge_tokenizers_and_align_models(
    model1: torch.nn.Module,
    tokenizer1,
    model2: torch.nn.Module,
    tokenizer2,
):
    """Merge two tokenizers and align model embeddings to the merged vocabulary."""
    vocab1 = tokenizer1.get_vocab()
    vocab2 = tokenizer2.get_vocab()

    if vocab1 == vocab2:
        return tokenizer1, model1, model2

    tokens_to_add = [
        token
        for token, _ in sorted(vocab2.items(), key=lambda item: item[1])
        if token not in vocab1
    ]

    if tokens_to_add:
        tokenizer1.add_tokens(tokens_to_add)
        vocab1 = tokenizer1.get_vocab()

    merged_vocab = vocab1
    merged_vocab_size = len(merged_vocab)

    _extend_embeddings_with_zeros(model1, merged_vocab_size)
    _remap_embeddings_to_vocab(model2, vocab2, merged_vocab)

    if hasattr(model1.config, "vocab_size"):
        model1.config.vocab_size = merged_vocab_size
    if hasattr(model2.config, "vocab_size"):
        model2.config.vocab_size = merged_vocab_size

    for model in (model1, model2):
        tie_weights = getattr(model, "tie_weights", None)
        if callable(tie_weights):
            try:
                tie_weights()
            except Exception:
                pass

    return tokenizer1, model1, model2


def _load_model(model_path: str, dtype: torch.dtype) -> torch.nn.Module:
    try:
        return AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    except Exception:
        return AutoModel.from_pretrained(model_path, torch_dtype=dtype)


def get_pre_trained_models(
    model1_path: str,
    model2_path: str,
    *,
    return_tokenizer: bool = False,
) -> Union[
    Tuple[jnp.ndarray, jnp.ndarray, List[Tuple[Any, Any]]],
    Tuple[jnp.ndarray, jnp.ndarray, List[Tuple[Any, Any]], Any],
]:
    """
    Loads two pre-trained models, merges their tokenizers (extending the first with the
    second), aligns the embedding weights to the merged vocabulary, then pads the
    smaller parameter vector and returns both flattened models plus shape metadata.
    """
    # For 7B models, it's recommended to use bfloat16 to save memory
    dtype = torch.bfloat16

    print(f"Loading model 1 from: {model1_path} with dtype={dtype}")
    model1 = _load_model(model1_path, dtype)

    print(f"Loading model 2 from: {model2_path} with dtype={dtype}")
    model2 = _load_model(model2_path, dtype)

    # Ensure models are in evaluation mode
    model1.eval()
    model2.eval()

    tokenizer1 = AutoTokenizer.from_pretrained(model1_path)
    tokenizer2 = AutoTokenizer.from_pretrained(model2_path)

    tokenizer1, model1, model2 = merge_tokenizers_and_align_models(
        model1, tokenizer1, model2, tokenizer2
    )

    print("Flattening parameters of both models...")
    flat_params1, param_shapes1, total_params1 = pytorch_to_jax_flattened(model1)
    flat_params2, param_shapes2, total_params2 = pytorch_to_jax_flattened(model2)

    # --- Memory Optimization ---
    del model1
    del model2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Cleaned up initial PyTorch models from memory.")

    # --- Padding Logic ---
    if total_params1 > total_params2:
        print(
            f"Model 1 ({total_params1} params) is larger than Model 2 ({total_params2} params). Padding Model 2."
        )
    elif total_params2 > total_params1:
        print(
            f"Model 2 ({total_params2} params) is larger than Model 1 ({total_params1} params). Padding Model 1."
        )
    else:
        print("Both models have the same number of parameters. No padding needed.")

    flat_params1, flat_params2, final_param_shapes = pad_flattened_params(
        flat_params1,
        total_params1,
        param_shapes1,
        flat_params2,
        total_params2,
        param_shapes2,
    )

    print("Model loading, flattening, and padding complete.")

    if return_tokenizer:
        return flat_params1, flat_params2, final_param_shapes, tokenizer1
    return flat_params1, flat_params2, final_param_shapes

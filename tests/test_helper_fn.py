import random
import numpy as np

import torch
import pytest
import jax

from helper_fn import (
    merge_tokenizers_and_align_models,
    pytorch_to_jax_flattened,
    jax_flattened_to_pytorch_model,
)

try:
    from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
except ImportError:
    AutoModel = AutoTokenizer = AutoModelForMaskedLM = None


def _version_tuple(version_str: str) -> tuple[int, int, int]:
    core = version_str.split("+")[0]
    parts: list[int] = []
    for item in core.split("."):
        digits = "".join(ch for ch in item if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


class _TinyToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 16)
        self.fc2 = torch.nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.mark.skipif(AutoModel is None, reason="transformers is required for this test")
def test_merged_tokenizer_alignment_with_mathbert_and_bertoverflow() -> None:
    if _version_tuple(torch.__version__) < (2, 6, 0):
        pytest.skip("Torch >= 2.6.0 required to load these checkpoints safely.")

    try:
        model1 = AutoModel.from_pretrained("models/MathBERT").eval()
        tokenizer1 = AutoTokenizer.from_pretrained("models/MathBERT")
        model2 = AutoModel.from_pretrained("models/BERTOverflow").eval()
        tokenizer2 = AutoTokenizer.from_pretrained("models/BERTOverflow")
    except OSError as exc:
        pytest.skip(f"Required models not available locally: {exc}")

    vocab1 = tokenizer1.get_vocab()
    vocab2 = tokenizer2.get_vocab()
    original_embedding1 = model1.get_input_embeddings().weight.detach().clone()
    original_embedding2 = model2.get_input_embeddings().weight.detach().clone()

    merged_tokenizer, aligned_model1, aligned_model2 = (
        merge_tokenizers_and_align_models(model1, tokenizer1, model2, tokenizer2)
    )

    merged_vocab = merged_tokenizer.get_vocab()
    union_size = len(set(vocab1) | set(vocab2))
    assert len(merged_vocab) == union_size

    aligned_embedding1 = aligned_model1.get_input_embeddings().weight.detach()
    aligned_embedding2 = aligned_model2.get_input_embeddings().weight.detach()

    unique_to_vocab2 = sorted(set(vocab2) - set(vocab1), key=vocab2.get)
    for token in unique_to_vocab2[:50]:
        idx = merged_vocab[token]
        torch.testing.assert_close(
            aligned_embedding1[idx], torch.zeros_like(aligned_embedding1[idx])
        )

    unique_to_vocab1 = sorted(set(vocab1) - set(vocab2), key=vocab1.get)
    for token in unique_to_vocab1[:50]:
        idx = merged_vocab[token]
        torch.testing.assert_close(
            aligned_embedding2[idx], torch.zeros_like(aligned_embedding2[idx])
        )

    shared_tokens = sorted(set(vocab1).intersection(vocab2), key=vocab1.get)[:50]
    for token in shared_tokens:
        new_idx = merged_vocab[token]
        torch.testing.assert_close(
            aligned_embedding1[new_idx],
            original_embedding1[vocab1[token]],
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            aligned_embedding2[new_idx],
            original_embedding2[vocab2[token]],
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.skipif(AutoModel is None, reason="transformers is required for this test")
def test_merge_tokenizers_is_noop_for_identical_qwen_vocab() -> None:
    if _version_tuple(torch.__version__) < (2, 6, 0):
        pytest.skip("Torch >= 2.6.0 required to load these checkpoints safely.")

    try:
        model1 = AutoModel.from_pretrained("models/Qwen2.5-Coder-0.5B-Instruct").eval()
        tokenizer1 = AutoTokenizer.from_pretrained("models/Qwen2.5-Coder-0.5B-Instruct")
        model2 = AutoModel.from_pretrained("models/Qwen2.5-0.5B-Instruct").eval()
        tokenizer2 = AutoTokenizer.from_pretrained("models/Qwen2.5-0.5B-Instruct")
    except OSError as exc:
        pytest.skip(f"Required models not available locally: {exc}")

    vocab1 = tokenizer1.get_vocab()
    vocab2 = tokenizer2.get_vocab()
    assert vocab1 == vocab2

    original_embedding1 = model1.get_input_embeddings().weight.detach().clone()
    original_embedding2 = model2.get_input_embeddings().weight.detach().clone()

    merged_tokenizer, aligned_model1, aligned_model2 = (
        merge_tokenizers_and_align_models(model1, tokenizer1, model2, tokenizer2)
    )

    assert merged_tokenizer is tokenizer1
    assert aligned_model1 is model1
    assert aligned_model2 is model2

    assert merged_tokenizer.get_vocab() == vocab1
    assert tokenizer2.get_vocab() == vocab2

    torch.testing.assert_close(
        aligned_model1.get_input_embeddings().weight.detach(),
        original_embedding1,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        aligned_model2.get_input_embeddings().weight.detach(),
        original_embedding2,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(AutoModel is None, reason="transformers is required for this test")
def test_model_outputs_preserved_after_embedding_alignment() -> None:
    if _version_tuple(torch.__version__) < (2, 6, 0):
        pytest.skip("Torch >= 2.6.0 required to load these checkpoints safely.")

    try:
        model1 = AutoModel.from_pretrained("models/MathBERT").eval()
        tokenizer1 = AutoTokenizer.from_pretrained("models/MathBERT")
        model2 = AutoModel.from_pretrained("models/BERTOverflow").eval()
        tokenizer2 = AutoTokenizer.from_pretrained("models/BERTOverflow")
    except OSError as exc:
        pytest.skip(f"Required models not available locally: {exc}")

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Math and programming questions appear on Stack Overflow.",
    ]

    encoded1 = tokenizer1(sentences, return_tensors="pt", padding=True)
    encoded2 = tokenizer2(sentences, return_tensors="pt", padding=True)

    input_ids1 = encoded1["input_ids"]
    attention_mask1 = encoded1["attention_mask"]
    input_ids2 = encoded2["input_ids"]
    attention_mask2 = encoded2["attention_mask"]

    def _hidden_state(output):
        return (
            output.last_hidden_state
            if hasattr(output, "last_hidden_state")
            else output[0]
        )

    with torch.no_grad():
        original_output1 = _hidden_state(
            model1(input_ids=input_ids1, attention_mask=attention_mask1)
        )
        original_output2 = _hidden_state(
            model2(input_ids=input_ids2, attention_mask=attention_mask2)
        )

    tokens1 = [tokenizer1.convert_ids_to_tokens(seq.tolist()) for seq in input_ids1]
    tokens2 = [tokenizer2.convert_ids_to_tokens(seq.tolist()) for seq in input_ids2]

    merged_tokenizer, aligned_model1, aligned_model2 = (
        merge_tokenizers_and_align_models(model1, tokenizer1, model2, tokenizer2)
    )
    merged_vocab = merged_tokenizer.get_vocab()

    def tokens_to_ids(token_sequences: list[list[str]]) -> torch.Tensor:
        mapped = []
        for seq in token_sequences:
            mapped.append([merged_vocab[token] for token in seq])
        return torch.tensor(mapped, dtype=torch.long)

    new_input_ids1 = tokens_to_ids(tokens1).to(input_ids1.device)
    new_input_ids2 = tokens_to_ids(tokens2).to(input_ids2.device)

    with torch.no_grad():
        new_output1 = _hidden_state(
            aligned_model1(input_ids=new_input_ids1, attention_mask=attention_mask1)
        )
        new_output2 = _hidden_state(
            aligned_model2(input_ids=new_input_ids2, attention_mask=attention_mask2)
        )

    torch.testing.assert_close(new_output1, original_output1, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(new_output2, original_output2, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(jax is None, reason="jax is required for this test")
def test_toy_model_roundtrip_through_jax_flattening() -> None:
    torch.manual_seed(0)
    original_model = _TinyToyModel().eval()

    flat_params, param_shapes, total_params = pytorch_to_jax_flattened(original_model)
    assert total_params == sum(p.numel() for p in original_model.parameters())

    skeleton_model = _TinyToyModel().eval()
    with torch.no_grad():
        for param in skeleton_model.parameters():
            param.zero_()

    reconstructed_model = jax_flattened_to_pytorch_model(
        flat_params, skeleton_model, param_shapes
    )

    for original_tensor, reconstructed_tensor in zip(
        original_model.parameters(), reconstructed_model.parameters()
    ):
        torch.testing.assert_close(
            reconstructed_tensor, original_tensor, rtol=1e-3, atol=1e-3
        )


@pytest.mark.skipif(
    AutoModelForMaskedLM is None,
    reason="transformers masked LM is required for this test",
)
def test_bert_overflow_roundtrip_preserves_masked_prediction_text() -> None:
    if _version_tuple(torch.__version__) < (2, 6, 0):
        pytest.skip("Torch >= 2.6.0 required to load these checkpoints safely.")

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    try:
        original_model = AutoModelForMaskedLM.from_pretrained(
            "models/BERTOverflow"
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained("models/BERTOverflow")
    except OSError as exc:
        pytest.skip(f"Required models not available locally: {exc}")

    flat_params, param_shapes, _ = pytorch_to_jax_flattened(original_model)

    try:
        skeleton_model = AutoModelForMaskedLM.from_pretrained(
            "models/BERTOverflow"
        ).eval()
    except OSError as exc:
        pytest.skip(f"Required models not available locally: {exc}")

    with torch.no_grad():
        for param in skeleton_model.parameters():
            param.zero_()

    reconstructed_model = jax_flattened_to_pytorch_model(
        flat_params, skeleton_model, param_shapes
    )

    prompt = "The quick brown fox [MASK] over the lazy dog."
    encoded = tokenizer(prompt, return_tensors="pt")

    def _fill_mask(model: torch.nn.Module) -> str:
        with torch.no_grad():
            logits = model(**encoded).logits
        mask_positions = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero(
            as_tuple=True
        )[0]
        if mask_positions.numel() != 1:
            raise AssertionError("Expected exactly one [MASK] token in the prompt.")
        mask_index = mask_positions.item()
        predicted_id = logits[0, mask_index].argmax(dim=-1).item()
        filled_ids = encoded["input_ids"].clone()
        filled_ids[0, mask_index] = predicted_id
        return tokenizer.decode(filled_ids[0], skip_special_tokens=True)

    original_text = _fill_mask(original_model)
    reconstructed_text = _fill_mask(reconstructed_model)

    assert reconstructed_text == original_text


@pytest.mark.skipif(AutoModel is None, reason="transformers is required for this test")
def test_bert_roundtrip_through_jax_flattening() -> None:
    if _version_tuple(torch.__version__) < (2, 6, 0):
        pytest.skip("Torch >= 2.6.0 required to load these checkpoints safely.")

    try:
        original_model = AutoModel.from_pretrained("models/BERTOverflow").eval()
    except OSError as exc:
        pytest.skip(f"Required models not available locally: {exc}")

    flat_params, param_shapes, _ = pytorch_to_jax_flattened(original_model)

    try:
        skeleton_model = AutoModel.from_pretrained("models/BERTOverflow").eval()
    except OSError as exc:
        pytest.skip(f"Required models not available locally: {exc}")

    with torch.no_grad():
        for param in skeleton_model.parameters():
            param.zero_()

    reconstructed_model = jax_flattened_to_pytorch_model(
        flat_params, skeleton_model, param_shapes
    )

    original_params = list(original_model.parameters())
    reconstructed_params = list(reconstructed_model.parameters())

    assert len(original_params) == len(reconstructed_params)
    for original_tensor, reconstructed_tensor in zip(
        original_params, reconstructed_params
    ):
        torch.testing.assert_close(
            reconstructed_tensor, original_tensor, rtol=1e-3, atol=1e-3
        )

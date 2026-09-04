"""Minimal local-only Hugging Face/PEFT helpers for real adapter updates."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


def load_lora_model(config: dict[str, Any], model_path: Path, *, adapter_path: Path | None = None):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"local_files_only": True, "torch_dtype": "auto"}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if adapter_path is None:
        lora = LoraConfig(
            r=int(config["training"]["lora_rank"]),
            lora_alpha=int(config["training"]["lora_alpha"]),
            lora_dropout=float(config["training"]["lora_dropout"]),
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    else:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    if not torch.cuda.is_available():
        model.to("mps" if torch.backends.mps.is_available() else "cpu")
    return model, tokenizer


def completion_tensors(tokenizer, prompt: str, completion: str, device, max_length: int = 4096):
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
    full = tokenizer(prompt + completion, add_special_tokens=True, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = full["input_ids"].to(device)
    attention = full["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[:, : min(len(prompt_ids), labels.shape[1])] = -100
    return input_ids, attention, labels


def sequence_logprob(model, tokenizer, prompt: str, completion: str, *, reference: bool = False):
    device = next(model.parameters()).device
    input_ids, attention, labels = completion_tensors(tokenizer, prompt, completion, device)
    context = model.disable_adapter() if reference and hasattr(model, "disable_adapter") else nullcontext()
    with context:
        logits = model(input_ids=input_ids, attention_mask=attention).logits[:, :-1]
    target = input_ids[:, 1:]
    mask = labels[:, 1:] != -100
    token_logp = torch.log_softmax(logits.float(), dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum() / mask.sum().clamp_min(1)

"""Build frozen Qwen3-Embedding text vectors using a local model directory only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, T5EncoderModel

from .data import read_manifest


def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask[:, -1].bool().all():
        return hidden_states[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, lengths]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--encoder", choices=("qwen3", "flan_t5"), default="qwen3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--instruction-prefix", default="")
    args = parser.parse_args()
    if not args.model_path.is_dir():
        raise FileNotFoundError(
            f"Local Qwen3 model directory is unavailable: {args.model_path}; no download attempted"
        )
    rows = read_manifest(args.manifest)
    texts_by_id = {}
    for row in rows:
        existing = texts_by_id.setdefault(row["stimulus_id"], row["text"])
        if existing != row["text"]:
            raise ValueError(f"Different text strings share stimulus_id {row['stimulus_id']}")
    stimulus_ids = sorted(texts_by_id)
    texts = [args.instruction_prefix + texts_by_id[key] for key in stimulus_ids]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True, padding_side="left"
    )
    model_cls = T5EncoderModel if args.encoder == "flan_t5" else AutoModel
    model = model_cls.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=(args.encoder == "qwen3"),
        torch_dtype=torch.bfloat16,
    ).cuda().eval()
    vectors = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        # These items define a symmetric text neighborhood and act as retrieval
        # candidates, so encode them as documents without a query-only prompt.
        tokens = tokenizer(
            batch, padding=True, truncation=True, max_length=args.max_length,
            return_tensors="pt",
        ).to("cuda")
        outputs = model(**tokens)
        if args.encoder == "flan_t5":
            valid = tokens["attention_mask"].bool()
            hidden = outputs.last_hidden_state.float().masked_fill(~valid[..., None], 0)
            embeddings = hidden.sum(1) / valid.sum(1, keepdim=True)
        else:
            embeddings = last_token_pool(outputs.last_hidden_state, tokens["attention_mask"])
        # Qwen3 embeddings are cosine-normalized at extraction. GLIM's FLAN
        # baseline keeps the mean-pooled encoder states unnormalized and lets
        # the learned text projection handle their scale.
        if args.encoder == "qwen3":
            embeddings = F.normalize(embeddings.float(), dim=-1)
        else:
            embeddings = embeddings.float()
        vectors.append(embeddings.cpu().numpy())
    embedding_array = np.concatenate(vectors).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        stimulus_ids=np.asarray(stimulus_ids, dtype=str),
        embeddings=embedding_array,
    )
    metadata = {
        "model": str(args.model_path.resolve()),
        "encoder": args.encoder,
        "pooling": ("attention-mask mean of encoder token states" if args.encoder == "flan_t5"
                    else "last non-padding token; left padding"),
        "normalization": "L2" if args.encoder == "qwen3" else "none",
        "max_length": args.max_length,
        "count": len(stimulus_ids),
        "dimension": int(embedding_array.shape[1]),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "text_scope": "All manifest stimuli; train-time CARG relations are restricted to each train batch",
        "instruction_prefix": args.instruction_prefix,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()


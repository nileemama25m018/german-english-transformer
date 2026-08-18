"""
Training, decoding, evaluation, and checkpoint utilities for Assignment 3.
"""

import math
import os
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Transformer, make_src_mask, make_tgt_mask

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() > 2:
            logits = logits.reshape(-1, logits.size(-1))
        target = target.reshape(-1)

        log_probs = F.log_softmax(logits, dim=-1)
        pad_mask = target.eq(self.pad_idx)

        with torch.no_grad():
            true_dist = torch.full_like(
                log_probs,
                self.smoothing / max(self.vocab_size - 2, 1),
            )
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0.0
            true_dist[pad_mask] = 0.0

        loss = F.kl_div(log_probs, true_dist, reduction="sum")
        normalizer = (~pad_mask).sum().clamp_min(1)
        return loss / normalizer


def _unpack_batch(batch):
    if isinstance(batch, dict):
        return batch["src"], batch["tgt"]
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1]
    raise ValueError("Expected batch to be a (src, tgt) pair or dict with 'src'/'tgt'.")


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    return_metrics: bool = False,
    step_callback=None,
    global_step_start: int = 0,
):
    pad_idx = getattr(loss_fn, "pad_idx", 1)
    total_loss = 0.0
    num_batches = 0
    num_correct_tokens = 0
    num_non_pad_tokens = 0
    total_prediction_confidence = 0.0
    global_step = global_step_start

    if is_train:
        model.train()
    else:
        model.eval()

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(data_iter, desc=f"{'train' if is_train else 'eval'}:{epoch_num}", leave=False):
            src, tgt = _unpack_batch(batch)
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx=pad_idx).to(device)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx=pad_idx).to(device)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            predictions = logits.argmax(dim=-1)
            non_pad_mask = tgt_output.ne(pad_idx)
            token_confidences = logits.softmax(dim=-1).gather(
                -1,
                tgt_output.unsqueeze(-1),
            ).squeeze(-1)
            num_correct_tokens += int(((predictions == tgt_output) & non_pad_mask).sum().item())
            num_non_pad_tokens += int(non_pad_mask.sum().item())
            total_prediction_confidence += float(token_confidences[non_pad_mask].sum().item())

            if is_train:
                if optimizer is None:
                    raise ValueError("optimizer must be provided during training")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                if step_callback is not None:
                    step_callback(
                        {
                            "global_step": global_step,
                            "loss": float(loss.item()),
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    )

            total_loss += float(loss.item())
            num_batches += 1

    if num_batches == 0:
        metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
            "prediction_confidence": 0.0,
            "global_step_end": global_step,
        }
        return metrics if return_metrics else 0.0

    avg_loss = total_loss / num_batches
    accuracy = num_correct_tokens / max(num_non_pad_tokens, 1)
    prediction_confidence = total_prediction_confidence / max(num_non_pad_tokens, 1)
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "prediction_confidence": prediction_confidence,
        "global_step_end": global_step,
    }
    return metrics if return_metrics else avg_loss


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    was_training = model.training
    model.eval()

    src = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.full((1, 1), start_symbol, dtype=src.dtype, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=-1).to(device)
            out = model.decode(memory, src_mask, ys, tgt_mask)
            next_word = torch.argmax(out[:, -1, :], dim=-1, keepdim=True)
            ys = torch.cat([ys, next_word], dim=1)

            if next_word.item() == end_symbol:
                break

    if was_training:
        model.train()
    return ys


def _lookup_token(vocab, idx: int) -> str:
    if hasattr(vocab, "lookup_token"):
        return vocab.lookup_token(idx)
    if hasattr(vocab, "itos"):
        return vocab.itos[idx]
    if hasattr(vocab, "idx_to_token"):
        mapping = vocab.idx_to_token
        if isinstance(mapping, dict):
            return mapping[idx]
        return mapping[idx]
    raise AttributeError("Vocabulary must provide lookup_token, itos, or idx_to_token.")


def _lookup_index(vocab, token: str, default: Optional[int] = None) -> Optional[int]:
    if hasattr(vocab, "lookup_index"):
        try:
            return vocab.lookup_index(token)
        except KeyError:
            return default
    if hasattr(vocab, "stoi"):
        return vocab.stoi.get(token, default)
    if hasattr(vocab, "token_to_idx"):
        return vocab.token_to_idx.get(token, default)
    return default


def _ids_to_tokens(sequence, vocab, pad_idx: Optional[int], start_idx: Optional[int], end_idx: Optional[int]):
    tokens = []
    for idx in sequence:
        idx = int(idx)
        if pad_idx is not None and idx == pad_idx:
            continue
        if start_idx is not None and idx == start_idx:
            continue
        if end_idx is not None and idx == end_idx:
            break
        tokens.append(_lookup_token(vocab, idx))
    return tokens


def _get_ngrams(tokens, order: int) -> Counter:
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[i:i + order]) for i in range(len(tokens) - order + 1))


def _closest_ref_length(pred_len: int, ref_lens) -> int:
    return min(ref_lens, key=lambda ref_len: (abs(ref_len - pred_len), ref_len))


def _corpus_bleu(predictions, references, max_order: int = 4) -> float:
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    pred_length = 0
    ref_length = 0

    for prediction, ref_group in zip(predictions, references):
        ref_group = [ref for ref in ref_group if ref is not None]
        pred_length += len(prediction)
        ref_length += _closest_ref_length(len(prediction), [len(ref) for ref in ref_group])

        for order in range(1, max_order + 1):
            pred_ngrams = _get_ngrams(prediction, order)
            max_ref_counts = Counter()
            for ref in ref_group:
                ref_ngrams = _get_ngrams(ref, order)
                for ngram, count in ref_ngrams.items():
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

            overlap = pred_ngrams & max_ref_counts
            matches_by_order[order - 1] += sum(overlap.values())
            possible_matches_by_order[order - 1] += max(len(prediction) - order + 1, 0)

    precisions = []
    for matches, possible in zip(matches_by_order, possible_matches_by_order):
        precisions.append(matches / possible if possible > 0 else 0.0)

    if min(precisions, default=0.0) > 0.0:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)
    else:
        geo_mean = 0.0

    if pred_length == 0:
        bp = 0.0
    elif pred_length > ref_length:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (ref_length / pred_length))

    return 100.0 * bp * geo_mean


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    start_symbol = _lookup_index(tgt_vocab, "<sos>", 2)
    end_symbol = _lookup_index(tgt_vocab, "<eos>", 3)
    pad_idx = _lookup_index(tgt_vocab, "<pad>", 1)

    predictions = []
    references = []

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="bleu", leave=False):
            src_batch, tgt_batch = _unpack_batch(batch)

            for src, tgt in zip(src_batch, tgt_batch):
                src = src.unsqueeze(0).to(device)
                src_mask = make_src_mask(src, pad_idx=pad_idx).to(device)

                pred_ids = greedy_decode(
                    model=model,
                    src=src,
                    src_mask=src_mask,
                    max_len=max_len,
                    start_symbol=start_symbol,
                    end_symbol=end_symbol,
                    device=device,
                ).squeeze(0).tolist()

                tgt_ids = tgt.tolist()
                predictions.append(
                    _ids_to_tokens(pred_ids, tgt_vocab, pad_idx, start_symbol, end_symbol)
                )
                references.append(
                    [_ids_to_tokens(tgt_ids, tgt_vocab, pad_idx, start_symbol, end_symbol)]
                )

    if was_training:
        model.train()

    return _corpus_bleu(predictions, references)


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": getattr(model, "model_config", None),
        "src_vocab_tokens": getattr(model, "src_vocab_tokens", None),
        "tgt_vocab_tokens": getattr(model, "tgt_vocab_tokens", None),
    }

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return int(checkpoint["epoch"])


def _get_encoder_qk_grad_norms(model: Transformer, layer_index: int = 0) -> dict:
    attention = model.encoder.layers[layer_index].self_attn
    q_grad = attention.W_q.weight.grad
    k_grad = attention.W_k.weight.grad
    return {
        "encoder_q_grad_norm": float(q_grad.norm().item()) if q_grad is not None else 0.0,
        "encoder_k_grad_norm": float(k_grad.norm().item()) if k_grad is not None else 0.0,
    }


def _tokenize_source_sentence(model: Transformer, sentence: str, device: str = "cpu"):
    tokens = [
        token.text.lower()
        for token in model.src_tokenizer(sentence.strip())
        if token.text.strip()
    ]
    display_tokens = ["<sos>", *tokens, "<eos>"]
    token_ids = [model.src_vocab.sos_idx]
    token_ids.extend(model.src_vocab.numericalize(tokens))
    token_ids.append(model.src_vocab.eos_idx)
    src = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_mask = make_src_mask(src, pad_idx=model.src_vocab.pad_idx).to(device)
    return display_tokens, src, src_mask


def collect_encoder_attention_for_sentence(
    model: Transformer,
    sentence: str,
    device: str = "cpu",
) -> dict:
    if model.src_vocab is None:
        raise RuntimeError("Model vocab is not initialized. Call model.register_vocabs(...) first.")

    was_training = model.training
    model.eval()

    display_tokens, src, src_mask = _tokenize_source_sentence(model, sentence, device=device)
    with torch.no_grad():
        model.encode(src, src_mask)
        attention = model.encoder.layers[-1].self_attn.attention_weights

    if was_training:
        model.train()

    return {
        "tokens": display_tokens,
        "attention": attention[0].detach().cpu(),
    }


def log_encoder_attention_heads_to_wandb(
    model: Transformer,
    sentence: str,
    device: str = "cpu",
    log_key: str = "encoder_last_layer_attention",
) -> dict:
    import matplotlib.pyplot as plt
    import wandb

    attention_info = collect_encoder_attention_for_sentence(model, sentence, device=device)
    tokens = attention_info["tokens"]
    attention = attention_info["attention"]

    images = {}
    for head_idx in range(attention.size(0)):
        fig, ax = plt.subplots(figsize=(8, 6))
        heatmap = ax.imshow(attention[head_idx].numpy(), cmap="viridis", aspect="auto")
        ax.set_title(f"Last Encoder Layer - Head {head_idx}")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticklabels(tokens)
        ax.set_xlabel("Key Tokens")
        ax.set_ylabel("Query Tokens")
        fig.colorbar(heatmap, ax=ax)
        fig.tight_layout()
        images[f"{log_key}/head_{head_idx}"] = wandb.Image(fig)
        plt.close(fig)

    return images


def run_training_experiment(config_overrides: Optional[dict] = None) -> dict:
    from dataset import Multi30kDataset
    from lr_scheduler import NoamScheduler

    try:
        import wandb
    except Exception:
        wandb = None

    config = {
        "project": "da6401-a3",
        "run_name": None,
        "batch_size": 64,
        "num_epochs": 15,
        "d_model": 256,
        "N": 3,
        "num_heads": 8,
        "d_ff": 512,
        "dropout": 0.1,
        "use_noam_scheduler": True,
        "fixed_lr": 1e-4,
        "use_attention_scaling": True,
        "positional_encoding_type": "sinusoidal",
        "positional_max_len": 5000,
        "warmup_steps": 4000,
        "smoothing": 0.1,
        "min_freq": 2,
        "max_vocab_size": None,
        "checkpoint_path": "checkpoint.pt",
        "best_checkpoint_path": "best_checkpoint.pt",
        "log_gradient_norms": False,
        "grad_log_steps": 1000,
        "grad_layer_index": 0,
        "compute_final_val_bleu": False,
        "log_attention_maps": False,
        "attention_sentence": "ein mann steht auf einer leiter und putzt ein fenster .",
        "attention_log_key": "encoder_last_layer_attention",
    }
    if config_overrides:
        config.update(config_overrides)

    run = None
    if wandb is not None:
        run = wandb.init(
            project=config["project"],
            name=config.get("run_name"),
            config=config,
        )
        config = dict(wandb.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = Multi30kDataset(
        split="train",
        min_freq=config["min_freq"],
        max_vocab_size=config["max_vocab_size"],
    )
    val_dataset = Multi30kDataset(
        split="validation",
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
    )
    test_dataset = Multi30kDataset(
        split="test",
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )

    model = Transformer(
        src_vocab_size=len(train_dataset.src_vocab),
        tgt_vocab_size=len(train_dataset.tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        min_freq=config["min_freq"],
        max_vocab_size=config["max_vocab_size"],
        max_len=100,
        auto_load_assets=False,
        positional_encoding_type=config["positional_encoding_type"],
        use_attention_scaling=config["use_attention_scaling"],
        positional_max_len=config["positional_max_len"],
    ).to(device)
    model.register_vocabs(train_dataset.src_vocab, train_dataset.tgt_vocab)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0 if config["use_noam_scheduler"] else config["fixed_lr"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = None
    if config["use_noam_scheduler"]:
        scheduler = NoamScheduler(
            optimizer,
            d_model=config["d_model"],
            warmup_steps=config["warmup_steps"],
        )
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_dataset.tgt_vocab),
        pad_idx=train_dataset.tgt_vocab.pad_idx,
        smoothing=config["smoothing"],
    )

    best_val_loss = float("inf")
    history = []
    global_step = 0

    def train_step_callback(step_info: dict) -> None:
        if run is None or not config["log_gradient_norms"]:
            return
        if step_info["global_step"] > config["grad_log_steps"]:
            return
        grad_metrics = _get_encoder_qk_grad_norms(
            model,
            layer_index=config["grad_layer_index"],
        )
        wandb.log(
            {
                "global_step": step_info["global_step"],
                "lr_step": step_info["lr"],
                **grad_metrics,
            }
        )

    for epoch in range(config["num_epochs"]):
        train_metrics = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
            return_metrics=True,
            step_callback=train_step_callback,
            global_step_start=global_step,
        )
        global_step = train_metrics["global_step_end"]
        val_metrics = run_epoch(
            val_loader,
            model,
            loss_fn,
            None,
            None,
            epoch_num=epoch,
            is_train=False,
            device=device,
            return_metrics=True,
        )
        train_loss = train_metrics["loss"]
        val_loss = val_metrics["loss"]
        val_accuracy = val_metrics["accuracy"]

        save_checkpoint(model, optimizer, scheduler, epoch, config["checkpoint_path"])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, config["best_checkpoint_path"])

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_prediction_confidence": train_metrics["prediction_confidence"],
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_prediction_confidence": val_metrics["prediction_confidence"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_metrics)

        if run is not None:
            wandb.log(epoch_metrics)

    if os.path.exists(config["best_checkpoint_path"]):
        load_checkpoint(config["best_checkpoint_path"], model, optimizer=None, scheduler=None)

    val_bleu = None
    if config["compute_final_val_bleu"]:
        val_bleu = evaluate_bleu(model, val_loader, train_dataset.tgt_vocab, device=device)
        print(f"Validation BLEU: {val_bleu:.2f}")

    bleu = evaluate_bleu(model, test_loader, train_dataset.tgt_vocab, device=device)
    print(f"Test BLEU: {bleu:.2f}")

    attention_log_payload = None
    if run is not None and config["log_attention_maps"]:
        attention_log_payload = log_encoder_attention_heads_to_wandb(
            model,
            sentence=config["attention_sentence"],
            device=device,
            log_key=config["attention_log_key"],
        )
        wandb.log(attention_log_payload)

    results = {
        "history": history,
        "test_bleu": bleu,
        "val_bleu": val_bleu,
        "best_val_loss": best_val_loss,
        "final_val_accuracy": history[-1]["val_accuracy"] if history else 0.0,
    }

    if run is not None:
        summary_payload = {"test_bleu": bleu}
        if val_bleu is not None:
            summary_payload["val_bleu"] = val_bleu
        wandb.log(summary_payload)
        wandb.finish()

    return results


if __name__ == "__main__":
    run_training_experiment()

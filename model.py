"""
model.py - Transformer components for DA6401 Assignment 3.
"""

import copy
import math
import os
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SerializableVocab:
    def __init__(self, tokens: Iterable[str]) -> None:
        self.itos = list(tokens)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.unk_idx = self.stoi["<unk>"]
        self.pad_idx = self.stoi["<pad>"]
        self.sos_idx = self.stoi["<sos>"]
        self.eos_idx = self.stoi["<eos>"]

    def __len__(self) -> int:
        return len(self.itos)

    def lookup_token(self, idx: int) -> str:
        return self.itos[idx]

    def lookup_index(self, token: str) -> int:
        return self.stoi.get(token, self.unk_idx)

    def numericalize(self, tokens: Iterable[str]) -> list[int]:
        return [self.lookup_index(token) for token in tokens]


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _scaled_dot_product_attention_impl(Q, K, V, mask, use_scaling=True)


def _scaled_dot_product_attention_impl(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_scaling: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Mask semantics:
        True  -> masked out
        False -> valid position
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if use_scaling:
        scores = scores / math.sqrt(d_k)

    if mask is not None:
        mask = mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    attn_weights = F.softmax(scores, dim=-1)

    if mask is not None:
        attn_weights = attn_weights.masked_fill(mask, 0.0)
        normalizer = attn_weights.sum(dim=-1, keepdim=True)
        attn_weights = torch.where(
            normalizer > 0,
            attn_weights / normalizer.clamp_min(1e-9),
            attn_weights,
        )

    output = torch.matmul(attn_weights, V)
    return output, attn_weights


def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """Build an encoder padding mask with shape [batch, 1, 1, src_len]."""
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """Build a decoder padding + causal mask with shape [batch, 1, tgt_len, tgt_len]."""
    batch_size, tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(1)
    return pad_mask | causal_mask.expand(batch_size, -1, -1, -1)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_scaling = use_scaling

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.attention_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        attn_output, attn_weights = _scaled_dot_product_attention_impl(
            Q,
            K,
            V,
            mask,
            use_scaling=self.use_scaling,
        )
        self.attention_weights = attn_weights

        attn_output = self._combine_heads(attn_output)
        attn_output = self.dropout(attn_output)
        return self.W_o(attn_output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len].to(dtype=x.dtype)
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds learned positional max_len={self.max_len}"
            )
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embeddings(positions)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Pre-LayerNorm encoder block.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_attention_scaling: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            num_heads,
            dropout,
            use_scaling=use_attention_scaling,
        )
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        x = x + self.dropout1(self.self_attn(attn_input, attn_input, attn_input, src_mask))

        ff_input = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(ff_input))
        return x


class DecoderLayer(nn.Module):
    """
    Pre-LayerNorm decoder block.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_attention_scaling: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            num_heads,
            dropout,
            use_scaling=use_attention_scaling,
        )
        self.cross_attn = MultiHeadAttention(
            d_model,
            num_heads,
            dropout,
            use_scaling=use_attention_scaling,
        )
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_input = self.norm1(x)
        x = x + self.dropout1(
            self.self_attn(self_attn_input, self_attn_input, self_attn_input, tgt_mask)
        )

        cross_attn_input = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(cross_attn_input, memory, memory, src_mask))

        ff_input = self.norm3(x)
        x = x + self.dropout3(self.feed_forward(ff_input))
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    _TOKENIZER_CACHE = {}
    _VOCAB_CACHE = {}
    _CHECKPOINT_CACHE = {}
    _DEFAULT_DRIVE_FILE_ID = "1MCaWxXTPz2HN6QrQy5uLXstnppns7afW"
    _DEFAULT_DRIVE_URL = ""

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        checkpoint_path: Optional[str] = None,
        min_freq: int = 2,
        max_vocab_size: Optional[int] = None,
        max_len: int = 100,
        auto_load_assets: bool = True,
        positional_encoding_type: str = "sinusoidal",
        use_attention_scaling: bool = True,
        positional_max_len: int = 5000,
    ) -> None:
        super().__init__()

        self.checkpoint_path = str(self._resolve_checkpoint_path(checkpoint_path))
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.src_vocab: Optional[_SerializableVocab] = None
        self.tgt_vocab: Optional[_SerializableVocab] = None
        self.src_vocab_tokens: Optional[list[str]] = None
        self.tgt_vocab_tokens: Optional[list[str]] = None

        checkpoint = None
        checkpoint_config = {}
        should_bootstrap_from_checkpoint = auto_load_assets and (
            src_vocab_size is None or tgt_vocab_size is None
        )
        if should_bootstrap_from_checkpoint:
            checkpoint = self._load_bootstrap_checkpoint(self.checkpoint_path)
            checkpoint_config = checkpoint.get("model_config") or {}

            src_vocab_size = checkpoint_config.get("src_vocab_size", src_vocab_size)
            tgt_vocab_size = checkpoint_config.get("tgt_vocab_size", tgt_vocab_size)
            d_model = checkpoint_config.get("d_model", d_model)
            N = checkpoint_config.get("N", N)
            num_heads = checkpoint_config.get("num_heads", num_heads)
            d_ff = checkpoint_config.get("d_ff", d_ff)
            dropout = checkpoint_config.get("dropout", dropout)
            min_freq = checkpoint_config.get("min_freq", min_freq)
            max_vocab_size = checkpoint_config.get("max_vocab_size", max_vocab_size)
            max_len = checkpoint_config.get("max_len", max_len)
            positional_encoding_type = checkpoint_config.get(
                "positional_encoding_type",
                positional_encoding_type,
            )
            use_attention_scaling = checkpoint_config.get(
                "use_attention_scaling",
                use_attention_scaling,
            )
            positional_max_len = checkpoint_config.get(
                "positional_max_len",
                positional_max_len,
            )

        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len

        self.src_tokenizer, self.tgt_tokenizer = self._load_tokenizers()
        should_load_vocab_assets = auto_load_assets and (
            checkpoint is not None or src_vocab_size is None or tgt_vocab_size is None
        )
        if should_load_vocab_assets:
            self._load_vocab_assets(checkpoint)
            if src_vocab_size is None and self.src_vocab is not None:
                src_vocab_size = len(self.src_vocab)
            if tgt_vocab_size is None and self.tgt_vocab is not None:
                tgt_vocab_size = len(self.tgt_vocab)

        if src_vocab_size is None or tgt_vocab_size is None:
            raise ValueError(
                "src_vocab_size and tgt_vocab_size must be provided unless auto-loading "
                "from a checkpoint is enabled and successful."
            )

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.src_pos_encoding = self._build_positional_encoding(
            positional_encoding_type,
            d_model,
            dropout,
            positional_max_len,
        )
        self.tgt_pos_encoding = self._build_positional_encoding(
            positional_encoding_type,
            d_model,
            dropout,
            positional_max_len,
        )

        encoder_layer = EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout,
            use_attention_scaling=use_attention_scaling,
        )
        decoder_layer = DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout,
            use_attention_scaling=use_attention_scaling,
        )
        self.encoder = Encoder(encoder_layer, N)
        self.decoder = Decoder(decoder_layer, N)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.positional_encoding_type = positional_encoding_type
        self.use_attention_scaling = use_attention_scaling
        self.positional_max_len = positional_max_len

        self._reset_parameters()

        if checkpoint is not None:
            self._load_state_from_checkpoint(checkpoint)

        self.model_config = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model": d_model,
            "N": N,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "dropout": dropout,
            "checkpoint_path": None,
            "min_freq": min_freq,
            "max_vocab_size": max_vocab_size,
            "max_len": max_len,
            "auto_load_assets": False,
            "positional_encoding_type": positional_encoding_type,
            "use_attention_scaling": use_attention_scaling,
            "positional_max_len": positional_max_len,
        }

    @staticmethod
    def _build_positional_encoding(
        positional_encoding_type: str,
        d_model: int,
        dropout: float,
        positional_max_len: int,
    ) -> nn.Module:
        if positional_encoding_type == "sinusoidal":
            return PositionalEncoding(d_model, dropout, max_len=positional_max_len)
        if positional_encoding_type == "learned":
            return LearnedPositionalEncoding(d_model, dropout, max_len=positional_max_len)
        raise ValueError(
            "positional_encoding_type must be either 'sinusoidal' or 'learned'"
        )

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path: Optional[str]) -> Path:
        if checkpoint_path is not None:
            return Path(checkpoint_path)

        env_path = os.environ.get("DA6401_TRANSFORMER_CHECKPOINT_PATH") or os.environ.get(
            "TRANSFORMER_CHECKPOINT_PATH"
        )
        if env_path:
            return Path(env_path)

        return Path(__file__).resolve().parent / ".model_cache" / "transformer_checkpoint.pt"

    @classmethod
    def _get_drive_source(cls) -> str:
        return (
            os.environ.get("DA6401_TRANSFORMER_CHECKPOINT_URL")
            or os.environ.get("TRANSFORMER_CHECKPOINT_URL")
            or os.environ.get("DA6401_TRANSFORMER_CHECKPOINT_FILE_ID")
            or os.environ.get("TRANSFORMER_CHECKPOINT_FILE_ID")
            or cls._DEFAULT_DRIVE_URL
            or cls._DEFAULT_DRIVE_FILE_ID
        )

    @staticmethod
    def _extract_drive_file_id(drive_source: str) -> str:
        if not drive_source:
            return ""

        if drive_source.startswith("http://") or drive_source.startswith("https://"):
            patterns = [
                r"/d/([a-zA-Z0-9_-]+)",
                r"id=([a-zA-Z0-9_-]+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, drive_source)
                if match:
                    return match.group(1)
            raise RuntimeError(f"Could not extract Google Drive file ID from URL: {drive_source}")

        return drive_source

    @staticmethod
    def _looks_like_checkpoint_file(path: Path) -> bool:
        if not path.exists() or path.stat().st_size == 0:
            return False

        with open(path, "rb") as file_obj:
            header = file_obj.read(8)

        return header.startswith(b"PK\x03\x04") or header.startswith(b"\x80")

    @classmethod
    def _download_from_url(cls, url: str, output_path: Path) -> bool:
        temp_path = output_path.with_suffix(output_path.suffix + ".download")
        if temp_path.exists():
            temp_path.unlink()

        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request) as response, open(temp_path, "wb") as file_obj:
                shutil.copyfileobj(response, file_obj)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            return False

        if cls._looks_like_checkpoint_file(temp_path):
            temp_path.replace(output_path)
            return True

        if temp_path.exists():
            temp_path.unlink()
        return False

    @classmethod
    def _load_tokenizers(cls):
        if "de" not in cls._TOKENIZER_CACHE or "en" not in cls._TOKENIZER_CACHE:
            import spacy

            cls._TOKENIZER_CACHE["de"] = spacy.blank("de").tokenizer
            cls._TOKENIZER_CACHE["en"] = spacy.blank("en").tokenizer

        return cls._TOKENIZER_CACHE["de"], cls._TOKENIZER_CACHE["en"]

    @staticmethod
    def _extract_vocab_tokens(vocab_or_tokens) -> list[str]:
        if vocab_or_tokens is None:
            raise ValueError("vocab_or_tokens cannot be None")
        if isinstance(vocab_or_tokens, (list, tuple)):
            return list(vocab_or_tokens)
        if hasattr(vocab_or_tokens, "itos"):
            return list(vocab_or_tokens.itos)
        if hasattr(vocab_or_tokens, "idx_to_token"):
            idx_to_token = vocab_or_tokens.idx_to_token
            return list(idx_to_token.values()) if isinstance(idx_to_token, dict) else list(idx_to_token)
        raise TypeError("Vocabulary object must provide 'itos' or 'idx_to_token'.")

    def register_vocabs(self, src_vocab, tgt_vocab) -> None:
        self.src_vocab_tokens = self._extract_vocab_tokens(src_vocab)
        self.tgt_vocab_tokens = self._extract_vocab_tokens(tgt_vocab)
        self.src_vocab = _SerializableVocab(self.src_vocab_tokens)
        self.tgt_vocab = _SerializableVocab(self.tgt_vocab_tokens)

    def _load_vocab_assets(self, checkpoint: Optional[dict]) -> None:
        src_tokens = checkpoint.get("src_vocab_tokens") if checkpoint is not None else None
        tgt_tokens = checkpoint.get("tgt_vocab_tokens") if checkpoint is not None else None

        if src_tokens is not None and tgt_tokens is not None:
            self.register_vocabs(src_tokens, tgt_tokens)
            return

        cache_key = (self.min_freq, self.max_vocab_size)
        cached_tokens = self._VOCAB_CACHE.get(cache_key)
        if cached_tokens is None:
            from dataset import Multi30kDataset

            train_dataset = Multi30kDataset(
                split="train",
                min_freq=self.min_freq,
                max_vocab_size=self.max_vocab_size,
            )
            cached_tokens = (
                list(train_dataset.src_vocab.itos),
                list(train_dataset.tgt_vocab.itos),
            )
            self._VOCAB_CACHE[cache_key] = cached_tokens

        self.register_vocabs(cached_tokens[0], cached_tokens[1])

    @classmethod
    def _ensure_checkpoint_available(cls, checkpoint_path: Path) -> Path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.exists() and cls._looks_like_checkpoint_file(checkpoint_path):
            return checkpoint_path
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        drive_source = cls._get_drive_source()
        if not drive_source:
            raise RuntimeError(
                "Checkpoint file is not available locally and no Google Drive source was configured. "
                "Set TRANSFORMER_CHECKPOINT_FILE_ID or TRANSFORMER_CHECKPOINT_URL, or hardcode "
                "Transformer._DEFAULT_DRIVE_FILE_ID before submission."
            )

        file_id = cls._extract_drive_file_id(drive_source)
        quoted_id = urllib.parse.quote(file_id, safe="")
        direct_urls = [
            f"https://drive.usercontent.google.com/download?id={quoted_id}&export=download&confirm=t",
            f"https://drive.google.com/uc?export=download&id={quoted_id}&confirm=t",
            f"https://drive.google.com/uc?export=download&id={quoted_id}",
        ]

        for url in direct_urls:
            if cls._download_from_url(url, checkpoint_path):
                break
        else:
            try:
                import gdown
            except Exception as exc:
                raise RuntimeError(
                    "Failed to download a valid checkpoint from Google Drive using direct URLs, "
                    "and gdown is not available as a fallback."
                ) from exc

            gdown.download(id=file_id, output=str(checkpoint_path), quiet=True)

        if not cls._looks_like_checkpoint_file(checkpoint_path):
            if checkpoint_path.exists():
                with open(checkpoint_path, "rb") as file_obj:
                    preview = file_obj.read(32)
                checkpoint_path.unlink()
            else:
                preview = b""
            raise RuntimeError(
                "Checkpoint download did not produce a valid PyTorch checkpoint file. "
                f"First bytes: {preview!r}"
            )

        return checkpoint_path

    @classmethod
    def _load_bootstrap_checkpoint(cls, checkpoint_path: str) -> dict:
        resolved_path = cls._ensure_checkpoint_available(Path(checkpoint_path))
        cache_key = str(resolved_path.resolve())
        if cache_key not in cls._CHECKPOINT_CACHE:
            cls._CHECKPOINT_CACHE[cache_key] = torch.load(
                resolved_path,
                map_location="cpu",
                weights_only=False,
            )
        return cls._CHECKPOINT_CACHE[cache_key]

    def _load_state_from_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.load_state_dict(state_dict)

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def _greedy_decode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        if self.tgt_vocab is None:
            raise RuntimeError("Target vocabulary is not initialized.")

        memory = self.encode(src, src_mask)
        ys = torch.full(
            (1, 1),
            self.tgt_vocab.sos_idx,
            dtype=src.dtype,
            device=src.device,
        )

        for _ in range(self.max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=-1).to(src.device)
            logits = self.decode(memory, src_mask, ys, tgt_mask)
            next_word = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ys = torch.cat([ys, next_word], dim=1)

            if next_word.item() == self.tgt_vocab.eos_idx:
                break

        return ys

    @staticmethod
    def _detokenize(tokens: list[str]) -> str:
        text = " ".join(tokens).strip()
        text = re.sub(r"\s+([?.!,;:%)\]\}])", r"\1", text)
        text = re.sub(r"([(\[\{])\s+", r"\1", text)
        text = text.replace(" n't", "n't")
        text = text.replace(" 'm", "'m")
        text = text.replace(" 're", "'re")
        text = text.replace(" 's", "'s")
        text = text.replace(" 've", "'ve")
        text = text.replace(" 'd", "'d")
        text = text.replace(" 'll", "'ll")
        return text

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_embeddings = self.src_embed(src) * math.sqrt(self.d_model)
        src_embeddings = self.src_pos_encoding(src_embeddings)
        return self.encoder(src_embeddings, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_embeddings = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_embeddings = self.tgt_pos_encoding(tgt_embeddings)
        decoded = self.decoder(tgt_embeddings, memory, src_mask, tgt_mask)
        return self.generator(decoded)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        Translate a single German sentence to English using greedy decoding.
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            self._load_vocab_assets(None)

        was_training = self.training
        self.eval()

        device = next(self.parameters()).device
        tokens = [
            token.text.lower()
            for token in self.src_tokenizer(src_sentence.strip())
            if token.text.strip()
        ]

        src_ids = [self.src_vocab.sos_idx]
        src_ids.extend(self.src_vocab.numericalize(tokens))
        src_ids.append(self.src_vocab.eos_idx)

        src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src, pad_idx=self.src_vocab.pad_idx).to(device)

        with torch.no_grad():
            predicted_ids = self._greedy_decode(src, src_mask).squeeze(0).tolist()

        predicted_tokens = []
        for idx in predicted_ids:
            if idx == self.tgt_vocab.sos_idx or idx == self.tgt_vocab.pad_idx:
                continue
            if idx == self.tgt_vocab.eos_idx:
                break
            predicted_tokens.append(self.tgt_vocab.lookup_token(int(idx)))

        if was_training:
            self.train()

        return self._detokenize(predicted_tokens)

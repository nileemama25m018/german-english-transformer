"""
Dataset and vocabulary utilities for Multi30k German-English translation.
"""

from collections import Counter
from typing import Iterable, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

import spacy
from datasets import load_dataset


SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]


class Vocabulary:
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

    def numericalize(self, tokens) -> list[int]:
        return [self.lookup_index(token) for token in tokens]


def collate_translation_batch(batch, src_pad_idx: int = 1, tgt_pad_idx: int = 1):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_idx)
    return src_batch, tgt_batch


class Multi30kDataset:
    _tokenizers = {}
    _vocab_cache = {}

    def __init__(
        self,
        split: str = "train",
        min_freq: int = 2,
        max_vocab_size: Optional[int] = None,
        src_vocab: Optional[Vocabulary] = None,
        tgt_vocab: Optional[Vocabulary] = None,
    ):
        self.split = self._normalize_split(split)
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        self.src_tokenizer = self._get_tokenizer("de")
        self.tgt_tokenizer = self._get_tokenizer("en")
        self.raw_data = self._load_split(self.split)

        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        self.data = self.process_data()

    @staticmethod
    def _normalize_split(split: str) -> str:
        mapping = {
            "val": "validation",
            "valid": "validation",
            "dev": "validation",
        }
        return mapping.get(split, split)

    @classmethod
    def _get_tokenizer(cls, language: str):
        if language not in cls._tokenizers:
            cls._tokenizers[language] = spacy.blank(language).tokenizer
        return cls._tokenizers[language]

    @staticmethod
    def _load_split(split: str):
        try:
            return load_dataset("bentrevett/multi30k", split=split)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Multi30k from Hugging Face. "
                "Make sure the 'datasets' package is installed and the dataset is accessible."
            ) from exc

    @staticmethod
    def _extract_pair(example):
        if "translation" in example:
            translation = example["translation"]
            return translation["de"], translation["en"]
        if "de" in example and "en" in example:
            return example["de"], example["en"]
        raise KeyError("Could not find German/English text fields in dataset example.")

    def _tokenize_de(self, text: str) -> list[str]:
        return [token.text.lower() for token in self.src_tokenizer(text.strip()) if token.text.strip()]

    def _tokenize_en(self, text: str) -> list[str]:
        return [token.text.lower() for token in self.tgt_tokenizer(text.strip()) if token.text.strip()]

    def _build_single_vocab(self, token_sequences) -> Vocabulary:
        counter = Counter()
        for tokens in token_sequences:
            counter.update(tokens)

        sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        if self.max_vocab_size is not None:
            sorted_tokens = sorted_tokens[: max(0, self.max_vocab_size - len(SPECIAL_TOKENS))]

        vocab_tokens = list(SPECIAL_TOKENS)
        for token, freq in sorted_tokens:
            if freq >= self.min_freq and token not in vocab_tokens:
                vocab_tokens.append(token)

        return Vocabulary(vocab_tokens)

    def build_vocab(self):
        cache_key = (self.min_freq, self.max_vocab_size)
        if cache_key in self._vocab_cache:
            return self._vocab_cache[cache_key]

        train_split = self._load_split("train")
        src_token_sequences = []
        tgt_token_sequences = []

        for example in train_split:
            src_text, tgt_text = self._extract_pair(example)
            src_token_sequences.append(self._tokenize_de(src_text))
            tgt_token_sequences.append(self._tokenize_en(tgt_text))

        src_vocab = self._build_single_vocab(src_token_sequences)
        tgt_vocab = self._build_single_vocab(tgt_token_sequences)
        self._vocab_cache[cache_key] = (src_vocab, tgt_vocab)
        return src_vocab, tgt_vocab

    def process_data(self):
        processed = []
        for example in self.raw_data:
            src_text, tgt_text = self._extract_pair(example)
            src_tokens = self._tokenize_de(src_text)
            tgt_tokens = self._tokenize_en(tgt_text)

            src_ids = [self.src_vocab.sos_idx]
            src_ids.extend(self.src_vocab.numericalize(src_tokens))
            src_ids.append(self.src_vocab.eos_idx)

            tgt_ids = [self.tgt_vocab.sos_idx]
            tgt_ids.extend(self.tgt_vocab.numericalize(tgt_tokens))
            tgt_ids.append(self.tgt_vocab.eos_idx)

            processed.append(
                (
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                )
            )

        return processed

    def collate_fn(self, batch):
        return collate_translation_batch(
            batch,
            src_pad_idx=self.src_vocab.pad_idx,
            tgt_pad_idx=self.tgt_vocab.pad_idx,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

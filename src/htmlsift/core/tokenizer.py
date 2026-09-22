"""Adapter: a `tokenizers.Tokenizer` presented with the transformers fast-tokenizer
call shape that `prep_page` expects, so the default install tokenizes with the small
`tokenizers` library and needs no `transformers` or `torch`.

`prep_page` calls the tokenizer as
`tok(text, add_special_tokens=True, return_offsets_mapping=True,
 return_special_tokens_mask=True, ...)` and reads `input_ids`, `offset_mapping` and
`special_tokens_mask` off the result. The `tokenizers` library carries the same
information on its `Encoding` object under different names, so this wraps the two up.
"""
from tokenizers import Tokenizer


class FastTokenizer:
    """A `tokenizers.Tokenizer` callable like a transformers fast tokenizer:
    `tok(text)` -> `{"input_ids", "offset_mapping", "special_tokens_mask"}`.

    Truncation and padding are disabled so a long page reaches the encoder windowing
    intact -- `prep_page` windows the sequence, the tokenizer must never cut it. The
    special-token post-processor and normalizer baked into `tokenizer.json` are
    applied by `encode`, so the ids, offsets and mask match what the model was
    trained on."""

    def __init__(self, tk):
        tk.no_truncation()
        tk.no_padding()
        self._tk = tk

    @classmethod
    def from_file(cls, path):
        """Load `tokenizer.json` at `path` -- the single self-contained tokenizer file."""
        return cls(Tokenizer.from_file(str(path)))

    def __call__(self, text, add_special_tokens=True, **_ignored):
        enc = self._tk.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": enc.ids,
                "offset_mapping": enc.offsets,
                "special_tokens_mask": enc.special_tokens_mask}

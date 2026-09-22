"""prep_page unit tests (no bundle): span/window/cap math and token->block membership.
page_members is exercised with a hand-built encoding so no real tokenizer is needed."""
import numpy as np

from htmlsift.core.prep_page import apply_cap, block_spans, page_members, window_starts


def test_block_spans():
    assert block_spans(["ab", "cde"]) == [(0, 2), (3, 6)]   # +1 char per joining newline


def test_window_starts_tiling_and_snap():
    assert window_starts(100, 100) == [0]
    assert window_starts(150, 100) == [0, 50]               # 50% stride
    assert window_starts(180, 100) == [0, 50, 80]           # last window snapped to n-window


def test_apply_cap_drops_and_renumbers():
    ids = np.arange(6, dtype=np.int64)
    members = [[0, 1, 2], [3, 4], [5]]                       # block 0 exceeds cap 2
    new_ids, new_members = apply_cap(ids, members, 2)
    assert list(new_ids) == [0, 1, 3, 4, 5]                  # token 2 dropped from the sequence
    assert new_members == [[0, 1], [2, 3], [4]]              # members renumbered onto compacted ids


def test_page_members_maps_tokens_to_blocks():
    # "ab\ncd": chars 0-1 ab, 2 newline, 3-4 cd; enc = CLS a b c d SEP
    enc = {"offset_mapping": [(0, 0), (0, 1), (1, 2), (3, 4), (4, 5), (0, 0)],
           "special_tokens_mask": [1, 0, 0, 0, 0, 1]}
    assert page_members(enc, ["ab", "cd"]) == [[1, 2], [3, 4]]


def test_page_members_tokenless_block_is_empty():
    enc = {"offset_mapping": [(0, 1)], "special_tokens_mask": [0]}
    assert page_members(enc, ["a", ""]) == [[0], []]         # empty block -> [] (pools to zeros)

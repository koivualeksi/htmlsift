"""Torch-free onnxruntime forward for the mini model (int8 table + BiGRU head).

Embed the page's distinct token ids through the int8 table graph, mean-pool by block
membership, concat any structural features, run the head graph, sigmoid. numpy +
onnxruntime only -- no torch. The table is a per-id lookup, so embedding each distinct id
once gives the same rows as embedding every token, and memory scales with the page's
vocabulary instead of its length.
"""
import numpy as np
import onnxruntime as ort

# onnxruntime's arena keeps its peak after a run. Pages estimated above SHRINK_MB give
# the memory back when they finish; smaller pages keep the arena for speed.
TABLE_KB_PER_ID = 12
HEAD_KB_PER_BLOCK = 21
SHRINK_MB = 100


def arena_mb(n_unique, n_blocks):
    """Estimated arena memory (MB) one page needs: table per distinct id + head per block."""
    return (TABLE_KB_PER_ID * n_unique + HEAD_KB_PER_BLOCK * n_blocks) / 1024


def pool(emb_u, inv, members):
    """Per-block mean of the member tokens' embeddings -> [n, H]. emb_u holds one row
    per distinct id and emb_u[inv[t]] is token t's. A token-less block pools to zeros."""
    pooled = np.zeros((len(members), emb_u.shape[1]), dtype=np.float32)
    for i, mem in enumerate(members):
        if mem:
            pooled[i] = emb_u[inv[mem]].mean(0)
    return pooled


def load_mini(table_path, head_path, threads=None):
    """The two mini graphs -> infer_fn(page) -> per-block probs (np.float32): the
    (page -> probs) contract the extractor consumes."""
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads, opts.inter_op_num_threads = threads, 1
    table = ort.InferenceSession(str(table_path), opts, providers=["CPUExecutionProvider"])
    head = ort.InferenceSession(str(head_path), opts, providers=["CPUExecutionProvider"])
    shrink = ort.RunOptions()
    shrink.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")

    def infer_fn(page):
        uniq, inv = np.unique(page["ids"], return_inverse=True)
        members = page["members"]
        ro = shrink if arena_mb(len(uniq), len(members)) > SHRINK_MB else None
        emb_u = table.run(["out"], {"ids": uniq.astype(np.int64)}, ro)[0]      # [U, H]
        x = pool(emb_u, inv, members)                                          # [n, H]
        if page.get("feats") is not None:
            x = np.concatenate([x, page["feats"].astype(np.float32)], 1)       # [n, H+K]
        logits = head.run(["logits"], {"x": x[None]}, ro)[0][0]                # [n]
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    return infer_fn

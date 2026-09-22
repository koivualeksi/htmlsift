"""Torch-free onnxruntime forward for the mini model (int8 table + BiGRU head).

Embed the token ids through the int8 table graph, mean-pool by block membership,
concat any structural features, run the head graph, sigmoid. numpy + onnxruntime only
-- no torch. Lifted unchanged from the research export so it stays bit-identical to the
graphs' parity-gated reference forward.
"""
import numpy as np
import onnxruntime as ort


def pool(emb, members):
    """Token embeddings emb [T, H] -> per-block mean vectors [n, H]. A token-less block
    pools to zeros (sum of nothing / 1)."""
    pooled = np.zeros((len(members), emb.shape[1]), dtype=np.float32)
    for i, mem in enumerate(members):
        if mem:
            pooled[i] = emb[mem].mean(0)
    return pooled


def load_mini(table_path, head_path, threads=None):
    """The two mini graphs -> infer_fn(page) -> per-block probs (np.float32): the
    (page -> probs) contract the extractor consumes."""
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads, opts.inter_op_num_threads = threads, 1
    table = ort.InferenceSession(str(table_path), opts, providers=["CPUExecutionProvider"])
    head = ort.InferenceSession(str(head_path), opts, providers=["CPUExecutionProvider"])

    def infer_fn(page):
        emb = table.run(["out"], {"ids": page["ids"].astype(np.int64)})[0]     # [T, H]
        x = pool(emb, page["members"])                                         # [n, H]
        if page.get("feats") is not None:
            x = np.concatenate([x, page["feats"].astype(np.float32)], 1)       # [n, H+K]
        logits = head.run(["logits"], {"x": x[None]})[0][0]                    # [n]
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    return infer_fn

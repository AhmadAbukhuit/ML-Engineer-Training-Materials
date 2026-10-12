"""Ring All-Reduce Communication Primitive from Scratch in NumPy."""

import numpy as np


def ring_allreduce(tensors: list[np.ndarray]) -> list[np.ndarray]:
    """Simulate Ring All-Reduce communication across P distributed ranks.

    In a physical cluster, each rank runs on a distinct GPU/machine connected
    in a logical ring topology: rank r sends to (r + 1) % P and receives from (r - 1) % P.

    Phase 1: Scatter-Reduce (P - 1 steps)
        At the end of step P - 1, rank r holds the complete elementwise sum of chunk r.
    Phase 2: All-Gather (P - 1 steps)
        At the end of step P - 1, all ranks hold the complete reduced array.

    Total data transferred per rank: 2 * ((P - 1) / P) * N elements,
    which is asymptotically independent of the number of ranks P!

    Parameters
    ----------
    tensors : list[np.ndarray]
        List of 1D NumPy arrays of identical length, one per simulated rank.

    Returns
    -------
    reduced_tensors : list[np.ndarray]
        List of arrays where every rank now contains the exact elementwise sum.
    """
    P = len(tensors)
    if P <= 1:
        return [t.copy() for t in tensors]

    N = len(tensors[0])
    for t in tensors:
        if len(t) != N:
            raise ValueError("All tensors across ranks must have identical length.")

    # Split arrays into P chunks (pad if necessary)
    chunk_size = int(np.ceil(N / P))
    padded_len = chunk_size * P
    pad_len = padded_len - N

    # Clone and pad
    buffers = [
        np.pad(t.astype(np.float64), (0, pad_len), mode="constant")
        for t in tensors
    ]

    # Reshape each buffer into P chunks of shape (chunk_size,)
    chunks = [np.split(buf, P) for buf in buffers]

    # --- Phase 1: Scatter-Reduce (P - 1 communication steps) ---
    for step in range(P - 1):
        # In step k, rank r sends chunk (r - step) % P to (r + 1) % P
        # and receives chunk (r - step - 1) % P from (r - 1) % P
        send_data = [None] * P
        for r in range(P):
            send_chunk_idx = (r - step) % P
            send_data[r] = chunks[r][send_chunk_idx].copy()

        for r in range(P):
            recv_chunk_idx = (r - step - 1) % P
            prev_rank = (r - 1) % P
            # Add received chunk to local chunk
            chunks[r][recv_chunk_idx] += send_data[prev_rank]

    # --- Phase 2: All-Gather (P - 1 communication steps) ---
    for step in range(P - 1):
        # In step k, rank r sends chunk (r - step + 1) % P to (r + 1) % P
        # and receives chunk (r - step) % P from (r - 1) % P
        send_data = [None] * P
        for r in range(P):
            send_chunk_idx = (r - step + 1) % P
            send_data[r] = chunks[r][send_chunk_idx].copy()

        for r in range(P):
            recv_chunk_idx = (r - step) % P
            prev_rank = (r - 1) % P
            # Overwrite local chunk with fully-reduced data from neighbor
            chunks[r][recv_chunk_idx] = send_data[prev_rank].copy()

    # Reconstruct original array size without padding
    result = []
    for r in range(P):
        full_buf = np.concatenate(chunks[r])
        result.append(full_buf[:N])

    return result

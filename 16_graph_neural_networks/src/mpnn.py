"""Message Passing Neural Network (MPNN) Framework (Gilmer et al., 2017)."""

from typing import Literal
import numpy as np


class MessagePassingLayer:
    """General Message Passing Layer.

    m_ij = W_msg @ [h_i || h_j]
    m_i  = Aggregate_{j in N(i)}(m_ij)
    h_i' = ReLU( W_upd @ [h_i || m_i] )

    Parameters
    ----------
    node_dim : int
        Dimension of node features.
    msg_dim : int
        Dimension of intermediate message vectors.
    out_dim : int
        Dimension of updated node representations.
    aggr : {'sum', 'mean', 'max'}
        Permutation-invariant aggregation operator.
    """

    def __init__(
        self,
        node_dim: int,
        msg_dim: int,
        out_dim: int,
        aggr: Literal["sum", "mean", "max"] = "sum",
        random_state: int | None = 42,
    ) -> None:
        self.node_dim = node_dim
        self.msg_dim = msg_dim
        self.out_dim = out_dim
        self.aggr = aggr

        rng = np.random.default_rng(random_state)
        # Message weights: takes concatenated [h_i || h_j] -> msg_dim
        self.W_msg = rng.normal(0, np.sqrt(2.0 / (2 * node_dim)), (2 * node_dim, msg_dim))
        # Update weights: takes concatenated [h_i || m_i] -> out_dim
        self.W_upd = rng.normal(0, np.sqrt(2.0 / (node_dim + msg_dim)), (node_dim + msg_dim, out_dim))

    def forward(self, H: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """Propagate messages across graph edges and update node states.

        Parameters
        ----------
        H : np.ndarray of shape (N, node_dim)
        adj : np.ndarray of shape (N, N)
            Adjacency matrix where adj[i, j] > 0 denotes an edge from j to i.
        """
        N = H.shape[0]
        # Messages from j to i: m_ij
        # Compute pair combinations for active edges
        edges_i, edges_j = np.where(adj > 0)
        num_edges = len(edges_i)

        if num_edges == 0:
            # No edges: message is zero
            m_agg = np.zeros((N, self.msg_dim))
        else:
            # Concat h_dst and h_src
            pairs = np.hstack([H[edges_i], H[edges_j]])  # (E, 2 * node_dim)
            messages = pairs @ self.W_msg  # (E, msg_dim)

            # Aggregate messages into target nodes edges_i
            m_agg = np.zeros((N, self.msg_dim))
            if self.aggr == "sum":
                np.add.at(m_agg, edges_i, messages)
            elif self.aggr == "mean":
                deg = np.zeros(N)
                np.add.at(deg, edges_i, 1.0)
                np.add.at(m_agg, edges_i, messages)
                deg = np.maximum(deg[:, np.newaxis], 1.0)
                m_agg = m_agg / deg
            elif self.aggr == "max":
                m_agg.fill(-np.inf)
                for idx in range(num_edges):
                    dst = edges_i[idx]
                    m_agg[dst] = np.maximum(m_agg[dst], messages[idx])
                # If node has no edges, reset to 0
                m_agg[np.isneginf(m_agg)] = 0.0

        # Update step: [H || m_agg] @ W_upd
        h_concat = np.hstack([H, m_agg])
        h_next = np.maximum(h_concat @ self.W_upd, 0.0)
        return h_next

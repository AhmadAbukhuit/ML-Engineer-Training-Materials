import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dpo_loss import DPOLoss


def test_dpo_loss_preference_alignment():
    dpo = DPOLoss(beta=0.1)

    # Synthetic batch: chosen response has higher policy log-probability than rejected
    policy_chosen = np.array([-1.2, -0.8, -1.5])
    policy_rejected = np.array([-2.5, -2.1, -2.8])

    # Reference model evaluated equal baseline probabilities
    ref_chosen = np.array([-2.0, -2.0, -2.0])
    ref_rejected = np.array([-2.0, -2.0, -2.0])

    metrics = dpo.compute_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )

    assert metrics["loss"] > 0.0
    assert metrics["reward_margin"] > 0.0
    assert metrics["accuracy"] == 1.0
    assert metrics["chosen_rewards"] > metrics["rejected_rewards"]

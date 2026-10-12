"""ZeRO (Zero Redundancy Optimizer) Memory Accounting & Analysis."""

from dataclasses import dataclass


@dataclass
class MemoryProfile:
    parameters_gb: float
    gradients_gb: float
    optimizer_states_gb: float
    total_static_gb: float


class ZeROMemoryEstimator:
    """Estimates Per-GPU Static Memory Breakdown across ZeRO Partitioning Stages.

    Reference: Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training
    Trillion Parameter Models", SC 2020.

    Mixed Precision Adam Memory Assumptions:
    - FP16/BF16 Parameters: 2 bytes per param
    - FP16/BF16 Gradients: 2 bytes per param
    - FP32 Master Parameters: 4 bytes per param
    - FP32 First Moment (m): 4 bytes per param
    - FP32 Second Moment (v): 4 bytes per param
    Total static footprint under baseline DDP: 16 bytes per parameter.

    Parameters
    ----------
    num_parameters : int
        Total number of trainable model parameters Psi.
    world_size : int
        Number of distributed GPUs / ranks (N_d).
    """

    def __init__(self, num_parameters: int, world_size: int = 8) -> None:
        if num_parameters <= 0:
            raise ValueError("num_parameters must be positive.")
        if world_size <= 0:
            raise ValueError("world_size must be positive.")

        self.num_parameters = num_parameters
        self.world_size = world_size

    def _to_gb(self, bytes_val: float) -> float:
        return bytes_val / (1024**3)

    def baseline_ddp(self) -> MemoryProfile:
        """Standard Distributed Data Parallel (no parameter/state partitioning)."""
        param_bytes = 2.0 * self.num_parameters
        grad_bytes = 2.0 * self.num_parameters
        opt_bytes = 12.0 * self.num_parameters
        total = param_bytes + grad_bytes + opt_bytes

        return MemoryProfile(
            parameters_gb=self._to_gb(param_bytes),
            gradients_gb=self._to_gb(grad_bytes),
            optimizer_states_gb=self._to_gb(opt_bytes),
            total_static_gb=self._to_gb(total),
        )

    def zero_stage_1(self) -> MemoryProfile:
        """ZeRO Stage 1: Partition Optimizer States (P_os)."""
        param_bytes = 2.0 * self.num_parameters
        grad_bytes = 2.0 * self.num_parameters
        opt_bytes = (12.0 * self.num_parameters) / self.world_size
        total = param_bytes + grad_bytes + opt_bytes

        return MemoryProfile(
            parameters_gb=self._to_gb(param_bytes),
            gradients_gb=self._to_gb(grad_bytes),
            optimizer_states_gb=self._to_gb(opt_bytes),
            total_static_gb=self._to_gb(total),
        )

    def zero_stage_2(self) -> MemoryProfile:
        """ZeRO Stage 2: Partition Optimizer States & Gradients (P_os+g)."""
        param_bytes = 2.0 * self.num_parameters
        grad_bytes = (2.0 * self.num_parameters) / self.world_size
        opt_bytes = (12.0 * self.num_parameters) / self.world_size
        total = param_bytes + grad_bytes + opt_bytes

        return MemoryProfile(
            parameters_gb=self._to_gb(param_bytes),
            gradients_gb=self._to_gb(grad_bytes),
            optimizer_states_gb=self._to_gb(opt_bytes),
            total_static_gb=self._to_gb(total),
        )

    def zero_stage_3(self) -> MemoryProfile:
        """ZeRO Stage 3: Partition Parameters, Gradients & Optimizer States (P_os+g+p)."""
        param_bytes = (2.0 * self.num_parameters) / self.world_size
        grad_bytes = (2.0 * self.num_parameters) / self.world_size
        opt_bytes = (12.0 * self.num_parameters) / self.world_size
        total = param_bytes + grad_bytes + opt_bytes

        return MemoryProfile(
            parameters_gb=self._to_gb(param_bytes),
            gradients_gb=self._to_gb(grad_bytes),
            optimizer_states_gb=self._to_gb(opt_bytes),
            total_static_gb=self._to_gb(total),
        )

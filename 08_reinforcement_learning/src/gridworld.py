"""Lightweight, self-contained GridWorld MDP environment."""

from typing import Any
import numpy as np


class GridWorld:
    """Discrete 2D GridWorld Environment.

    Actions:
        0: Up
        1: Right
        2: Down
        3: Left

    Parameters
    ----------
    height : int
        Grid height (rows).
    width : int
        Grid width (columns).
    start_pos : tuple of (int, int)
        Starting coordinate (r, c).
    goal_pos : tuple of (int, int)
        Goal coordinate (r, c).
    obstacles : list of tuple of (int, int)
        Coordinates of blocked obstacle cells.
    step_penalty : float
        Reward per movement step (encourages shortest path).
    goal_reward : float
        Reward upon reaching goal state.
    """

    ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # Up, Right, Down, Left

    def __init__(
        self,
        height: int = 4,
        width: int = 4,
        start_pos: tuple[int, int] = (0, 0),
        goal_pos: tuple[int, int] = (3, 3),
        obstacles: list[tuple[int, int]] | None = None,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
    ) -> None:
        self.height = height
        self.width = width
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = set(obstacles or [])
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward

        self.n_states = height * width
        self.n_actions = len(self.ACTIONS)
        self.agent_pos = start_pos

    def pos_to_state(self, pos: tuple[int, int]) -> int:
        """Convert 2D (r, c) to discrete state index."""
        return pos[0] * self.width + pos[1]

    def state_to_pos(self, state: int) -> tuple[int, int]:
        """Convert discrete state index to 2D (r, c)."""
        return (state // self.width, state % self.width)

    def reset(self) -> int:
        """Reset environment to start position."""
        self.agent_pos = self.start_pos
        return self.pos_to_state(self.agent_pos)

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        """Take an action in the environment.

        Returns
        -------
        next_state : int
        reward : float
        done : bool
        info : dict
        """
        dr, dc = self.ACTIONS[action]
        new_r = np.clip(self.agent_pos[0] + dr, 0, self.height - 1)
        new_c = np.clip(self.agent_pos[1] + dc, 0, self.width - 1)
        candidate_pos = (int(new_r), int(new_c))

        if candidate_pos in self.obstacles:
            # Blocked by obstacle, stay in place
            pass
        else:
            self.agent_pos = candidate_pos

        next_state = self.pos_to_state(self.agent_pos)
        done = self.agent_pos == self.goal_pos
        reward = self.goal_reward if done else self.step_penalty

        return next_state, reward, done, {}

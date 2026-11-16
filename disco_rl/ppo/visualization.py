"""Side-by-side deterministic PPO episodes, including pre-reset terminal frames."""

from dataclasses import dataclass
from pathlib import Path

import dm_env
import haiku as hk
import jax
from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from disco_rl.environments import jittable_envs
from disco_rl.ppo.training import PPOResult
from disco_rl.ppo.training import PPOTrainer


@dataclass(frozen=True)
class Episode:
  frames: np.ndarray
  actions: np.ndarray
  returns: np.ndarray


@dataclass(frozen=True)
class EpisodeComparison:
  episodes: tuple[Episode, ...]
  summary: pd.DataFrame
  html: str


def _episode_runner(trainer: PPOTrainer, env, episode_seed: int):
  core = env._env

  @jax.jit
  def step(params, state, obs):
    mean, _, _ = trainer.network.apply(params, obs)
    actions = trainer.bounded_action(mean)
    # The wrapper resets LAST state; retain the actual transition for rendering.
    step_keys = jax.vmap(lambda key: jax.random.split(key)[1])(state.rng)
    frame_state = jax.vmap(core.step)(step_keys, state.state, actions)
    next_state, timestep = env.step(state, actions)
    return next_state, timestep, frame_state[0], actions[0]

  def run(params: hk.Params) -> Episode:
    state, timestep = env.reset(jax.random.PRNGKey(episode_seed))
    frames = [np.asarray(jax.device_get(state.state[0]))]
    actions, rewards = [0.0], [0.0]
    for _ in range(core._rows):
      state, timestep, frame_state, action = step(
          params, state, timestep.observation
      )
      frames.append(np.asarray(jax.device_get(frame_state)))
      actions.append(float(action))
      rewards.append(float(timestep.reward[0]))
      if (
          not np.isfinite(frames[-1]).all()
          or not np.isfinite([actions[-1], rewards[-1]]).all()
      ):
        raise FloatingPointError(
            'Non-finite state, action, or reward during visualisation.'
        )
      if int(timestep.step_type[0]) == dm_env.StepType.LAST:
        return Episode(
            frames=np.stack(frames), actions=np.asarray(actions),
            returns=np.cumsum(rewards),
        )
    raise RuntimeError(
        'The visualised Catch episode did not terminate within its row count.'
    )

  return run


def visualize_episode(
    result: PPOResult, *, training_seed: int = 301, episode_seed: int = 42,
    middle_steps: int = 24_576, output_dir: Path,
) -> EpisodeComparison:
  """Save GIF, PNG, HTML, and CSV; return HTML for the notebook to display."""
  trainer = result.trainer
  final_steps = trainer.config.total_steps
  if (
      middle_steps <= 0 or middle_steps % trainer.config.batch_size
      or final_steps <= middle_steps
  ):
    raise ValueError(
        'The middle checkpoint must be a completed update before the final checkpoint.'
    )
  env = trainer.env_factory(1)
  if not isinstance(env, jittable_envs.ContinuousCatchJittableEnvironment):
    raise ValueError('Episode visualisation requires continuous Catch.')
  rows, columns = env._env._rows, env._env._columns
  radius = env._env._catch_radius
  run_episode = _episode_runner(trainer, env, episode_seed)
  steps = (0, middle_steps, final_steps)
  episodes = tuple(
      run_episode(result.checkpoint(training_seed, checkpoint))
      for checkpoint in steps
  )
  summary = pd.DataFrame([
      dict(
          training_transitions=checkpoint, training_seed=training_seed,
          episode_seed=episode_seed, episode_steps=len(episode.actions) - 1,
          episode_return=episode.returns[-1], caught=episode.returns[-1] > 0,
      )
      for checkpoint, episode in zip(steps, episodes)
  ])
  print(summary.to_string(index=False))
  print('One fixed episode, not an average success rate. Press Play below to animate.')

  fig, axes = plt.subplots(
      1, 3, figsize=(13, 5), sharex=True, sharey=True
  )
  panel_artists = []
  for ax, checkpoint, episode in zip(axes, steps, episodes):
    label = (
        'Untrained' if checkpoint == 0
        else ('Fully trained' if checkpoint == final_steps else 'Early training')
    )
    ax.set(
        title=f'{label}\n{checkpoint:,} environment transitions',
        xlabel='Horizontal position', xlim=(-0.5, columns - 0.5),
        ylim=(rows - 0.5, -0.5), aspect='equal',
    )
    ax.set_xticks(np.arange(columns))
    ax.set_yticks(np.arange(rows))
    ax.grid(alpha=0.2)
    ax.axhline(rows - 1, color='grey', linewidth=1, linestyle=':')
    ax.axvline(
        episode.frames[0, 1], color='orange',
        linewidth=1, linestyle=':', alpha=0.4,
    )
    ball, = ax.plot(
        [], [], 'o', color='orange', markersize=12, label='Ball'
    )
    paddle, = ax.plot(
        [], [], color='royalblue', linewidth=7,
        solid_capstyle='round', label='Paddle / catch radius',
    )
    status = ax.text(
        0.5, -0.18, '', transform=ax.transAxes, ha='center', va='top'
    )
    panel_artists.append((ball, paddle, status))
  axes[0].set_ylabel('Falling ball row')
  axes[0].legend(loc='upper left', fontsize=8)
  fig.suptitle(
      'Same continuous Catch episode | deterministic PPO | '
      f'training seed {training_seed}'
  )
  fig.tight_layout(rect=(0, 0.07, 1, 0.92))

  def draw(frame):
    artists = []
    for episode, (ball, paddle, status) in zip(episodes, panel_artists):
      index = min(frame, len(episode.frames) - 1)
      ball_y, ball_x, paddle_y, paddle_x = episode.frames[index]
      ball.set_data([ball_x], [ball_y])
      paddle.set_data(
          [paddle_x - radius, paddle_x + radius], [paddle_y, paddle_y]
      )
      finished = index == len(episode.frames) - 1
      outcome = (
          ('CAUGHT' if episode.returns[index] > 0 else 'MISSED')
          if finished else 'In progress'
      )
      status.set_text(
          f'Step {index} | action {episode.actions[index]:+.2f}\n'
          f'{outcome} | return {episode.returns[index]:+.0f}'
      )
      status.set_color(
          ('forestgreen' if episode.returns[index] > 0 else 'firebrick')
          if finished else 'black'
      )
      artists.extend((ball, paddle, status))
    return artists

  frame_count = max(len(episode.frames) for episode in episodes)
  movie = animation.FuncAnimation(
      fig, draw, frames=list(range(frame_count)) + [frame_count - 1] * 4,
      init_func=lambda: draw(0), interval=400, blit=False, repeat=False,
  )
  html = movie.to_jshtml(default_mode='once')
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / 'episode_comparison.html').write_text(html, encoding='utf-8')
  movie.save(
      output_dir / 'episode_comparison.gif',
      writer=animation.PillowWriter(fps=2.5), dpi=150,
  )
  draw(frame_count - 1)
  fig.savefig(output_dir / 'episode_comparison_final.png', dpi=150)
  summary.to_csv(output_dir / 'episode_comparison.csv', index=False)
  print(f'GIF, final-frame PNG, HTML, and episode results saved to {output_dir}')
  plt.close(fig)
  return EpisodeComparison(episodes=episodes, summary=summary, html=html)

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from jaxatari._dtypes import COUNTER_DTYPE, counter_array
from jaxatari.core import make
from jaxatari.games.jax_beamrider import WhiteUFOPattern


def _get_base_env(env):
    while hasattr(env, "_env"):
        env = env._env
    return env


def test_beamrider_steps_follow_counter_dtype():
    env = make("beamrider")
    _, state = env.reset(jax.random.PRNGKey(0))

    assert state.steps.dtype == np.dtype(COUNTER_DTYPE)


@pytest.mark.skipif(np.dtype(COUNTER_DTYPE) != np.dtype(np.int64), reason="requires x64 counters")
def test_beamrider_step_counter_crosses_int32_boundary():
    env = make("beamrider")
    _, state = env.reset(jax.random.PRNGKey(1))

    large_step = counter_array(np.iinfo(np.int32).max + 5)
    state = state._replace(steps=large_step)

    _, new_state, _, _, _ = env._handle_init_phase(
        state,
        state.level.line_positions,
        state.level.blue_line_counter,
    )

    assert new_state.steps.dtype == large_step.dtype
    assert int(new_state.steps) == int(large_step) + 1


@pytest.mark.skipif(np.dtype(COUNTER_DTYPE) != np.dtype(np.int64), reason="requires x64 counters")
def test_beamrider_three_lanes_mod_handles_large_step_counters():
    env = make("beamrider", mods=["three_lanes"])
    base = _get_base_env(env)
    _, state = env.reset(jax.random.PRNGKey(2))

    white_ufo_pos = jnp.array(
        [
            [71.0, 81.0, 91.0],
            [70.0, 70.0, 70.0],
        ],
        dtype=jnp.float32,
    )
    shoot_timer = jnp.full((3,), base.ufo_pattern_durations[int(WhiteUFOPattern.SHOOT)], dtype=jnp.int32)
    shoot_pattern = jnp.full((3,), int(WhiteUFOPattern.SHOOT), dtype=jnp.int32)
    large_step = counter_array(np.iinfo(np.int32).max + 3001)
    state = state._replace(steps=large_step, ufo_killed=jnp.array(True))

    shot_pos, shot_lane, shot_timer_out, hit_count = base._enemy_shot_step(
        state,
        white_ufo_pos,
        shoot_pattern,
        shoot_timer,
    )

    active = shot_pos[1] <= float(base.consts.BOTTOM_CLIP)
    assert int(jnp.sum(active)) == 3
    assert bool(jnp.all(jnp.isin(shot_lane[active], jnp.array([2, 3, 4], dtype=jnp.int32))))
    assert bool(jnp.all(shot_timer_out[active] == 0))
    assert int(hit_count) == 0

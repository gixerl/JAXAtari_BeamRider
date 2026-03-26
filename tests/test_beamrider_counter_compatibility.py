import numpy as np
import jax
import jax.numpy as jnp

from jaxatari._dtypes import COUNTER_DTYPE, counter_array
from jaxatari.games.jax_beamrider import JaxBeamrider, WhiteUFOPattern
from jaxatari.games.mods.beamrider_mods import BeamriderEnvMod
from jaxatari.modification import JaxAtariModWrapper


def _get_base_env(env):
    while hasattr(env, "_env"):
        env = env._env
    return env


def _make_three_lanes_env():
    base_env = JaxBeamrider()
    controller = BeamriderEnvMod(env=base_env, mods_config=["three_lanes"])
    return JaxAtariModWrapper(env=controller, mods_config=["three_lanes"])


def test_beamrider_steps_follow_counter_dtype():
    env = JaxBeamrider()
    _, state = env.reset(jax.random.PRNGKey(0))

    assert state.steps.dtype == np.dtype(COUNTER_DTYPE)


def test_beamrider_step_counter_increments_with_counter_dtype():
    env = JaxBeamrider()
    _, state = env.reset(jax.random.PRNGKey(1))

    step_value = counter_array(3001)
    state = state._replace(steps=step_value)

    _, new_state, _, _, _ = env._handle_init_phase(
        state,
        state.level.line_positions,
        state.level.blue_line_counter,
    )

    assert new_state.steps.dtype == step_value.dtype
    assert int(new_state.steps) == int(step_value) + 1


def test_beamrider_three_lanes_mod_handles_counter_dtype_steps():
    env = _make_three_lanes_env()
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
    state = state._replace(steps=counter_array(3001), ufo_killed=jnp.array(True))

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

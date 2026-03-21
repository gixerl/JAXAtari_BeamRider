import jax
import jax.numpy as jnp

from jaxatari.games.jax_beamrider import JaxBeamrider
from jaxatari.games.mods.beamrider_mods import BeamriderEnvMod
from jaxatari.modification import JaxAtariModWrapper


def _get_base_env(env):
    while hasattr(env, "_env"):
        env = env._env
    return env


def _make_fog_env():
    base_env = JaxBeamrider()
    controller = BeamriderEnvMod(env=base_env, mods_config=["fog_of_war"])
    return JaxAtariModWrapper(env=controller, mods_config=["fog_of_war"])


def _build_render_state(base, state, *, show_top_ufo=False, show_lower_enemy_shot=False):
    white_ufo_pos = base.enemy_offscreen_ufo
    if show_top_ufo:
        white_ufo_pos = white_ufo_pos.at[:, 0].set(jnp.array([81.0, 60.0], dtype=jnp.float32))

    enemy_shot_pos = base.bullet_offscreen_shots
    if show_lower_enemy_shot:
        enemy_shot_pos = enemy_shot_pos.at[:, 0].set(jnp.array([81.0, 150.0], dtype=jnp.float32))

    level = state.level._replace(
        blue_line_counter=jnp.array(500, dtype=jnp.int32),
        standby_phase=jnp.array(0, dtype=jnp.int32),
        white_ufo_left=jnp.array(1, dtype=jnp.int32),
        white_ufo_pos=white_ufo_pos,
        ufo_explosion_frame=jnp.zeros((3,), dtype=jnp.int32),
        ufo_explosion_pos=base.enemy_offscreen_ufo,
        enemy_shot_pos=enemy_shot_pos,
        enemy_shot_timer=jnp.zeros((9,), dtype=jnp.int32),
        enemy_shot_explosion_frame=jnp.zeros((9,), dtype=jnp.int32),
        enemy_shot_explosion_pos=base.enemy_offscreen_shots,
    )
    return state._replace(level=level)


def test_beamrider_fog_of_war_hides_upper_playfield_objects():
    plain_env = JaxBeamrider()
    fog_env = _make_fog_env()
    fog_base = _get_base_env(fog_env)
    _, state = fog_env.reset(jax.random.PRNGKey(0))

    state_with_top_ufo = _build_render_state(fog_base, state, show_top_ufo=True)
    state_without_top_ufo = _build_render_state(fog_base, state, show_top_ufo=False)

    fog_render_with_ufo = fog_env.render(state_with_top_ufo)
    fog_render_without_ufo = fog_env.render(state_without_top_ufo)
    assert bool(jnp.array_equal(fog_render_with_ufo, fog_render_without_ufo))

    plain_render_with_ufo = plain_env.render(state_with_top_ufo)
    plain_render_without_ufo = plain_env.render(state_without_top_ufo)
    assert not bool(jnp.array_equal(plain_render_with_ufo, plain_render_without_ufo))


def test_beamrider_fog_of_war_keeps_lower_playfield_visible():
    fog_env = _make_fog_env()
    fog_base = _get_base_env(fog_env)
    _, state = fog_env.reset(jax.random.PRNGKey(1))

    state_with_lower_shot = _build_render_state(fog_base, state, show_lower_enemy_shot=True)
    state_without_lower_shot = _build_render_state(fog_base, state, show_lower_enemy_shot=False)

    fog_render_with_shot = fog_env.render(state_with_lower_shot)
    fog_render_without_shot = fog_env.render(state_without_lower_shot)
    assert not bool(jnp.array_equal(fog_render_with_shot, fog_render_without_shot))

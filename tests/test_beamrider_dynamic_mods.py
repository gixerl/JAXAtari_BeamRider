import jax
import jax.numpy as jnp

from scripts.utils import load_game_environment, load_game_mods


def test_beamrider_hardcore_mod_uses_dynamic_loader():
    env, _ = load_game_environment("beamrider")
    apply_mods = load_game_mods("beamrider", ["hardcore"])
    modded_env = apply_mods(env)

    assert modded_env.consts.STARTING_LIVES == 1
    assert modded_env.consts.MAX_LIVES == 1

    obs, state = modded_env.reset(jax.random.PRNGKey(0))

    assert int(obs.lives) == 1
    assert int(state.lives) == 1

    for _ in range(5):
        obs, state, reward, done, info = modded_env.step(
            state,
            jnp.array(0, dtype=jnp.int32),
        )
        assert int(state.lives) <= 1

import pytest
import jax
import jax.numpy as jnp

from jaxatari.games.jax_beamrider import BeamriderConstants, BouncerState, JaxBeamrider, LaneBlockerState, WhiteUFOUpdate, WhiteUFOPattern
from jaxatari.games.mods.beamrider.beamrider_mod_plugins import DoubleEnemySpeedMod
from jaxatari.games.mods.beamrider_mods import BeamriderEnvMod
from jaxatari.modification import JaxAtariModWrapper


def _get_base_env(env):
    while hasattr(env, "_env"):
        env = env._env
    return env


def _make_fast_env():
    mod_consts = BeamriderConstants()._replace(**DoubleEnemySpeedMod.constants_overrides)
    base_env = JaxBeamrider(consts=mod_consts)
    controller = BeamriderEnvMod(env=base_env, mods_config=["double_enemy_speed"])
    env = JaxAtariModWrapper(env=controller, mods_config=["double_enemy_speed"])
    return env, _get_base_env(env)


def _enemy_updates_from_state(state):
    return {
        "ufo": WhiteUFOUpdate(
            pos=state.level.white_ufo_pos,
            vel=state.level.white_ufo_vel,
            time_on_lane=state.level.white_ufo_time_on_lane,
            attack_time=state.level.white_ufo_attack_time,
            already_left=state.level.white_ufo_already_left,
            spawn_delay=state.level.white_ufo_spawn_delay,
            pattern_id=state.level.white_ufo_pattern_id,
            pattern_timer=state.level.white_ufo_pattern_timer,
            rngs=state.level.white_ufo_rngs,
        ),
        "bouncer": (
            state.level.bouncer_pos,
            state.level.bouncer_vel,
            state.level.bouncer_state,
            state.level.bouncer_timer,
            state.level.bouncer_active,
            state.level.bouncer_lane,
            state.level.bouncer_step_index,
        ),
        "meteoroid": (
            state.level.chasing_meteoroid_pos,
            state.level.chasing_meteoroid_active,
            state.level.chasing_meteoroid_vel_y,
            state.level.chasing_meteoroid_phase,
            state.level.chasing_meteoroid_frame,
            state.level.chasing_meteoroid_lane,
            state.level.chasing_meteoroid_side,
            state.level.chasing_meteoroid_spawn_timer,
            state.level.chasing_meteoroid_remaining,
            state.level.chasing_meteoroid_wave_active,
        ),
        "rock": (
            state.level.falling_rock_pos,
            state.level.falling_rock_active,
            state.level.falling_rock_lane,
            state.level.falling_rock_vel_y,
        ),
        "blocker": (
            state.level.lane_blocker_pos,
            state.level.lane_blocker_active,
            state.level.lane_blocker_lane,
            state.level.lane_blocker_vel_y,
            state.level.lane_blocker_phase,
            state.level.lane_blocker_timer,
        ),
        "kamikaze": (
            state.level.kamikaze_pos,
            state.level.kamikaze_active,
            state.level.kamikaze_lane,
            state.level.kamikaze_vel_y,
            state.level.kamikaze_tracking,
            state.level.kamikaze_spawn_timer,
        ),
        "coin": (
            state.level.coin_pos,
            state.level.coin_active,
            state.level.coin_timer,
            state.level.coin_side,
            state.level.coin_spawn_count,
        ),
        "rejuv": (
            state.level.rejuvenator_pos,
            state.level.rejuvenator_active,
            state.level.rejuvenator_dead,
            state.level.rejuvenator_frame,
            state.level.rejuvenator_lane,
        ),
        "shots": (
            state.level.enemy_shot_pos,
            state.level.enemy_shot_vel,
            state.level.enemy_shot_timer,
            jnp.array(0, dtype=jnp.int32),
        ),
    }


def _prepare_collision_state(state, bullet_type):
    shot_pos = jnp.array([80.0, 100.0], dtype=jnp.float32)
    if int(bullet_type) == 2:
        shot_pos = jnp.array([78.0, 100.0], dtype=jnp.float32)

    level = state.level._replace(
        blue_line_counter=jnp.array(500, dtype=jnp.int32),
        standby_phase=jnp.array(0, dtype=jnp.int32),
        player_pos=jnp.array(77.0, dtype=jnp.float32),
        player_vel=jnp.array(0.0, dtype=jnp.float32),
        player_shot_pos=shot_pos,
        player_shot_vel=jnp.array([0.0, -1.0], dtype=jnp.float32),
        player_shot_frame=jnp.array(4, dtype=jnp.int32),
        bullet_type=bullet_type,
        shot_type_pending=bullet_type,
        shooting_cooldown=jnp.array(0, dtype=jnp.int32),
        shooting_delay=jnp.array(0, dtype=jnp.int32),
    )
    return state._replace(level=level, sector=jnp.array(12, dtype=jnp.int32), lives=jnp.array(3, dtype=jnp.int32), steps=jnp.array(5000, dtype=jnp.int32))


@pytest.mark.parametrize(
    ("enemy_kind", "bullet_type"),
    [
        ("ufo", 1),
        ("bouncer", 2),
        ("meteoroid", 2),
        ("rock", 2),
        ("blocker", 1),
        ("rejuvenator", 1),
        ("coin", 1),
        ("kamikaze", 2),
    ],
)
def test_beamrider_double_enemy_speed_swept_collisions_keep_projectile_hits(enemy_kind, bullet_type):
    env, base = _make_fast_env()
    _, state = env.reset(jax.random.PRNGKey(0))
    state = _prepare_collision_state(state, jnp.array(bullet_type, dtype=jnp.int32))
    enemy_updates = _enemy_updates_from_state(state)

    if enemy_kind == "ufo":
        prev_ufo = base.enemy_offscreen_ufo.at[:, 0].set(jnp.array([80.0, 92.0], dtype=jnp.float32))
        next_ufo = prev_ufo.at[:, 0].set(jnp.array([80.0, 108.0], dtype=jnp.float32))
        state = state._replace(level=state.level._replace(white_ufo_pos=prev_ufo, white_ufo_left=jnp.array(1, dtype=jnp.int32)))
        enemy_updates["ufo"] = enemy_updates["ufo"]._replace(pos=next_ufo)
    elif enemy_kind == "bouncer":
        state = state._replace(
            level=state.level._replace(
                bouncer_pos=jnp.array([80.0, 92.0], dtype=jnp.float32),
                bouncer_active=jnp.array(True),
            )
        )
        enemy_updates["bouncer"] = (
            jnp.array([80.0, 108.0], dtype=jnp.float32),
            state.level.bouncer_vel,
            state.level.bouncer_state,
            state.level.bouncer_timer,
            jnp.array(True),
            state.level.bouncer_lane,
            state.level.bouncer_step_index,
        )
    elif enemy_kind == "meteoroid":
        prev = base.enemy_offscreen_meteoroids.at[:, 0].set(jnp.array([80.0, 92.0], dtype=jnp.float32))
        next_pos = prev.at[:, 0].set(jnp.array([80.0, 108.0], dtype=jnp.float32))
        active = jnp.array([True] + [False] * (base.consts.CHASING_METEOROID_MAX - 1), dtype=jnp.bool_)
        state = state._replace(level=state.level._replace(chasing_meteoroid_pos=prev, chasing_meteoroid_active=active))
        enemy_updates["meteoroid"] = (
            next_pos,
            active,
            state.level.chasing_meteoroid_vel_y,
            state.level.chasing_meteoroid_phase,
            state.level.chasing_meteoroid_frame,
            state.level.chasing_meteoroid_lane,
            state.level.chasing_meteoroid_side,
            state.level.chasing_meteoroid_spawn_timer,
            state.level.chasing_meteoroid_remaining,
            state.level.chasing_meteoroid_wave_active,
        )
    elif enemy_kind == "rock":
        prev = base.enemy_offscreen_falling.at[:, 0].set(jnp.array([80.0, 92.0], dtype=jnp.float32))
        next_pos = prev.at[:, 0].set(jnp.array([80.0, 108.0], dtype=jnp.float32))
        active = jnp.array([True] + [False] * (base.consts.FALLING_ROCK_MAX - 1), dtype=jnp.bool_)
        state = state._replace(level=state.level._replace(falling_rock_pos=prev, falling_rock_active=active))
        enemy_updates["rock"] = (next_pos, active, state.level.falling_rock_lane, state.level.falling_rock_vel_y)
    elif enemy_kind == "blocker":
        prev = jnp.array([[80.0], [92.0]], dtype=jnp.float32)
        next_pos = jnp.array([[80.0], [108.0]], dtype=jnp.float32)
        state = state._replace(
            level=state.level._replace(
                lane_blocker_pos=prev,
                lane_blocker_active=jnp.array([True], dtype=jnp.bool_),
                lane_blocker_phase=jnp.array([int(LaneBlockerState.DESCEND)], dtype=jnp.int32),
            )
        )
        enemy_updates["blocker"] = (
            next_pos,
            jnp.array([True], dtype=jnp.bool_),
            state.level.lane_blocker_lane,
            state.level.lane_blocker_vel_y,
            state.level.lane_blocker_phase,
            state.level.lane_blocker_timer,
        )
    elif enemy_kind == "rejuvenator":
        state = state._replace(
            level=state.level._replace(
                rejuvenator_pos=jnp.array([80.0, 92.0], dtype=jnp.float32),
                rejuvenator_active=jnp.array(True),
                rejuvenator_dead=jnp.array(False),
            )
        )
        enemy_updates["rejuv"] = (
            jnp.array([80.0, 108.0], dtype=jnp.float32),
            jnp.array(True),
            jnp.array(False),
            state.level.rejuvenator_frame,
            state.level.rejuvenator_lane,
        )
    elif enemy_kind == "coin":
        prev = base.enemy_offscreen_coins.at[:, 0].set(jnp.array([72.0, 100.0], dtype=jnp.float32))
        next_pos = prev.at[:, 0].set(jnp.array([88.0, 100.0], dtype=jnp.float32))
        active = jnp.array([True] + [False] * (base.consts.COIN_MAX - 1), dtype=jnp.bool_)
        state = state._replace(level=state.level._replace(coin_pos=prev, coin_active=active))
        enemy_updates["coin"] = (next_pos, active, state.level.coin_timer, state.level.coin_side, state.level.coin_spawn_count)
    elif enemy_kind == "kamikaze":
        state = state._replace(
            level=state.level._replace(
                kamikaze_pos=jnp.array([[80.0], [92.0]], dtype=jnp.float32),
                kamikaze_active=jnp.array([True], dtype=jnp.bool_),
            )
        )
        enemy_updates["kamikaze"] = (
            jnp.array([[80.0], [108.0]], dtype=jnp.float32),
            jnp.array([True], dtype=jnp.bool_),
            state.level.kamikaze_lane,
            state.level.kamikaze_vel_y,
            state.level.kamikaze_tracking,
            state.level.kamikaze_spawn_timer,
        )

    result = base._collisions_step(
        state,
        state.level.player_pos,
        state.level.player_vel,
        state.level.player_shot_pos,
        state.level.player_shot_vel,
        state.level.player_shot_frame,
        state.level.torpedoes_left,
        state.level.bullet_type,
        state.level.shooting_cooldown,
        state.level.shooting_delay,
        state.level.shot_type_pending,
        enemy_updates,
        jax.random.PRNGKey(1),
    )

    assert bool(jnp.array_equal(result["player"][2], base.bullet_offscreen))

    if enemy_kind == "ufo":
        assert bool(jnp.array_equal(result["ufo"][0][:, 0], base.enemy_offscreen))
        assert int(result["ufo"][3]) == 0
    elif enemy_kind == "bouncer":
        assert not bool(result["bouncer"][4])
        assert bool(jnp.array_equal(result["bouncer"][0], base.enemy_offscreen))
    elif enemy_kind == "meteoroid":
        assert not bool(result["meteoroid"][1][0])
        assert bool(jnp.array_equal(result["meteoroid"][0][:, 0], base.enemy_offscreen))
    elif enemy_kind == "rock":
        assert not bool(result["rock"][1][0])
        assert bool(jnp.array_equal(result["rock"][0][:, 0], base.enemy_offscreen))
    elif enemy_kind == "blocker":
        assert bool(result["blocker"][1][0])
        assert int(result["blocker"][4][0]) == int(LaneBlockerState.RETREAT)
    elif enemy_kind == "rejuvenator":
        assert bool(result["rejuv"][1])
        assert bool(result["rejuv"][2])
    elif enemy_kind == "coin":
        assert not bool(result["coin"][1][0])
        assert bool(jnp.array_equal(result["coin"][0][:, 0], base.enemy_offscreen))
    elif enemy_kind == "kamikaze":
        assert not bool(result["kamikaze"][1][0])
        assert bool(jnp.array_equal(result["kamikaze"][0], base.kamikaze_offscreen))


def test_beamrider_double_enemy_speed_mothership_moves_farther_per_step():
    plain_env = JaxBeamrider()
    fast_env, fast_base = _make_fast_env()
    _ = fast_env

    _, plain_state = plain_env.reset(jax.random.PRNGKey(1))
    _, fast_state = fast_base.reset(jax.random.PRNGKey(1))

    def _with_mothership(state):
        level = state.level._replace(
            mothership_stage=jnp.array(2, dtype=jnp.int32),
            mothership_timer=jnp.array(1, dtype=jnp.int32),
            mothership_position=jnp.array(50.0, dtype=jnp.float32),
            white_ufo_left=jnp.array(0, dtype=jnp.int32),
        )
        return state._replace(level=level, sector=jnp.array(11, dtype=jnp.int32))

    plain_state = _with_mothership(plain_state)
    fast_state = _with_mothership(fast_state)

    plain_pos, plain_timer, plain_stage, _ = plain_env._mothership_step(
        plain_state,
        plain_state.level.white_ufo_left,
        plain_state.level.ufo_explosion_frame,
        jnp.array(False),
    )
    fast_pos, fast_timer, fast_stage, _ = fast_base._mothership_step(
        fast_state,
        fast_state.level.white_ufo_left,
        fast_state.level.ufo_explosion_frame,
        jnp.array(False),
    )

    assert float(fast_pos) > float(plain_pos)
    assert int(fast_timer) > int(plain_timer)
    assert int(fast_stage) == int(plain_stage)


@pytest.mark.parametrize("pattern_id", [int(WhiteUFOPattern.DROP_RIGHT), int(WhiteUFOPattern.TRIPLE_SHOT_LEFT)])
def test_beamrider_double_enemy_speed_white_ufos_keep_diagonal_patterns_at_2x_speed(pattern_id):
    plain_env = JaxBeamrider()
    _, fast_base = _make_fast_env()

    pos = jnp.array([plain_env.top_lanes_x[3], 60.0], dtype=jnp.float32)
    white_ufo_vel_x = jnp.array(0.0, dtype=jnp.float32)
    white_ufo_vel_y = jnp.array(0.0, dtype=jnp.float32)
    already_left = jnp.array(False)

    plain_vx, plain_vy = plain_env._white_ufo_normal(
        pos,
        white_ufo_vel_x,
        white_ufo_vel_y,
        jnp.array(pattern_id, dtype=jnp.int32),
        already_left,
    )
    fast_vx, fast_vy = fast_base._white_ufo_normal(
        pos,
        white_ufo_vel_x,
        white_ufo_vel_y,
        jnp.array(pattern_id, dtype=jnp.int32),
        already_left,
    )

    assert float(fast_vx) == pytest.approx(float(plain_vx) * 2.0, rel=1e-5, abs=1e-5)
    assert float(fast_vy) == pytest.approx(float(plain_vy) * 2.0, rel=1e-5, abs=1e-5)


def test_beamrider_double_enemy_speed_white_ufo_step_matches_two_stock_motion_substeps():
    plain_env = JaxBeamrider()
    _, fast_base = _make_fast_env()

    position = jnp.array([plain_env.top_lanes_x[3], 60.0], dtype=jnp.float32)
    pattern_id = jnp.array(int(WhiteUFOPattern.DROP_RIGHT), dtype=jnp.int32)
    zero_f = jnp.array(0.0, dtype=jnp.float32)

    def _clip_ufo_position(env, position_in):
        new_x = position_in[0]
        new_y = position_in[1]
        on_top_lane = new_y <= env.consts.TOP_CLIP
        clipped_x = jnp.clip(new_x, env.consts.LEFT_CLIP_PLAYER, env.consts.RIGHT_CLIP_PLAYER)
        new_x = jnp.where(on_top_lane, clipped_x, new_x)
        new_y = jnp.clip(new_y, env.consts.TOP_CLIP, env.consts.PLAYER_POS_Y + 1.0)
        return jnp.array([new_x, new_y], dtype=jnp.float32)

    plain_vx_1, plain_vy_1 = plain_env._white_ufo_normal(position, zero_f, zero_f, pattern_id, jnp.array(True))
    pos_1 = _clip_ufo_position(plain_env, position + jnp.array([plain_vx_1, plain_vy_1], dtype=jnp.float32))
    plain_vx_2, plain_vy_2 = plain_env._white_ufo_normal(pos_1, plain_vx_1, plain_vy_1, pattern_id, jnp.array(True))
    expected_position = _clip_ufo_position(plain_env, pos_1 + jnp.array([plain_vx_2, plain_vy_2], dtype=jnp.float32))

    fast_position, fast_vx, fast_vy, _, _, _, _, fast_pattern_id, fast_pattern_timer, _ = fast_base._white_ufo_step(
        jnp.array(12, dtype=jnp.int32),
        position,
        jnp.array([0.0, 0.0], dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(False),
        jnp.array(0, dtype=jnp.int32),
        pattern_id,
        jnp.array(10, dtype=jnp.int32),
        jax.random.PRNGKey(11),
    )

    assert bool(jnp.allclose(fast_position, expected_position, atol=1e-5, rtol=1e-5))
    assert float(fast_vx) == pytest.approx(float(plain_vx_2), rel=1e-5, abs=1e-5)
    assert float(fast_vy) == pytest.approx(float(plain_vy_2), rel=1e-5, abs=1e-5)
    assert int(fast_pattern_id) == int(pattern_id)
    assert int(fast_pattern_timer) == 9


@pytest.mark.parametrize("rock_y", [50.0, 70.0, 90.0])
def test_beamrider_double_enemy_speed_falling_rocks_are_twice_as_fast_across_all_accel_bands(rock_y):
    plain_env = JaxBeamrider()
    _, fast_base = _make_fast_env()

    _, plain_state = plain_env.reset(jax.random.PRNGKey(5))
    _, fast_state = fast_base.reset(jax.random.PRNGKey(5))

    active = jnp.array([True] + [False] * (plain_env.consts.FALLING_ROCK_MAX - 1), dtype=jnp.bool_)
    pos = plain_env.enemy_offscreen_falling.at[:, 0].set(jnp.array([80.0, rock_y], dtype=jnp.float32))
    plain_vel = jnp.array([plain_env.consts.FALLING_ROCK_INIT_VEL] + [0.0] * (plain_env.consts.FALLING_ROCK_MAX - 1), dtype=jnp.float32)
    fast_vel = jnp.array([fast_base.consts.FALLING_ROCK_INIT_VEL] + [0.0] * (fast_base.consts.FALLING_ROCK_MAX - 1), dtype=jnp.float32)
    lane = jnp.array([3] + [0] * (plain_env.consts.FALLING_ROCK_MAX - 1), dtype=jnp.int32)

    plain_state = plain_state._replace(level=plain_state.level._replace(falling_rock_pos=pos, falling_rock_active=active, falling_rock_vel_y=plain_vel, falling_rock_lane=lane))
    fast_state = fast_state._replace(level=fast_state.level._replace(falling_rock_pos=pos, falling_rock_active=active, falling_rock_vel_y=fast_vel, falling_rock_lane=lane))

    plain_pos, _, _, _ = plain_env._falling_rock_step(plain_state, jax.random.PRNGKey(6))
    fast_pos, _, _, _ = fast_base._falling_rock_step(fast_state, jax.random.PRNGKey(6))

    plain_delta = float(plain_pos[1, 0] - pos[1, 0])
    fast_delta = float(fast_pos[1, 0] - pos[1, 0])

    assert fast_delta == pytest.approx(plain_delta * 2.0, rel=1e-5, abs=1e-5)


def test_beamrider_double_enemy_speed_mothership_phase_starts_meteoroid_waves_more_often():
    plain_env = JaxBeamrider()
    _, fast_base = _make_fast_env()

    _, plain_state = plain_env.reset(jax.random.PRNGKey(8))
    _, fast_state = fast_base.reset(jax.random.PRNGKey(8))

    inactive = jnp.zeros((plain_env.consts.CHASING_METEOROID_MAX,), dtype=jnp.bool_)
    offscreen = plain_env.enemy_offscreen_meteoroids
    zeros_i = jnp.zeros((plain_env.consts.CHASING_METEOROID_MAX,), dtype=jnp.int32)
    zeros_f = jnp.zeros((plain_env.consts.CHASING_METEOROID_MAX,), dtype=jnp.float32)
    ones_i = jnp.ones((plain_env.consts.CHASING_METEOROID_MAX,), dtype=jnp.int32)

    def _with_mothership_window(state):
        level = state.level._replace(
            player_pos=jnp.array(77.0, dtype=jnp.float32),
            white_ufo_left=jnp.array(0, dtype=jnp.int32),
            mothership_stage=jnp.array(2, dtype=jnp.int32),
            mothership_position=jnp.array(80.0, dtype=jnp.float32),
            chasing_meteoroid_pos=offscreen,
            chasing_meteoroid_active=inactive,
            chasing_meteoroid_vel_y=zeros_f,
            chasing_meteoroid_phase=zeros_i,
            chasing_meteoroid_frame=zeros_i,
            chasing_meteoroid_lane=zeros_i,
            chasing_meteoroid_side=ones_i,
            chasing_meteoroid_spawn_timer=jnp.array(0, dtype=jnp.int32),
            chasing_meteoroid_remaining=jnp.array(0, dtype=jnp.int32),
            chasing_meteoroid_wave_active=jnp.array(False),
        )
        return state._replace(level=level, sector=jnp.array(11, dtype=jnp.int32))

    plain_state = _with_mothership_window(plain_state)
    fast_state = _with_mothership_window(fast_state)
    key = jax.random.PRNGKey(67)

    plain_out = plain_env._chasing_meteoroid_step(
        plain_state,
        plain_state.level.player_pos,
        plain_state.level.player_vel,
        plain_state.level.white_ufo_left,
        key,
    )
    fast_out = fast_base._chasing_meteoroid_step(
        fast_state,
        fast_state.level.player_pos,
        fast_state.level.player_vel,
        fast_state.level.white_ufo_left,
        key,
    )

    assert int(jnp.sum(plain_out[1], dtype=jnp.int32)) == 0
    assert int(jnp.sum(fast_out[1], dtype=jnp.int32)) == 1
    assert bool(fast_out[9])


def test_beamrider_double_enemy_speed_mothership_phase_halves_meteoroid_spawn_intervals():
    plain_env = JaxBeamrider()
    _, fast_base = _make_fast_env()

    _, plain_state = plain_env.reset(jax.random.PRNGKey(9))
    _, fast_state = fast_base.reset(jax.random.PRNGKey(9))

    inactive = jnp.zeros((plain_env.consts.CHASING_METEOROID_MAX,), dtype=jnp.bool_)
    offscreen = plain_env.enemy_offscreen_meteoroids
    zeros_i = jnp.zeros((plain_env.consts.CHASING_METEOROID_MAX,), dtype=jnp.int32)
    zeros_f = jnp.zeros((plain_env.consts.CHASING_METEOROID_MAX,), dtype=jnp.float32)
    ones_i = jnp.ones((plain_env.consts.CHASING_METEOROID_MAX,), dtype=jnp.int32)

    def _with_active_wave(state):
        level = state.level._replace(
            player_pos=jnp.array(77.0, dtype=jnp.float32),
            white_ufo_left=jnp.array(0, dtype=jnp.int32),
            mothership_stage=jnp.array(2, dtype=jnp.int32),
            mothership_position=jnp.array(80.0, dtype=jnp.float32),
            chasing_meteoroid_pos=offscreen,
            chasing_meteoroid_active=inactive,
            chasing_meteoroid_vel_y=zeros_f,
            chasing_meteoroid_phase=zeros_i,
            chasing_meteoroid_frame=zeros_i,
            chasing_meteoroid_lane=zeros_i,
            chasing_meteoroid_side=ones_i,
            chasing_meteoroid_spawn_timer=jnp.array(0, dtype=jnp.int32),
            chasing_meteoroid_remaining=jnp.array(3, dtype=jnp.int32),
            chasing_meteoroid_wave_active=jnp.array(True),
        )
        return state._replace(level=level, sector=jnp.array(11, dtype=jnp.int32))

    plain_state = _with_active_wave(plain_state)
    fast_state = _with_active_wave(fast_state)
    key = jax.random.PRNGKey(10)

    plain_out = plain_env._chasing_meteoroid_step(
        plain_state,
        plain_state.level.player_pos,
        plain_state.level.player_vel,
        plain_state.level.white_ufo_left,
        key,
    )
    fast_out = fast_base._chasing_meteoroid_step(
        fast_state,
        fast_state.level.player_pos,
        fast_state.level.player_vel,
        fast_state.level.white_ufo_left,
        key,
    )

    plain_spawn_timer = int(plain_out[7])
    fast_spawn_timer = int(fast_out[7])

    assert int(jnp.sum(plain_out[1], dtype=jnp.int32)) == 1
    assert int(jnp.sum(fast_out[1], dtype=jnp.int32)) == 1
    assert fast_spawn_timer == max((plain_spawn_timer + 1) // 2, 1)


def test_beamrider_double_enemy_speed_enemy_shots_are_twice_as_fast():
    plain_env = JaxBeamrider()
    _, fast_base = _make_fast_env()

    _, plain_state = plain_env.reset(jax.random.PRNGKey(7))
    _, fast_state = fast_base.reset(jax.random.PRNGKey(7))

    shot_pos = plain_env.bullet_offscreen_shots.at[:, 0].set(jnp.array([81.0, 80.0], dtype=jnp.float32))
    shot_lane = jnp.array([3] + [0] * 8, dtype=jnp.int32)
    shot_timer = jnp.array([1] + [0] * 8, dtype=jnp.int32)

    plain_level = plain_state.level._replace(enemy_shot_pos=shot_pos, enemy_shot_vel=shot_lane, enemy_shot_timer=shot_timer)
    fast_level = fast_state.level._replace(enemy_shot_pos=shot_pos, enemy_shot_vel=shot_lane, enemy_shot_timer=shot_timer)
    plain_state = plain_state._replace(level=plain_level)
    fast_state = fast_state._replace(level=fast_level)

    empty_ufo_pos = plain_env.enemy_offscreen_ufo
    empty_patterns = jnp.zeros((3,), dtype=jnp.int32)
    empty_timers = jnp.zeros((3,), dtype=jnp.int32)

    plain_pos, _, _, _ = plain_env._enemy_shot_step(plain_state, empty_ufo_pos, empty_patterns, empty_timers)
    fast_pos, _, _, _ = fast_base._enemy_shot_step(fast_state, empty_ufo_pos, empty_patterns, empty_timers)

    plain_delta = float(plain_pos[1, 0] - shot_pos[1, 0])
    fast_delta = float(fast_pos[1, 0] - shot_pos[1, 0])

    assert fast_delta == pytest.approx(plain_delta * 2.0, rel=1e-5, abs=1e-5)


def test_beamrider_double_enemy_speed_handles_all_enemy_types_in_one_step():
    env, base = _make_fast_env()
    _, state = env.reset(jax.random.PRNGKey(2))

    level = state.level._replace(
        blue_line_counter=jnp.array(500, dtype=jnp.int32),
        standby_phase=jnp.array(0, dtype=jnp.int32),
        player_pos=jnp.array(77.0, dtype=jnp.float32),
        player_vel=jnp.array(0.0, dtype=jnp.float32),
        white_ufo_left=jnp.array(5, dtype=jnp.int32),
        white_ufo_pos=jnp.array(
            [
                [base.top_lanes_x[1], base.top_lanes_x[3], base.top_lanes_x[5]],
                [60.0, 60.0, 60.0],
            ],
            dtype=jnp.float32,
        ),
        white_ufo_vel=jnp.zeros((2, 3), dtype=jnp.float32),
        white_ufo_pattern_id=jnp.array(
            [
                int(WhiteUFOPattern.DROP_STRAIGHT),
                int(WhiteUFOPattern.SHOOT),
                int(WhiteUFOPattern.DROP_RIGHT),
            ],
            dtype=jnp.int32,
        ),
        white_ufo_pattern_timer=jnp.array([10, base.ufo_pattern_durations[int(WhiteUFOPattern.SHOOT)], 10], dtype=jnp.int32),
        enemy_shot_pos=base.bullet_offscreen_shots,
        enemy_shot_vel=jnp.zeros((9,), dtype=jnp.int32),
        enemy_shot_timer=jnp.zeros((9,), dtype=jnp.int32),
        chasing_meteoroid_pos=jnp.array(
            [
                [80.0] + [-100.0] * (base.consts.CHASING_METEOROID_MAX - 1),
                [54.0] + [-100.0] * (base.consts.CHASING_METEOROID_MAX - 1),
            ],
            dtype=jnp.float32,
        ),
        chasing_meteoroid_active=jnp.array([True] + [False] * (base.consts.CHASING_METEOROID_MAX - 1), dtype=jnp.bool_),
        chasing_meteoroid_vel_y=jnp.array([0.0] * base.consts.CHASING_METEOROID_MAX, dtype=jnp.float32),
        chasing_meteoroid_phase=jnp.array([0] * base.consts.CHASING_METEOROID_MAX, dtype=jnp.int32),
        chasing_meteoroid_frame=jnp.array([0] * base.consts.CHASING_METEOROID_MAX, dtype=jnp.int32),
        chasing_meteoroid_lane=jnp.array([0] * base.consts.CHASING_METEOROID_MAX, dtype=jnp.int32),
        chasing_meteoroid_side=jnp.array([1] * base.consts.CHASING_METEOROID_MAX, dtype=jnp.int32),
        rejuvenator_pos=jnp.array([80.0, 60.0], dtype=jnp.float32),
        rejuvenator_active=jnp.array(True),
        rejuvenator_dead=jnp.array(False),
        rejuvenator_lane=jnp.array(3, dtype=jnp.int32),
        falling_rock_pos=jnp.array(
            [
                [80.0] + [-100.0] * (base.consts.FALLING_ROCK_MAX - 1),
                [50.0] + [-100.0] * (base.consts.FALLING_ROCK_MAX - 1),
            ],
            dtype=jnp.float32,
        ),
        falling_rock_active=jnp.array([True] + [False] * (base.consts.FALLING_ROCK_MAX - 1), dtype=jnp.bool_),
        falling_rock_vel_y=jnp.array([base.consts.FALLING_ROCK_INIT_VEL] + [0.0] * (base.consts.FALLING_ROCK_MAX - 1), dtype=jnp.float32),
        falling_rock_lane=jnp.array([3] + [0] * (base.consts.FALLING_ROCK_MAX - 1), dtype=jnp.int32),
        lane_blocker_pos=jnp.array([[80.0], [120.0]], dtype=jnp.float32),
        lane_blocker_active=jnp.array([True], dtype=jnp.bool_),
        lane_blocker_vel_y=jnp.array([base.consts.LANE_BLOCKER_INIT_VEL], dtype=jnp.float32),
        lane_blocker_lane=jnp.array([3], dtype=jnp.int32),
        lane_blocker_phase=jnp.array([int(LaneBlockerState.DESCEND)], dtype=jnp.int32),
        lane_blocker_timer=jnp.array([0], dtype=jnp.int32),
        bouncer_pos=jnp.array([80.0, base.consts.BOUNCER_SPAWN_HEIGHT], dtype=jnp.float32),
        bouncer_vel=jnp.array([0.0, 0.0], dtype=jnp.float32),
        bouncer_state=jnp.array(int(BouncerState.DOWN), dtype=jnp.int32),
        bouncer_timer=jnp.array(1, dtype=jnp.int32),
        bouncer_active=jnp.array(True),
        bouncer_lane=jnp.array(3, dtype=jnp.int32),
        bouncer_step_index=jnp.array(0, dtype=jnp.int32),
        kamikaze_pos=jnp.array([[80.0], [70.0]], dtype=jnp.float32),
        kamikaze_active=jnp.array([True], dtype=jnp.bool_),
        kamikaze_lane=jnp.array([3], dtype=jnp.int32),
        kamikaze_vel_y=jnp.array([0.2], dtype=jnp.float32),
        kamikaze_tracking=jnp.array([False], dtype=jnp.bool_),
        kamikaze_spawn_timer=jnp.array([0], dtype=jnp.int32),
        coin_active=jnp.array([True] + [False] * (base.consts.COIN_MAX - 1), dtype=jnp.bool_),
        coin_pos=base.enemy_offscreen_coins.at[:, 0].set(jnp.array([72.0, 100.0], dtype=jnp.float32)),
        coin_timer=jnp.zeros((base.consts.COIN_MAX,), dtype=jnp.int32),
        coin_side=jnp.zeros((base.consts.COIN_MAX,), dtype=jnp.int32),
    )
    state = state._replace(level=level, sector=jnp.array(12, dtype=jnp.int32), steps=jnp.array(3001, dtype=jnp.int32))

    obs, new_state, reward, done, info = env.step(state, jnp.array(0, dtype=jnp.int32))

    assert obs is not None
    assert new_state is not None
    assert reward is not None
    assert done is not None
    assert info is not None
    assert bool(jnp.isfinite(new_state.level.player_pos))


def test_beamrider_double_enemy_speed_meteoroids_still_find_drop_lanes():
    env, base = _make_fast_env()
    _, state = env.reset(jax.random.PRNGKey(3))

    active = jnp.array([True] + [False] * (base.consts.CHASING_METEOROID_MAX - 1), dtype=jnp.bool_)
    pos = base.enemy_offscreen_meteoroids.at[:, 0].set(jnp.array([55.0, 54.0], dtype=jnp.float32))
    state = state._replace(
        level=state.level._replace(
            player_pos=base.bottom_lanes[0],
            chasing_meteoroid_pos=pos,
            chasing_meteoroid_active=active,
            chasing_meteoroid_phase=jnp.zeros((base.consts.CHASING_METEOROID_MAX,), dtype=jnp.int32),
            chasing_meteoroid_frame=jnp.zeros((base.consts.CHASING_METEOROID_MAX,), dtype=jnp.int32),
            chasing_meteoroid_lane=jnp.zeros((base.consts.CHASING_METEOROID_MAX,), dtype=jnp.int32),
            chasing_meteoroid_side=jnp.array([1] + [1] * (base.consts.CHASING_METEOROID_MAX - 1), dtype=jnp.int32),
            white_ufo_left=jnp.array(5, dtype=jnp.int32),
        ),
        sector=jnp.array(12, dtype=jnp.int32),
    )

    pos_out, _, _, phase_out, _, lane_out, _, _, _, _ = base._chasing_meteoroid_step(
        state,
        state.level.player_pos,
        state.level.player_vel,
        state.level.white_ufo_left,
        jax.random.PRNGKey(4),
    )

    assert float(pos_out[0, 0]) > 55.0
    assert int(phase_out[0]) == 2
    assert int(lane_out[0]) == 1

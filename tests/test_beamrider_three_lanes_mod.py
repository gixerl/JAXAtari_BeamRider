import jax
import jax.numpy as jnp

from jaxatari.core import make
from jaxatari.games.jax_beamrider import BouncerState, LaneBlockerState, WhiteUFOPattern
from jaxatari.games.mods.beamrider.beamrider_mod_plugins import THREE_LANE_GUIDE_POSITIONS


def _get_base_env(env):
    while hasattr(env, "_env"):
        env = env._env
    return env


def test_beamrider_three_lanes_mod_overrides_lane_geometry():
    env = make("beamrider", mods=["three_lanes"])
    base = _get_base_env(env)

    assert tuple(map(float, base.bottom_lanes.tolist())) == (0.0, 52.0, 77.0, 102.0, 154.0)
    assert tuple(map(float, base.top_lanes_x.tolist())) == (71.0, 71.0, 71.0, 81.0, 91.0, 91.0, 91.0)
    assert base.consts.LEFT_CLIP_PLAYER == 52
    assert base.consts.RIGHT_CLIP_PLAYER == 117


def test_beamrider_three_lanes_mod_enemy_shots_stay_on_center_track():
    env = make("beamrider", mods=["three_lanes"])
    base = _get_base_env(env)
    _, state = env.reset(jax.random.PRNGKey(0))

    white_ufo_pos = jnp.array(
        [
            [71.0, 81.0, 91.0],
            [70.0, 70.0, 70.0],
        ],
        dtype=jnp.float32,
    )
    shoot_timer = jnp.full((3,), base.ufo_pattern_durations[int(WhiteUFOPattern.SHOOT)], dtype=jnp.int32)
    shoot_pattern = jnp.full((3,), int(WhiteUFOPattern.SHOOT), dtype=jnp.int32)
    state = state._replace(steps=jnp.array(3001), ufo_killed=jnp.array(True))

    shot_pos, shot_lane, shot_timer_out, hit_count = base._enemy_shot_step(
        state,
        white_ufo_pos,
        shoot_pattern,
        shoot_timer,
    )

    active = shot_pos[1] <= float(base.consts.BOTTOM_CLIP)
    assert bool(jnp.all(jnp.isin(shot_lane[active], jnp.array([2, 3, 4], dtype=jnp.int32))))
    assert int(jnp.sum(active)) == 3
    assert int(hit_count) == 0
    assert bool(jnp.all(shot_timer_out[active] == 0))


def test_beamrider_three_lanes_mod_preserves_original_center_guide_markers():
    assert tuple(map(tuple, THREE_LANE_GUIDE_POSITIONS.tolist())) == (
        (72, 53),
        (70, 67),
        (68, 81),
        (66, 95),
        (65, 109),
        (63, 119),
        (62, 129),
        (61, 139),
        (60, 149),
        (58, 159),
        (83, 55),
        (83, 69),
        (83, 83),
        (83, 97),
        (83, 111),
        (83, 121),
        (83, 131),
        (83, 141),
        (83, 151),
        (83, 161),
        (94, 51),
        (96, 65),
        (98, 79),
        (99, 93),
        (101, 107),
        (102, 117),
        (104, 127),
        (105, 137),
        (106, 147),
        (107, 157),
    )


def test_beamrider_three_lanes_mod_full_step_handles_all_enemy_types():
    env = make("beamrider", mods=["three_lanes"])
    base = _get_base_env(env)
    _, state = env.reset(jax.random.PRNGKey(1))

    level = state.level._replace(
        blue_line_counter=jnp.array(500, dtype=jnp.int32),
        standby_phase=jnp.array(0, dtype=jnp.int32),
        player_pos=jnp.array(77.0, dtype=jnp.float32),
        player_vel=jnp.array(0.0, dtype=jnp.float32),
        white_ufo_left=jnp.array(5, dtype=jnp.int32),
        white_ufo_pos=jnp.array(
            [
                [71.0, 81.0, 91.0],
                [60.0, 60.0, 60.0],
            ],
            dtype=jnp.float32,
        ),
        white_ufo_vel=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        ),
        white_ufo_pattern_id=jnp.array(
            [
                int(WhiteUFOPattern.DROP_STRAIGHT),
                int(WhiteUFOPattern.SHOOT),
                int(WhiteUFOPattern.DROP_STRAIGHT),
            ],
            dtype=jnp.int32,
        ),
        white_ufo_pattern_timer=jnp.array([10, base.ufo_pattern_durations[int(WhiteUFOPattern.SHOOT)], 10], dtype=jnp.int32),
        enemy_shot_pos=base.bullet_offscreen_shots,
        enemy_shot_vel=jnp.zeros((9,), dtype=jnp.int32),
        enemy_shot_timer=jnp.zeros((9,), dtype=jnp.int32),
        chasing_meteoroid_pos=jnp.array(
            [
                [71.0] + [-100.0] * (base.consts.CHASING_METEOROID_MAX - 1),
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
        rejuvenator_pos=jnp.array([71.0, 60.0], dtype=jnp.float32),
        rejuvenator_active=jnp.array(True),
        rejuvenator_dead=jnp.array(False),
        rejuvenator_lane=jnp.array(2, dtype=jnp.int32),
        falling_rock_pos=jnp.array(
            [
                [71.0] + [-100.0] * (base.consts.FALLING_ROCK_MAX - 1),
                [50.0] + [-100.0] * (base.consts.FALLING_ROCK_MAX - 1),
            ],
            dtype=jnp.float32,
        ),
        falling_rock_active=jnp.array([True] + [False] * (base.consts.FALLING_ROCK_MAX - 1), dtype=jnp.bool_),
        falling_rock_vel_y=jnp.array([base.consts.FALLING_ROCK_INIT_VEL] + [0.0] * (base.consts.FALLING_ROCK_MAX - 1), dtype=jnp.float32),
        falling_rock_lane=jnp.array([2] + [0] * (base.consts.FALLING_ROCK_MAX - 1), dtype=jnp.int32),
        lane_blocker_pos=jnp.array([[71.0], [120.0]], dtype=jnp.float32),
        lane_blocker_active=jnp.array([True], dtype=jnp.bool_),
        lane_blocker_vel_y=jnp.array([base.consts.LANE_BLOCKER_INIT_VEL], dtype=jnp.float32),
        lane_blocker_lane=jnp.array([2], dtype=jnp.int32),
        lane_blocker_phase=jnp.array([int(LaneBlockerState.DESCEND)], dtype=jnp.int32),
        lane_blocker_timer=jnp.array([0], dtype=jnp.int32),
        bouncer_pos=jnp.array([71.0, base.consts.BOUNCER_SPAWN_HEIGHT], dtype=jnp.float32),
        bouncer_vel=jnp.array([0.0, 0.0], dtype=jnp.float32),
        bouncer_state=jnp.array(int(BouncerState.DOWN), dtype=jnp.int32),
        bouncer_timer=jnp.array(1, dtype=jnp.int32),
        bouncer_active=jnp.array(True),
        bouncer_lane=jnp.array(2, dtype=jnp.int32),
        bouncer_step_index=jnp.array(0, dtype=jnp.int32),
        kamikaze_pos=jnp.array([[91.0], [70.0]], dtype=jnp.float32),
        kamikaze_active=jnp.array([True], dtype=jnp.bool_),
        kamikaze_lane=jnp.array([4], dtype=jnp.int32),
        kamikaze_vel_y=jnp.array([0.2], dtype=jnp.float32),
        kamikaze_tracking=jnp.array([False], dtype=jnp.bool_),
        kamikaze_spawn_timer=jnp.array([0], dtype=jnp.int32),
        coin_active=jnp.array([False] * base.consts.COIN_MAX, dtype=jnp.bool_),
        coin_pos=base.enemy_offscreen_coins,
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

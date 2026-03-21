from functools import partial

import jax
import jax.numpy as jnp

from jaxatari.modification import JaxAtariInternalModPlugin
from jaxatari.games.jax_beamrider import (
    BLUE_LINE_INIT_TABLE,
    LaneBlockerState,
    WhiteUFOUpdate,
    WhiteUFOPattern,
    _get_index_bullet,
    _get_index_falling_rock,
    _get_index_kamikaze,
    _get_index_lane_blocker,
    _get_index_rejuvenator,
    _get_index_ufo,
    _get_player_shot_screen_x,
    _get_ufo_alignment,
)


def _get_white_ufo_mask(renderer, y_pos):
    white_ufo_masks = renderer.SHAPE_MASKS["white_ufo"]
    sprite_idx = jnp.clip(_get_index_ufo(y_pos) - 1, 0, white_ufo_masks.shape[0] - 1)
    return white_ufo_masks[sprite_idx]


def _render_enemy_explosion(renderer, raster, explosion_frame, explosion_pos):
    sprite_idx, y_offset = renderer._get_enemy_explosion_visuals(explosion_frame)
    sprite = renderer.SHAPE_MASKS["enemy_explosion"][sprite_idx]
    x_pos = explosion_pos[0] + _get_ufo_alignment(explosion_pos[1])
    y_pos = explosion_pos[1] + y_offset
    return renderer.jr.render_at_clipped(raster, x_pos, y_pos, sprite)


def _render_ufo_like_enemy(renderer, raster, x_pos, y_pos, active, align=True, mask=None):
    mask = _get_white_ufo_mask(renderer, y_pos) if mask is None else mask
    render_y = jnp.where(active, y_pos, 500.0)
    aligned_x = x_pos + _get_ufo_alignment(render_y) if align else x_pos
    render_x = jnp.where(active, aligned_x, 500.0)
    return renderer.jr.render_at_clipped(raster, render_x, render_y, mask)


def _render_mask_when_active(renderer, raster, x_pos, y_pos, active, mask, align=True):
    render_y = jnp.where(active, y_pos, 500.0)
    aligned_x = x_pos + _get_ufo_alignment(render_y) if align else x_pos
    render_x = jnp.where(active, aligned_x, 500.0)
    return renderer.jr.render_at_clipped(raster, render_x, render_y, mask)


MOTHERSHIP_LASER_SEGMENT_COUNT = 20
MOTHERSHIP_LASER_SEGMENT_SPACING = 1.0
MOTHERSHIP_LASER_SPEED = 0.7
MOTHERSHIP_LASER_DURATION = 200
MOTHERSHIP_LASER_SPRITE_IDX = 1
MOTHERSHIP_LASER_PHASE_MASK = 0xFF
MOTHERSHIP_LASER_FIRING_BIT = 1 << 8
MOTHERSHIP_LASER_POST_FIRE_BIT = 1 << 9
MOTHERSHIP_LASER_LANE_SHIFT = 10

TELEPORT_UFO_PATTERN_ID = 10
TELEPORT_UFO_PATTERN_DURATION = 42
TELEPORT_UFO_PATTERN_WEIGHT = 0.2
TELEPORT_UFO_TIMER_MASK = 0xFF
TELEPORT_UFO_USED_BIT = 1 << 8
TELEPORT_UFO_LANE_OFFSETS = jnp.array([-2, -1, 1, 2], dtype=jnp.int32)

THREE_LANE_TOP_IDS = jnp.array([2, 3, 4], dtype=jnp.int32)
THREE_LANE_TOP_LANES = jnp.array([71.0, 71.0, 71.0, 81.0, 91.0, 91.0, 91.0], dtype=jnp.float32)
THREE_LANE_BOTTOM_LANES = jnp.array([0.0, 52.0, 77.0, 102.0, 154.0], dtype=jnp.float32)
THREE_LANE_TOP_TO_BOTTOM = jnp.array(
    [
        (-0.52, 4.0),
        (-0.52, 4.0),
        (-0.52, 4.0),
        (0.0, 4.0),
        (0.52, 4.0),
        (0.52, 4.0),
        (0.52, 4.0),
    ],
    dtype=jnp.float32,
)
THREE_LANE_BOTTOM_TO_TOP = jnp.array(
    [
        (-0.52, 4.0),
        (-0.52, 4.0),
        (0.0, 4.0),
        (0.52, 4.0),
        (0.52, 4.0),
    ],
    dtype=jnp.float32,
)
THREE_LANE_LEFT_BOUND = 71.0
THREE_LANE_RIGHT_BOUND = 91.0
THREE_LANE_BACKGROUND_BLUE_RGB = (45, 109, 152)
THREE_LANE_BACKGROUND_HORIZON_Y = 45
THREE_LANE_GUIDE_MARKER_SIZE = jnp.array([[2, 1]], dtype=jnp.int32)
# Preserve the stock center-three background guide markers exactly; the mod
# only removes the outer lanes visually.
THREE_LANE_GUIDE_POSITIONS = jnp.array(
    [
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
    ],
    dtype=jnp.int32,
)
THREE_LANE_GUIDE_SIZES = jnp.tile(THREE_LANE_GUIDE_MARKER_SIZE, (THREE_LANE_GUIDE_POSITIONS.shape[0], 1))


def _get_lane_x(env, lane, y_pos):
    return env.top_lanes_x[lane] + env.lane_dx_over_dy[lane] * (y_pos - float(env.consts.TOP_CLIP))


def _canonical_three_lane_ufo_lane(lane: jnp.ndarray) -> jnp.ndarray:
    lane = lane.astype(jnp.int32)
    return jnp.where(lane <= 2, 2, jnp.where(lane >= 4, 4, 3))


def _is_three_lane_shootable(lane: jnp.ndarray) -> jnp.ndarray:
    lane = lane.astype(jnp.int32)
    return (lane >= 2) & (lane <= 4)


def _get_mothership_y(env):
    return float(env.consts.MOTHERSHIP_EMERGE_Y - env.consts.MOTHERSHIP_HEIGHT)


def _get_mothership_stop_x(env, lane):
    head_y = float(env.consts.MOTHERSHIP_EMERGE_Y) + 1.0
    laser_left_x = _get_lane_x(env, lane, head_y) + _get_ufo_alignment(head_y)
    ship_half_width = env.consts.MOTHERSHIP_SPRITE_SIZE[1] / 2.0
    laser_half_width = env.consts.ENEMY_SHOT_SPRITE_SIZES[MOTHERSHIP_LASER_SPRITE_IDX][1] / 2.0
    return (laser_left_x + laser_half_width - ship_half_width).astype(jnp.float32)


def _pack_mothership_laser_timer(lane, phase_timer, firing, post_fire):
    lane = lane.astype(jnp.int32)
    phase_timer = phase_timer.astype(jnp.int32) & jnp.int32(MOTHERSHIP_LASER_PHASE_MASK)
    firing_bits = jnp.where(firing, jnp.int32(MOTHERSHIP_LASER_FIRING_BIT), jnp.int32(0))
    post_fire_bits = jnp.where(post_fire, jnp.int32(MOTHERSHIP_LASER_POST_FIRE_BIT), jnp.int32(0))
    return (lane << MOTHERSHIP_LASER_LANE_SHIFT) | post_fire_bits | firing_bits | phase_timer


def _get_mothership_laser_phase_timer(timer):
    return timer.astype(jnp.int32) & jnp.int32(MOTHERSHIP_LASER_PHASE_MASK)


def _get_mothership_laser_lane_from_timer(timer):
    lane = timer.astype(jnp.int32) >> MOTHERSHIP_LASER_LANE_SHIFT
    return jnp.clip(lane, 1, 5)


def _is_mothership_laser_firing(timer):
    return (timer.astype(jnp.int32) & jnp.int32(MOTHERSHIP_LASER_FIRING_BIT)) != 0


def _has_mothership_laser_post_fire(timer):
    return (timer.astype(jnp.int32) & jnp.int32(MOTHERSHIP_LASER_POST_FIRE_BIT)) != 0


def _get_mothership_laser_lane(env, state):
    return _get_mothership_laser_lane_from_timer(state.level.mothership_timer)


def _get_mothership_laser_segments(env, state):
    timer = state.level.mothership_timer.astype(jnp.int32)
    phase_timer = _get_mothership_laser_phase_timer(timer)
    laser_active = (state.level.mothership_stage.astype(jnp.int32) == 2) & _is_mothership_laser_firing(timer) & (
        phase_timer < MOTHERSHIP_LASER_DURATION
    )
    lane = _get_mothership_laser_lane(env, state)
    segment_idx = jnp.arange(MOTHERSHIP_LASER_SEGMENT_COUNT, dtype=jnp.float32)
    head_y = jnp.array(float(env.consts.MOTHERSHIP_EMERGE_Y) + 1.0, dtype=jnp.float32)
    head_y = head_y + phase_timer.astype(jnp.float32) * MOTHERSHIP_LASER_SPEED
    seg_y = head_y + (segment_idx * MOTHERSHIP_LASER_SEGMENT_SPACING)
    seg_x = _get_lane_x(env, lane, seg_y) + _get_ufo_alignment(seg_y)
    visible = laser_active & (seg_y <= float(env.consts.BOTTOM_CLIP)) & (seg_y >= float(env.consts.TOP_CLIP) - 8.0)
    return seg_x, seg_y, visible


def _laser_hits_box(env, state, left_x, top_y, width, height):
    seg_x, seg_y, visible = _get_mothership_laser_segments(env, state)
    seg_size = env.enemy_shot_sprite_sizes[MOTHERSHIP_LASER_SPRITE_IDX]
    seg_h = seg_size[0]
    seg_w = seg_size[1]
    hits = visible & (
        (seg_x < left_x + width)
        & (left_x < seg_x + seg_w)
        & (seg_y - env.consts.ENEMY_HITBOX_TOP_EXTENSION < top_y + height)
        & (top_y < seg_y + seg_h)
    )
    return jnp.any(hits)


def _is_teleport_ufo_pattern(pattern_id):
    return pattern_id.astype(jnp.int32) == jnp.int32(TELEPORT_UFO_PATTERN_ID)


def _get_teleport_ufo_remaining(timer):
    return timer.astype(jnp.int32) & jnp.int32(TELEPORT_UFO_TIMER_MASK)


def _has_teleport_ufo_been_used(timer):
    return (timer.astype(jnp.int32) & jnp.int32(TELEPORT_UFO_USED_BIT)) != 0


def _pack_teleport_ufo_timer(remaining, used):
    remaining = remaining.astype(jnp.int32) & jnp.int32(TELEPORT_UFO_TIMER_MASK)
    used_bits = jnp.where(used, jnp.int32(TELEPORT_UFO_USED_BIT), jnp.int32(0))
    return remaining | used_bits


def _get_teleport_ufo_lane_bounds_from_stage(stage):
    in_restricted_stage = stage.astype(jnp.int32) >= 6
    min_lane = jnp.where(in_restricted_stage, 1, 0)
    max_lane = jnp.where(in_restricted_stage, 5, 6)
    return min_lane.astype(jnp.int32), max_lane.astype(jnp.int32)


def _get_teleport_ufo_lane_bounds_from_y(y_pos):
    in_restricted_stage = y_pos.astype(jnp.float32) >= 86.0
    min_lane = jnp.where(in_restricted_stage, 1, 0)
    max_lane = jnp.where(in_restricted_stage, 5, 6)
    return min_lane.astype(jnp.int32), max_lane.astype(jnp.int32)


def _get_teleport_ufo_candidate_lanes(current_lane, min_lane, max_lane):
    current_lane = current_lane.astype(jnp.int32)
    candidates = current_lane + TELEPORT_UFO_LANE_OFFSETS
    valid = (candidates >= min_lane) & (candidates <= max_lane)
    return candidates.astype(jnp.int32), valid


def _init_white_ufo_pattern_timer(pattern, duration):
    is_triple = (pattern == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)) | (pattern == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT))
    is_teleport = _is_teleport_ufo_pattern(pattern)
    timer = jnp.where(is_triple, duration | (15 << 3), duration)
    return jnp.where(is_teleport, _pack_teleport_ufo_timer(duration, jnp.array(False)), timer)


def _get_fog_of_war_top_y(state, consts) -> jnp.ndarray:
    active_lines = state.level.line_positions >= 0
    safe_line_positions = jnp.where(
        active_lines,
        state.level.line_positions.astype(jnp.int32),
        jnp.array(consts.BLUE_LINE_OFFSCREEN_Y, dtype=jnp.int32),
    )
    top_line_y = jnp.where(
        jnp.any(active_lines),
        jnp.min(safe_line_positions),
        jnp.array(consts.MIN_BLUE_LINE_POS, dtype=jnp.int32),
    )
    top_dot_ufo_visible_limit = jnp.array(48, dtype=jnp.int32)
    return jnp.maximum(top_line_y - 1, top_dot_ufo_visible_limit)


def _get_fog_of_war_cutoff_y(fog_top, consts) -> jnp.ndarray:
    player_y = jnp.array(consts.PLAYER_POS_Y, dtype=jnp.int32)
    visible_band_height = jnp.maximum(player_y - fog_top, 0)
    return player_y - jnp.floor_divide(visible_band_height, 3)


def _triangle_wave(values, period):
    phase = jnp.mod(values, period)
    half_period = period // 2
    return jnp.abs(phase - half_period)


def _apply_fog_of_war(renderer, raster, state):
    fog_top = _get_fog_of_war_top_y(state, renderer.consts)
    fog_cutoff = _get_fog_of_war_cutoff_y(fog_top, renderer.consts)
    fog_height = jnp.maximum(fog_cutoff - fog_top, 1)
    fog_left = jnp.array(8, dtype=jnp.int32)
    fog_right = fog_left + jnp.array(renderer.SHAPE_MASKS["blue_line"].shape[1], dtype=jnp.int32)
    xx = renderer.jr._xx
    yy = renderer.jr._yy
    rel_y = jnp.clip(yy - fog_top, 0, fog_height)
    in_fog_band = (yy >= fog_top) & (yy < fog_cutoff) & (xx >= fog_left) & (xx < fog_right)

    black_id = jnp.array(renderer.COLOR_TO_ID[(0, 0, 0)], dtype=raster.dtype)
    shadow_id = jnp.array(renderer.COLOR_TO_ID[(80, 0, 132)], dtype=raster.dtype)
    body_id = jnp.array(renderer.COLOR_TO_ID[(104, 25, 154)], dtype=raster.dtype)
    edge_id = jnp.array(renderer.COLOR_TO_ID[(45, 109, 152)], dtype=raster.dtype)

    local_x = xx - fog_left
    phase_fast = state.steps // 16
    phase_slow = state.steps // 27
    wave_a = _triangle_wave(local_x + phase_fast, 42) // 7
    wave_b = _triangle_wave((local_x * 3) + phase_slow, 64) // 11
    wave_c = _triangle_wave((local_x * 5) + phase_fast, 96) // 16
    boundary_offset = jnp.clip(wave_a - wave_b + wave_c - 2, -3, 4)
    local_cutoff = fog_cutoff + boundary_offset
    depth_to_edge = local_cutoff - yy

    in_fog_band = in_fog_band & (yy < local_cutoff)

    wave_scroll = state.steps // 18
    wave_band = jnp.mod(depth_to_edge + wave_scroll, 14)
    fill_id = jnp.where(
        depth_to_edge <= 3,
        edge_id,
        jnp.where(
            wave_band < 3,
            edge_id,
            jnp.where(wave_band < 7, body_id, jnp.where(wave_band < 10, shadow_id, black_id)),
        ),
    )

    boundary_noise = jnp.mod((local_x // 6) + (yy // 3) + (state.steps // 19), 7)
    edge_shadow = (boundary_noise <= 1) & (depth_to_edge <= 4) & (depth_to_edge >= 2)
    fill_id = jnp.where(edge_shadow, shadow_id, fill_id)

    return jnp.where(in_fog_band, fill_id, raster)


class FogOfWarMod(JaxAtariInternalModPlugin):
    """Hide the upper two thirds of the playfield behind opaque fog."""

    @partial(jax.jit, static_argnums=(0,))
    def _render_mothership(self, raster, state):
        renderer = self._env.renderer
        raster = type(renderer)._render_mothership(renderer, raster, state)
        return _apply_fog_of_war(renderer, raster, state)


class HardcoreMod(JaxAtariInternalModPlugin):
    """Start with one life and never allow extra lives."""

    constants_overrides = {
        "STARTING_LIVES": 1,
        "MAX_LIVES": 1,
    }


class ThreeLanesMod(JaxAtariInternalModPlugin):
    """Collapse Beamrider's track to the center three lanes."""

    constants_overrides = {
        "LEFT_CLIP_PLAYER": 52,
        "RIGHT_CLIP_PLAYER": 117,
    }

    attribute_overrides = {
        "bottom_lanes": THREE_LANE_BOTTOM_LANES,
        "top_lanes_x": THREE_LANE_TOP_LANES,
        "lane_vectors_t2b": THREE_LANE_TOP_TO_BOTTOM,
        "lane_vectors_b2t": THREE_LANE_BOTTOM_TO_TOP,
        "lane_dx_over_dy": THREE_LANE_TOP_TO_BOTTOM[:, 0] / THREE_LANE_TOP_TO_BOTTOM[:, 1],
        "middle_lane_spawn": THREE_LANE_TOP_IDS,
    }

    @partial(jax.jit, static_argnums=(0,))
    def _render_colored_background(self, raster, state):
        renderer = self._env.renderer
        raster = type(renderer)._render_colored_background(renderer, raster, state)

        blue_id = renderer.COLOR_TO_ID[THREE_LANE_BACKGROUND_BLUE_RGB]
        clear_mask = (renderer.jr._yy > THREE_LANE_BACKGROUND_HORIZON_Y) & (raster == blue_id)
        row_background = raster[:, :1]
        raster = jnp.where(clear_mask, row_background, raster)

        return renderer.jr.draw_rects(
            raster,
            THREE_LANE_GUIDE_POSITIONS,
            THREE_LANE_GUIDE_SIZES,
            blue_id,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _white_ufo_choose_pattern(
        self,
        key,
        *,
        allow_shoot,
        prev_pattern,
        is_kamikaze_zone,
        sector,
        stage,
        lane,
        is_on_lane,
    ):
        pattern_choices = jnp.array(
            [
                int(WhiteUFOPattern.DROP_STRAIGHT),
                int(WhiteUFOPattern.DROP_LEFT),
                int(WhiteUFOPattern.DROP_RIGHT),
                int(WhiteUFOPattern.SHOOT),
                int(WhiteUFOPattern.MOVE_BACK),
                int(WhiteUFOPattern.KAMIKAZE),
                int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT),
                int(WhiteUFOPattern.TRIPLE_SHOT_LEFT),
            ],
            dtype=jnp.int32,
        )
        pattern_probs = self._env.ufo_pattern_probs

        is_move_back = prev_pattern == int(WhiteUFOPattern.MOVE_BACK)
        chain_mask = jnp.ones_like(pattern_probs).at[0].set(jnp.where(is_move_back, 0.0, 1.0))
        pattern_probs = pattern_probs * chain_mask

        shoot_mask = jnp.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        pattern_probs = jnp.where(allow_shoot, pattern_probs, pattern_probs * shoot_mask)

        kamikaze_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0], dtype=jnp.float32)
        pattern_probs = jnp.where(is_kamikaze_zone, pattern_probs, pattern_probs * kamikaze_mask)

        move_back_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        pattern_probs = jnp.where(stage >= 4, pattern_probs, pattern_probs * move_back_mask)

        can_triple = (sector >= 7) & (stage >= 4) & (stage <= 6) & is_on_lane
        can_triple_right = can_triple & (lane >= 2) & (lane <= 3)
        can_triple_left = can_triple & (lane >= 3) & (lane <= 4)

        triple_right_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0], dtype=jnp.float32)
        triple_left_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
        pattern_probs = jnp.where(can_triple_right, pattern_probs, pattern_probs * triple_right_mask)
        pattern_probs = jnp.where(can_triple_left, pattern_probs, pattern_probs * triple_left_mask)

        prob_sum = jnp.sum(pattern_probs)
        pattern_probs = jnp.where(prob_sum > 0, pattern_probs / prob_sum, pattern_probs)

        pattern = jax.random.choice(key, pattern_choices, shape=(), p=pattern_probs)
        duration = self._env.ufo_pattern_durations[pattern]
        return pattern, duration

    @partial(jax.jit, static_argnums=(0,))
    def _white_ufo_update_pattern_state(
        self,
        sector,
        position,
        time_on_lane,
        attack_time,
        already_left,
        spawn_delay,
        pattern_id,
        pattern_timer,
        key,
    ):
        on_top_lane = position[1] <= self._env.consts.TOP_CLIP
        time_on_lane = jnp.where(on_top_lane, time_on_lane + 1, 0)
        attack_time = jnp.where(on_top_lane, 0, attack_time)

        ufo_x = position[0].astype(jnp.float32)
        ufo_y = position[1].astype(jnp.float32)
        lane_x_at_ufo_y = self._env.top_lanes_x + self._env.lane_dx_over_dy * (ufo_y - float(self._env.consts.TOP_CLIP))
        raw_lane_id = jnp.argmin(jnp.abs(lane_x_at_ufo_y - ufo_x)).astype(jnp.int32)
        closest_lane_id = _canonical_three_lane_ufo_lane(raw_lane_id)
        closest_lane_x = lane_x_at_ufo_y[closest_lane_id]
        dist_to_lane = jnp.abs(closest_lane_x - ufo_x)
        is_on_lane = dist_to_lane <= 0.25

        is_triple = (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)) | (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT))

        shots_left = pattern_timer & 7
        last_lane = (pattern_timer >> 3) & 15
        shoot_now = (pattern_timer >> 7) & 1

        def update_triple(_):
            can_shoot = (shots_left > 0) & is_on_lane & (closest_lane_id != last_lane)
            new_shoot_now = jnp.where(shoot_now == 1, 0, jnp.where(can_shoot, 1, 0))
            new_shots_left = jnp.where(can_shoot, shots_left - 1, shots_left)
            new_last_lane = jnp.where(can_shoot, closest_lane_id, last_lane)
            return (new_shoot_now << 7) | (new_last_lane << 3) | new_shots_left

        pattern_timer = jnp.where(
            is_triple,
            update_triple(None),
            jnp.maximum(pattern_timer - 1, jnp.zeros_like(pattern_timer)),
        )

        allow_shoot = (~on_top_lane) & _is_three_lane_shootable(closest_lane_id)

        is_drop_pattern = (
            (pattern_id == int(WhiteUFOPattern.DROP_STRAIGHT))
            | (pattern_id == int(WhiteUFOPattern.DROP_LEFT))
            | (pattern_id == int(WhiteUFOPattern.DROP_RIGHT))
            | (pattern_id == int(WhiteUFOPattern.MOVE_BACK))
        )
        is_shoot_pattern = pattern_id == int(WhiteUFOPattern.SHOOT)
        is_engagement_pattern = is_drop_pattern | is_shoot_pattern | is_triple
        attack_time = jnp.where((~on_top_lane) & is_engagement_pattern, attack_time + 1, attack_time)

        is_retreat = pattern_id == int(WhiteUFOPattern.RETREAT)
        is_move_back = pattern_id == int(WhiteUFOPattern.MOVE_BACK)
        movement_finished = (is_retreat | is_move_back) & on_top_lane
        pattern_id = jnp.where(movement_finished, int(WhiteUFOPattern.IDLE), pattern_id)
        pattern_timer = jnp.where(movement_finished, 0, pattern_timer)
        attack_time = jnp.where(movement_finished, 0, attack_time)

        triple_finished = is_triple & ((pattern_timer & 7) == 0) & jnp.logical_not((pattern_timer >> 7) & 1) & is_on_lane

        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT), 1, 0)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT), -1, lane_offset)
        target_lane_id = jnp.clip(closest_lane_id + lane_offset, 2, 4)

        triple_stuck = is_triple & is_on_lane & (shots_left > 0) & (target_lane_id == closest_lane_id) & (closest_lane_id == last_lane)
        triple_finished = triple_finished | triple_stuck

        pattern_finished_off_top = (~on_top_lane) & is_engagement_pattern & jnp.where(is_triple, triple_finished, pattern_timer == 0) & is_on_lane

        key_start_roll, key_start_choice, key_retreat_roll, key_chain_choice, _ = jax.random.split(key, 5)
        retreat_roll = jax.random.uniform(key_retreat_roll)
        retreat_prob = self._env._white_ufo_retreat_prob(attack_time)
        retreat_now = pattern_finished_off_top & (retreat_roll < retreat_prob)
        pattern_id = jnp.where(retreat_now, int(WhiteUFOPattern.RETREAT), pattern_id)
        pattern_timer = jnp.where(retreat_now, self._env.consts.WHITE_UFO_RETREAT_DURATION, pattern_timer)
        attack_time = jnp.where(retreat_now, 0, attack_time)

        chain_next = pattern_finished_off_top & (~retreat_now)
        ufo_stage = _get_index_ufo(position[1])

        def choose_chain_pattern(_):
            is_kamikaze_zone = position[1] >= self._env.consts.KAMIKAZE_Y_THRESHOLD
            pattern, duration = self._white_ufo_choose_pattern(
                key_chain_choice,
                allow_shoot=allow_shoot,
                prev_pattern=pattern_id,
                is_kamikaze_zone=is_kamikaze_zone,
                sector=sector,
                stage=ufo_stage,
                lane=closest_lane_id,
                is_on_lane=is_on_lane,
            )
            return pattern, _init_white_ufo_pattern_timer(pattern, duration)

        pattern_id, pattern_timer = jax.lax.cond(
            chain_next,
            choose_chain_pattern,
            lambda _: (pattern_id, pattern_timer),
            operand=None,
        )

        should_choose_new = on_top_lane & (pattern_id == int(WhiteUFOPattern.IDLE)) & (pattern_timer == 0) & (spawn_delay == 0)
        p_start = type(self._env).entropy_heat_prob_static(
            jnp.where(already_left, time_on_lane * 10, time_on_lane),
            alpha=self._env.consts.WHITE_UFO_ATTACK_ALPHA,
            p_min=jnp.where(already_left, 0.1, self._env.consts.WHITE_UFO_ATTACK_P_MIN),
            p_max=self._env.consts.WHITE_UFO_ATTACK_P_MAX,
        )
        start_roll = jax.random.uniform(key_start_roll)
        start_attack = should_choose_new & (start_roll < p_start)

        def choose_new_pattern(_):
            pattern, duration = self._white_ufo_choose_pattern(
                key_start_choice,
                allow_shoot=jnp.array(False),
                prev_pattern=pattern_id,
                is_kamikaze_zone=jnp.array(False),
                sector=sector,
                stage=ufo_stage,
                lane=closest_lane_id,
                is_on_lane=is_on_lane,
            )
            return pattern, _init_white_ufo_pattern_timer(pattern, duration)

        pattern_id, pattern_timer = jax.lax.cond(
            start_attack,
            choose_new_pattern,
            lambda _: (pattern_id, pattern_timer),
            operand=None,
        )

        return pattern_id, pattern_timer, time_on_lane, attack_time

    @partial(jax.jit, static_argnums=(0,))
    def _white_ufo_top_lane(self, white_ufo_pos, white_ufo_vel_x, pattern_id, key):
        hold_position = (pattern_id == int(WhiteUFOPattern.SHOOT)) | (white_ufo_pos[1] > float(self._env.consts.TOP_CLIP))
        min_speed = float(self._env.consts.WHITE_UFO_TOP_LANE_MIN_SPEED)
        turn_speed = float(self._env.consts.WHITE_UFO_TOP_LANE_TURN_SPEED)

        vx = jnp.where(hold_position, 0.0, white_ufo_vel_x)
        need_boost = (~hold_position) & (jnp.abs(vx) < min_speed)
        random_sign = jnp.where(jax.random.uniform(key) < 0.5, -1.0, 1.0)
        direction = jnp.where(vx == 0.0, random_sign, jnp.sign(vx))
        vx = jnp.where(need_boost, direction * min_speed, vx)

        do_bounce = ~hold_position
        vx = jnp.where(do_bounce & (white_ufo_pos[0] >= THREE_LANE_RIGHT_BOUND), -turn_speed, vx)
        vx = jnp.where(do_bounce & (white_ufo_pos[0] <= THREE_LANE_LEFT_BOUND), turn_speed, vx)
        return vx, 0.0

    @partial(jax.jit, static_argnums=(0,))
    def _white_ufo_normal(self, white_ufo_pos, white_ufo_vel_x, white_ufo_vel_y, pattern_id, already_left):
        speed_factor = self._env.consts.WHITE_UFO_SPEED_FACTOR
        retreat_mult = self._env.consts.WHITE_UFO_RETREAT_SPEED_MULT
        x, y = white_ufo_pos[0], white_ufo_pos[1]

        lane_x_at_y = self._env.top_lanes_x + self._env.lane_dx_over_dy * (y - float(self._env.consts.TOP_CLIP))
        raw_lane_id = jnp.argmin(jnp.abs(lane_x_at_y - x))
        closest_lane_id = _canonical_three_lane_ufo_lane(raw_lane_id)

        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.DROP_RIGHT), 1, 0)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.DROP_LEFT), -1, lane_offset)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT), 1, lane_offset)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT), -1, lane_offset)
        target_lane_id = jnp.clip(closest_lane_id + lane_offset, 2, 4)

        lane_vector = self._env.lane_vectors_t2b[target_lane_id]
        target_lane_x = lane_x_at_y[target_lane_id]

        is_retreat = pattern_id == int(WhiteUFOPattern.RETREAT)
        is_move_back = pattern_id == int(WhiteUFOPattern.MOVE_BACK)
        is_kamikaze = pattern_id == int(WhiteUFOPattern.KAMIKAZE)
        is_triple = (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)) | (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT))

        cross_track = target_lane_x - x
        distance_to_lane = jnp.abs(cross_track)
        direction = jnp.sign(cross_track)

        def seek_lane(_):
            attack_vx = jnp.where(direction == 0, 0.0, direction * 0.5)
            retreat_vx = jnp.where(direction == 0, 0.0, direction * speed_factor * retreat_mult * 2.0)
            new_vx = jnp.where(is_retreat | is_kamikaze | is_triple, retreat_vx, attack_vx)

            normal_vy = 0.25
            retreat_vy = -lane_vector[1] * speed_factor * retreat_mult
            move_back_vy = -lane_vector[1] * speed_factor
            kamikaze_vy = lane_vector[1] * speed_factor * retreat_mult
            triple_vy = 0.25

            new_vy = jnp.where(is_retreat, retreat_vy, normal_vy)
            new_vy = jnp.where(is_move_back, move_back_vy, new_vy)
            new_vy = jnp.where(is_kamikaze, kamikaze_vy, new_vy)
            new_vy = jnp.where(is_triple, triple_vy, new_vy)
            return new_vx, new_vy

        def follow_lane(_):
            normal_vx = lane_vector[0] * speed_factor
            normal_vy = lane_vector[1] * speed_factor

            retreat_vx = -lane_vector[0] * speed_factor * retreat_mult
            retreat_vy = -lane_vector[1] * speed_factor * retreat_mult

            move_back_vx = -lane_vector[0] * speed_factor
            move_back_vy = -lane_vector[1] * speed_factor

            kamikaze_vx = lane_vector[0] * speed_factor * retreat_mult
            kamikaze_vy = lane_vector[1] * speed_factor * retreat_mult

            triple_vy = 0.25

            new_vx = jnp.where(is_retreat, retreat_vx, jnp.where(is_move_back, move_back_vx, normal_vx))
            new_vx = jnp.where(is_kamikaze, kamikaze_vx, new_vx)

            new_vy = jnp.where(is_retreat, retreat_vy, jnp.where(is_move_back, move_back_vy, normal_vy))
            new_vy = jnp.where(is_kamikaze, kamikaze_vy, new_vy)
            new_vy = jnp.where(is_triple, triple_vy, new_vy)
            return new_vx, new_vy

        return jax.lax.cond(distance_to_lane <= 0.25, follow_lane, seek_lane, operand=None)

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_shot_step(self, state, white_ufo_pos, white_ufo_pattern_id, white_ufo_pattern_timer):
        lane_vectors = self._env.lane_vectors_t2b
        lanes_top_x = self._env.top_lanes_x
        lane_dx_over_dy = self._env.lane_dx_over_dy

        offscreen = self._env.bullet_offscreen_shots

        shot_pos = state.level.enemy_shot_pos.astype(jnp.float32)
        shot_lane = state.level.enemy_shot_vel.astype(jnp.int32)
        shot_timer = state.level.enemy_shot_timer.astype(jnp.int32)

        shot_active = shot_pos[1] <= float(self._env.consts.BOTTOM_CLIP)
        shot_timer = jnp.where(shot_active, shot_timer + 1, 0)

        shoot_duration = self._env.ufo_pattern_durations[int(WhiteUFOPattern.SHOOT)]
        is_triple = (white_ufo_pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)) | (white_ufo_pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT))
        shoot_now_triple = (white_ufo_pattern_timer >> 7) & 1

        wants_spawn = (white_ufo_pattern_id == int(WhiteUFOPattern.SHOOT)) & (white_ufo_pattern_timer == shoot_duration)
        wants_spawn = wants_spawn | (is_triple & (shoot_now_triple == 1))

        ufo_on_screen = white_ufo_pos[1] <= float(self._env.consts.BOTTOM_CLIP)
        ufo_not_on_top_lane = white_ufo_pos[1] > float(self._env.consts.TOP_CLIP)
        ufo_x = white_ufo_pos[0].astype(jnp.float32)
        ufo_y = white_ufo_pos[1].astype(jnp.float32)

        lane_x_at_ufo_y = lanes_top_x[:, None] + lane_dx_over_dy[:, None] * (ufo_y[None, :] - float(self._env.consts.TOP_CLIP))
        raw_closest_lane = jnp.argmin(jnp.abs(lane_x_at_ufo_y - ufo_x[None, :]), axis=0).astype(jnp.int32)
        closest_lane = _canonical_three_lane_ufo_lane(raw_closest_lane)
        allowed_shot_lane = _is_three_lane_shootable(closest_lane)

        ufo_shot_active = jnp.reshape(shot_active, (3, 3))
        first_inactive_slot = jnp.argmax(jnp.logical_not(ufo_shot_active), axis=1)
        has_inactive_slot = jnp.any(jnp.logical_not(ufo_shot_active), axis=1)

        can_shoot = (state.steps > 2000) | state.ufo_killed
        spawn = jnp.logical_and.reduce(
            jnp.array(
                [
                    wants_spawn,
                    ufo_on_screen,
                    ufo_not_on_top_lane,
                    allowed_shot_lane,
                    has_inactive_slot,
                ]
            )
        ) & can_shoot

        spawn_y = jnp.clip(ufo_y + 4.0, float(self._env.consts.TOP_CLIP), float(self._env.consts.BOTTOM_CLIP))
        spawn_x = jnp.take(lanes_top_x, closest_lane) + jnp.take(lane_dx_over_dy, closest_lane) * (
            spawn_y - float(self._env.consts.TOP_CLIP)
        )

        spawn_slots = jnp.arange(3) * 3 + first_inactive_slot
        spawn_mask = (jax.nn.one_hot(spawn_slots, 9) * spawn[:, None]).sum(axis=0).astype(jnp.bool_)

        spawn_x_expanded = jnp.repeat(spawn_x, 3)
        spawn_y_expanded = jnp.repeat(spawn_y, 3)
        spawn_pos_expanded = jnp.stack([spawn_x_expanded, spawn_y_expanded])

        shot_pos = jnp.where(spawn_mask[None, :], spawn_pos_expanded, shot_pos)

        closest_lane_expanded = jnp.repeat(closest_lane, 3)
        shot_lane = jnp.where(spawn_mask, closest_lane_expanded, shot_lane)
        shot_timer = jnp.where(spawn_mask, 0, shot_timer)
        shot_active = jnp.logical_or(shot_active, spawn_mask)

        should_move = shot_active & ((shot_timer % 4) == 2)
        speed = float(self._env.consts.WHITE_UFO_SHOT_SPEED_FACTOR)
        lane_dy = jnp.take(lane_vectors[:, 1], shot_lane)
        y_after = shot_pos[1] + jnp.where(should_move, lane_dy * speed, 0.0)
        x_after = jnp.take(lanes_top_x, shot_lane) + jnp.take(lane_dx_over_dy, shot_lane) * (
            y_after - float(self._env.consts.TOP_CLIP)
        )
        shot_pos = jnp.where(shot_active, jnp.stack([x_after, y_after]), shot_pos)

        moved_offscreen = shot_pos[1] > float(self._env.consts.BOTTOM_CLIP)
        shot_pos = jnp.where(moved_offscreen, offscreen, shot_pos)
        shot_timer = jnp.where(moved_offscreen, 0, shot_timer)
        shot_active = shot_active & (~moved_offscreen)

        player_left = state.level.player_pos.astype(jnp.float32)
        player_y = float(self._env.consts.PLAYER_POS_Y)
        player_size = self._env.player_sprite_size

        shot_x = shot_pos[0] + _get_ufo_alignment(shot_pos[1])
        shot_y = shot_pos[1]

        sprite_idx = (jnp.floor_divide(shot_timer, 4) % 2).astype(jnp.int32)
        shot_sizes = jnp.take(self._env.enemy_shot_sprite_sizes, sprite_idx, axis=0)

        hits = (
            shot_active
            & (shot_x < player_left + player_size[1])
            & (player_left < shot_x + shot_sizes[:, 1])
            & (shot_y < player_y + player_size[0])
            & (player_y < shot_y + shot_sizes[:, 0])
        )

        hit_count = jnp.sum(hits, dtype=jnp.int32)
        shot_pos = jnp.where(hits[None, :], offscreen, shot_pos)
        shot_timer = jnp.where(hits, 0, shot_timer)
        return shot_pos, shot_lane, shot_timer, hit_count


class SameEnemiesMod(JaxAtariInternalModPlugin):
    """Render all Beamrider enemy types using the white UFO visuals."""

    @partial(jax.jit, static_argnums=(0,))
    def _render_bouncer(self, raster, state):
        renderer = self._env.renderer
        explosion_frame = state.level.bouncer_explosion_frame

        def render_explosion(r_in):
            return _render_enemy_explosion(
                renderer,
                r_in,
                explosion_frame,
                state.level.bouncer_explosion_pos,
            )

        def render_active_bouncer(r_in):
            return _render_ufo_like_enemy(
                renderer,
                r_in,
                state.level.bouncer_pos[0],
                state.level.bouncer_pos[1],
                state.level.bouncer_active,
                align=True,
            )

        return jax.lax.cond(
            explosion_frame > 0,
            render_explosion,
            render_active_bouncer,
            raster,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_chasing_meteoroids(self, raster, state):
        renderer = self._env.renderer

        def body_fun(raster, idx):
            explosion_frame = state.level.chasing_meteoroid_explosion_frame[idx]

            def render_explosion(r_in):
                return _render_enemy_explosion(
                    renderer,
                    r_in,
                    explosion_frame,
                    state.level.chasing_meteoroid_explosion_pos[:, idx],
                )

            def render_enemy(r_in):
                return _render_ufo_like_enemy(
                    renderer,
                    r_in,
                    state.level.chasing_meteoroid_pos[0][idx],
                    state.level.chasing_meteoroid_pos[1][idx],
                    state.level.chasing_meteoroid_active[idx],
                    align=True,
                )

            new_raster = jax.lax.cond(
                explosion_frame > 0,
                render_explosion,
                render_enemy,
                raster,
            )
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(self._env.consts.CHASING_METEOROID_MAX))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_falling_rocks(self, raster, state):
        renderer = self._env.renderer

        def body_fun(raster, idx):
            explosion_frame = state.level.falling_rock_explosion_frame[idx]

            def render_explosion(r_in):
                return _render_enemy_explosion(
                    renderer,
                    r_in,
                    explosion_frame,
                    state.level.falling_rock_explosion_pos[:, idx],
                )

            def render_enemy(r_in):
                return _render_ufo_like_enemy(
                    renderer,
                    r_in,
                    state.level.falling_rock_pos[0][idx],
                    state.level.falling_rock_pos[1][idx],
                    state.level.falling_rock_active[idx],
                    align=True,
                )

            new_raster = jax.lax.cond(
                explosion_frame > 0,
                render_explosion,
                render_enemy,
                raster,
            )
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(self._env.consts.FALLING_ROCK_MAX))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_lane_blockers(self, raster, state):
        renderer = self._env.renderer

        def body_fun(raster, idx):
            explosion_frame = state.level.lane_blocker_explosion_frame[idx]

            def render_explosion(r_in):
                return _render_enemy_explosion(
                    renderer,
                    r_in,
                    explosion_frame,
                    state.level.lane_blocker_explosion_pos[:, idx],
                )

            def render_enemy(r_in):
                y_pos = state.level.lane_blocker_pos[1][idx]
                mask = _get_white_ufo_mask(renderer, y_pos)

                clip_rows = jnp.maximum(
                    0,
                    jnp.round(y_pos - self._env.consts.LANE_BLOCKER_BOTTOM_Y).astype(jnp.int32),
                )
                is_sinking = state.level.lane_blocker_phase[idx] == int(LaneBlockerState.SINK)

                def clip_mask(base_mask):
                    height = base_mask.shape[0]
                    visible_rows = jnp.maximum(0, height - clip_rows)
                    row_idx = jnp.arange(height)
                    row_mask = row_idx < visible_rows
                    transparent = jnp.array(renderer.jr.TRANSPARENT_ID, dtype=base_mask.dtype)
                    return jnp.where(row_mask[:, None], base_mask, transparent)

                mask = jax.lax.cond(is_sinking, clip_mask, lambda m: m, mask)
                return _render_ufo_like_enemy(
                    renderer,
                    r_in,
                    state.level.lane_blocker_pos[0][idx],
                    y_pos,
                    state.level.lane_blocker_active[idx],
                    align=True,
                    mask=mask,
                )

            new_raster = jax.lax.cond(
                explosion_frame > 0,
                render_explosion,
                render_enemy,
                raster,
            )
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(self._env.consts.LANE_BLOCKER_MAX))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_rejuvenator(self, raster, state):
        renderer = self._env.renderer
        explosion_frame = state.level.rejuvenator_explosion_frame

        def render_explosion(r_in):
            return _render_enemy_explosion(
                renderer,
                r_in,
                explosion_frame,
                state.level.rejuvenator_explosion_pos,
            )

        def render_enemy(r_in):
            return _render_ufo_like_enemy(
                renderer,
                r_in,
                state.level.rejuvenator_pos[0],
                state.level.rejuvenator_pos[1],
                state.level.rejuvenator_active,
                align=False,
            )

        return jax.lax.cond(
            explosion_frame > 0,
            render_explosion,
            render_enemy,
            raster,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_kamikaze(self, raster, state):
        renderer = self._env.renderer
        explosion_frame = state.level.kamikaze_explosion_frame[0]

        def render_explosion(r_in):
            return _render_enemy_explosion(
                renderer,
                r_in,
                explosion_frame,
                state.level.kamikaze_explosion_pos[:, 0],
            )

        def render_enemy(r_in):
            return _render_ufo_like_enemy(
                renderer,
                r_in,
                state.level.kamikaze_pos[0][0],
                state.level.kamikaze_pos[1][0],
                state.level.kamikaze_active[0],
                align=True,
            )

        return jax.lax.cond(
            explosion_frame > 0,
            render_explosion,
            render_enemy,
            raster,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_mothership(self, raster, state):
        renderer = self._env.renderer
        stage = state.level.mothership_stage.astype(jnp.int32)
        timer = state.level.mothership_timer
        pos_x = state.level.mothership_position
        pos_y = jnp.array(
            self._env.consts.MOTHERSHIP_EMERGE_Y - self._env.consts.MOTHERSHIP_HEIGHT,
            dtype=jnp.float32,
        )

        def render_none(r):
            return r

        def render_ship(r):
            mask = _get_white_ufo_mask(renderer, pos_y)
            active = (stage > 0) & (stage < 4)
            return _render_ufo_like_enemy(
                renderer,
                r,
                pos_x,
                pos_y,
                active,
                align=False,
                mask=mask,
            )

        def render_exploding(r):
            explosion_masks = renderer.SHAPE_MASKS["mothership_explosion"]
            step_duration = self._env.consts.MOTHERSHIP_EXPLOSION_STEP_DURATION
            step_idx = jnp.clip(timer // step_duration, 0, 8)
            sprite_idx = renderer._mothership_explosion_seq[step_idx]
            exp_mask = explosion_masks[sprite_idx]
            return renderer.jr.render_at_clipped(r, pos_x, pos_y, exp_mask)

        return jax.lax.switch(
            stage,
            [render_none, render_ship, render_ship, render_ship, render_none, render_exploding],
            raster,
        )


class ToasterMod(JaxAtariInternalModPlugin):
    """Disable Beamrider's heavier visual effects while keeping gameplay unchanged."""

    conflicts_with = ["same_enemies"]

    @partial(jax.jit, static_argnums=(0,))
    def _render_colored_background(self, raster, state):
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_blue_lines(self, raster, state):
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_lives(self, raster, state):
        renderer = self._env.renderer
        hp_mask = renderer.SHAPE_MASKS["live"]
        max_visible_lives = max(self._env.consts.MAX_LIVES - 1, 0)

        def body_fun(r_in, idx):
            is_visible = (state.lives - 1) > idx
            pos_x = jnp.where(is_visible, 32 + (idx * 9), -100)
            new_raster = renderer.jr.render_at_clipped(r_in, pos_x, 183, hp_mask)
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(max_visible_lives))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_player_and_bullet(self, raster, state):
        renderer = self._env.renderer
        player_masks = renderer.SHAPE_MASKS["player_sprite"]
        dead_player_mask = renderer.SHAPE_MASKS["dead_player"]

        raster = jax.lax.cond(
            state.level.death_timer > 0,
            lambda r: renderer.jr.render_at(r, state.level.player_pos, self._env.consts.PLAYER_POS_Y, dead_player_mask),
            lambda r: renderer.jr.render_at(r, state.level.player_pos, self._env.consts.PLAYER_POS_Y, player_masks[9]),
            raster,
        )

        bullet_mask = renderer.SHAPE_MASKS["bullet_sprite"][
            _get_index_bullet(state.level.player_shot_pos[1], state.level.bullet_type, self._env.consts.LASER_ID)
        ]
        shot_x_screen = _get_player_shot_screen_x(
            state.level.player_shot_pos,
            state.level.player_shot_vel,
            state.level.bullet_type,
            self._env.consts.LASER_ID,
        )
        return renderer.jr.render_at_clipped(
            raster,
            shot_x_screen,
            state.level.player_shot_pos[1],
            bullet_mask,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_enemy_shots(self, raster, state):
        renderer = self._env.renderer
        shot_mask = renderer.SHAPE_MASKS["enemy_shot"][0]

        def body_fun(r_in, idx):
            visible = (state.level.enemy_shot_explosion_frame[idx] == 0) & (
                state.level.enemy_shot_pos[1][idx] <= self._env.consts.BOTTOM_CLIP
            )
            y_pos = jnp.where(visible, state.level.enemy_shot_pos[1][idx], 500.0)
            x_pos = state.level.enemy_shot_pos[0][idx] + _get_ufo_alignment(y_pos)
            new_raster = renderer.jr.render_at_clipped(r_in, x_pos, y_pos, shot_mask)
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(9))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_white_ufos(self, raster, state):
        renderer = self._env.renderer
        hide_all = state.level.blue_line_counter < len(BLUE_LINE_INIT_TABLE)

        def body_fun(r_in, idx):
            y_pos = state.level.white_ufo_pos[1][idx]
            mask = _get_white_ufo_mask(renderer, y_pos)
            visible = (state.level.ufo_explosion_frame[idx] == 0) & jnp.logical_not(hide_all)
            new_raster = _render_mask_when_active(
                renderer,
                r_in,
                state.level.white_ufo_pos[0][idx],
                y_pos,
                visible,
                mask,
                align=True,
            )
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(3))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_bouncer(self, raster, state):
        renderer = self._env.renderer
        visible = state.level.bouncer_active & (state.level.bouncer_explosion_frame == 0)
        return _render_mask_when_active(
            renderer,
            raster,
            state.level.bouncer_pos[0],
            state.level.bouncer_pos[1],
            visible,
            renderer.SHAPE_MASKS["bouncer"],
            align=True,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_chasing_meteoroids(self, raster, state):
        renderer = self._env.renderer
        mask = renderer.SHAPE_MASKS["chasing_meteoroid"]

        def body_fun(r_in, idx):
            visible = state.level.chasing_meteoroid_active[idx] & (
                state.level.chasing_meteoroid_explosion_frame[idx] == 0
            )
            new_raster = _render_mask_when_active(
                renderer,
                r_in,
                state.level.chasing_meteoroid_pos[0][idx],
                state.level.chasing_meteoroid_pos[1][idx],
                visible,
                mask,
                align=True,
            )
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(self._env.consts.CHASING_METEOROID_MAX))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_falling_rocks(self, raster, state):
        renderer = self._env.renderer
        rock_masks = renderer.SHAPE_MASKS["falling_rocks"]

        def body_fun(r_in, idx):
            y_pos = state.level.falling_rock_pos[1][idx]
            sprite_idx = _get_index_falling_rock(y_pos) - 1
            mask = rock_masks[sprite_idx]
            visible = state.level.falling_rock_active[idx] & (state.level.falling_rock_explosion_frame[idx] == 0)
            new_raster = _render_mask_when_active(
                renderer,
                r_in,
                state.level.falling_rock_pos[0][idx],
                y_pos,
                visible,
                mask,
                align=True,
            )
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(self._env.consts.FALLING_ROCK_MAX))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_lane_blockers(self, raster, state):
        renderer = self._env.renderer
        blocker_masks = renderer.SHAPE_MASKS["lane_blocker"]

        def body_fun(r_in, idx):
            y_pos = state.level.lane_blocker_pos[1][idx]
            sprite_idx = _get_index_lane_blocker(y_pos) - 1
            sprite_idx = jnp.clip(sprite_idx, 0, blocker_masks.shape[0] - 1)
            mask = blocker_masks[sprite_idx]

            clip_rows = jnp.maximum(
                0,
                jnp.round(y_pos - self._env.consts.LANE_BLOCKER_BOTTOM_Y).astype(jnp.int32),
            )
            is_sinking = state.level.lane_blocker_phase[idx] == int(LaneBlockerState.SINK)

            def clip_mask(base_mask):
                height = base_mask.shape[0]
                visible_rows = jnp.maximum(0, height - clip_rows)
                row_idx = jnp.arange(height)
                row_mask = row_idx < visible_rows
                transparent = jnp.array(renderer.jr.TRANSPARENT_ID, dtype=base_mask.dtype)
                return jnp.where(row_mask[:, None], base_mask, transparent)

            mask = jax.lax.cond(is_sinking, clip_mask, lambda m: m, mask)
            visible = state.level.lane_blocker_active[idx] & (state.level.lane_blocker_explosion_frame[idx] == 0)
            new_raster = _render_mask_when_active(
                renderer,
                r_in,
                state.level.lane_blocker_pos[0][idx],
                y_pos,
                visible,
                mask,
                align=True,
            )
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(self._env.consts.LANE_BLOCKER_MAX))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_rejuvenator(self, raster, state):
        renderer = self._env.renderer
        rejuv_masks = renderer.SHAPE_MASKS["rejuvenator"]
        stage = _get_index_rejuvenator(state.level.rejuvenator_pos[1])
        sprite_idx = jnp.where(state.level.rejuvenator_dead, 4, jnp.clip(stage - 1, 0, 3))
        mask = rejuv_masks[sprite_idx]
        visible = state.level.rejuvenator_active & (state.level.rejuvenator_explosion_frame == 0)
        return _render_mask_when_active(
            renderer,
            raster,
            state.level.rejuvenator_pos[0],
            state.level.rejuvenator_pos[1],
            visible,
            mask,
            align=False,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_kamikaze(self, raster, state):
        renderer = self._env.renderer
        y_pos = state.level.kamikaze_pos[1][0]
        sprite_idx = _get_index_kamikaze(y_pos) - 1
        sprite_idx = jnp.clip(sprite_idx, 0, 3)
        mask = renderer.SHAPE_MASKS["kamikaze"][sprite_idx]
        visible = state.level.kamikaze_active[0] & (state.level.kamikaze_explosion_frame[0] == 0)
        return _render_mask_when_active(
            renderer,
            raster,
            state.level.kamikaze_pos[0][0],
            y_pos,
            visible,
            mask,
            align=True,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_coins(self, raster, state):
        renderer = self._env.renderer
        coin_masks = renderer.SHAPE_MASKS["coin"]
        static_mask = coin_masks[renderer._coin_anim_seq[0]]

        def body_fun(r_in, idx):
            visible = state.level.coin_active[idx] & (state.level.coin_explosion_frame[idx] == 0)
            new_raster = _render_mask_when_active(
                renderer,
                r_in,
                state.level.coin_pos[0][idx],
                state.level.coin_pos[1][idx],
                visible,
                static_mask,
                align=True,
            )
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(self._env.consts.COIN_MAX))
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_mothership(self, raster, state):
        renderer = self._env.renderer
        visible = state.level.mothership_stage.astype(jnp.int32) == 2
        y_pos = jnp.array(
            self._env.consts.MOTHERSHIP_EMERGE_Y - self._env.consts.MOTHERSHIP_HEIGHT,
            dtype=jnp.float32,
        )
        return _render_mask_when_active(
            renderer,
            raster,
            state.level.mothership_position,
            y_pos,
            visible,
            renderer.SHAPE_MASKS["mothership"],
            align=False,
        )


class MothershipLaserMod(JaxAtariInternalModPlugin):
    """Turn the mothership sequence into a stationary lane laser attack."""

    conflicts_with = ["toaster"]

    @partial(jax.jit, static_argnums=(0,))
    def _mothership_step(self, state, white_ufo_left, enemy_explosion_frame, is_hit):
        stage = state.level.mothership_stage.astype(jnp.int32)
        timer = state.level.mothership_timer.astype(jnp.int32)
        pos_x = state.level.mothership_position
        sector = state.sector

        is_ltr = (sector % 2) != 0

        def idle_logic():
            explosions_finished = jnp.all(enemy_explosion_frame == 0)
            start = (white_ufo_left == 0) & explosions_finished
            return jnp.where(start, 1, 0), jnp.where(start, 1, 0), pos_x.astype(jnp.float32)

        def emergence_logic():
            next_timer = timer + 1
            finished = next_timer > 15
            s = jnp.clip((timer - 1) // 2, 0, 6)
            rel_x = jnp.take(self._env.mothership_anim_x, s)
            travel_x = jnp.where(
                is_ltr,
                rel_x.astype(jnp.float32),
                (160 - 16 - rel_x + 8).astype(jnp.float32),
            )
            random_lane = jax.random.randint(state.rng, (), 1, 6, dtype=jnp.int32)
            packed_timer = _pack_mothership_laser_timer(
                random_lane,
                jnp.array(1, dtype=jnp.int32),
                jnp.array(False),
                jnp.array(False),
            )
            next_pos = travel_x
            next_stage = jnp.where(finished, 2, 1)
            next_timer = jnp.where(finished, packed_timer, next_timer)
            return next_stage, next_timer, next_pos

        def moving_or_firing_logic():
            target_lane = _get_mothership_laser_lane_from_timer(timer)
            stop_x = _get_mothership_stop_x(self._env, target_lane)
            exit_x = jnp.where(
                is_ltr,
                (160.0 - 16.0 - self._env.mothership_anim_x[6] + 8.0).astype(jnp.float32),
                self._env.mothership_anim_x[6].astype(jnp.float32),
            )
            phase_timer = _get_mothership_laser_phase_timer(timer)
            firing = _is_mothership_laser_firing(timer)
            post_fire = _has_mothership_laser_post_fire(timer)

            def moving_logic():
                move_target_x = jnp.where(post_fire, exit_x, stop_x)
                should_move = (phase_timer % 6 == 2) | (phase_timer % 6 == 0)
                dx = jnp.where(is_ltr, 1.0, -1.0)
                moved_x = pos_x + jnp.where(should_move, dx, 0.0)
                reached = jnp.where(is_ltr, moved_x >= move_target_x, moved_x <= move_target_x)
                next_pos = jnp.where(reached, move_target_x, moved_x)
                continue_timer = _pack_mothership_laser_timer(
                    target_lane,
                    phase_timer + 1,
                    jnp.array(False),
                    post_fire,
                )
                firing_timer = _pack_mothership_laser_timer(
                    target_lane,
                    jnp.array(0, dtype=jnp.int32),
                    jnp.array(True),
                    jnp.array(False),
                )
                next_stage = jnp.where(post_fire & reached, 3, 2)
                next_timer = jnp.where(
                    post_fire & reached,
                    jnp.array(1, dtype=jnp.int32),
                    jnp.where(reached, firing_timer, continue_timer),
                )
                final_stage = jnp.where(is_hit, 5, next_stage)
                final_timer = jnp.where(is_hit, 0, next_timer)
                return final_stage, final_timer, next_pos

            def firing_logic():
                next_phase_timer = phase_timer + 1
                finished = next_phase_timer >= MOTHERSHIP_LASER_DURATION
                continue_timer = _pack_mothership_laser_timer(
                    target_lane,
                    jnp.array(1, dtype=jnp.int32),
                    jnp.array(False),
                    jnp.array(True),
                )
                next_stage = jnp.where(finished, 2, 2)
                next_timer = jnp.where(
                    finished,
                    continue_timer,
                    _pack_mothership_laser_timer(
                        target_lane,
                        next_phase_timer,
                        jnp.array(True),
                        jnp.array(False),
                    ),
                )
                final_stage = jnp.where(is_hit, 5, next_stage)
                final_timer = jnp.where(is_hit, 0, next_timer)
                return final_stage, final_timer, stop_x

            return jax.lax.cond(firing, firing_logic, moving_logic)

        def descending_logic():
            next_timer = timer + 1
            finished = next_timer > 15
            s = jnp.clip(6 - (timer - 1) // 2, 0, 6)
            rel_x = jnp.take(self._env.mothership_anim_x, s)
            calculated_pos = jnp.where(
                is_ltr,
                (160 - 16 - rel_x + 8).astype(jnp.float32),
                rel_x.astype(jnp.float32),
            )
            return jnp.where(finished, 4, 3), jnp.where(finished, 0, next_timer), calculated_pos

        def done_logic():
            return 0, 0, self._env.mothership_offscreen

        def exploding_logic():
            duration = 9 * self._env.consts.MOTHERSHIP_EXPLOSION_STEP_DURATION
            finished = timer >= duration
            return jnp.where(finished, 4, 5), timer + 1, pos_x

        new_stage, new_timer, new_pos = jax.lax.switch(
            stage,
            [idle_logic, emergence_logic, moving_or_firing_logic, descending_logic, done_logic, exploding_logic],
        )
        sector_advance = new_stage == 4
        return new_pos, new_timer, new_stage, sector_advance

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_shot_step(self, state, white_ufo_pos, white_ufo_pattern_id, white_ufo_pattern_timer):
        shot_pos, shot_lane, shot_timer, hit_count = type(self._env)._enemy_shot_step(
            self._env,
            state,
            white_ufo_pos,
            white_ufo_pattern_id,
            white_ufo_pattern_timer,
        )

        player_left = state.level.player_pos.astype(jnp.float32)
        player_top = float(self._env.consts.PLAYER_POS_Y)
        player_size = self._env.player_sprite_size
        laser_hit = _laser_hits_box(
            self._env,
            state,
            player_left,
            player_top,
            player_size[1],
            player_size[0],
        ).astype(jnp.int32)
        return shot_pos, shot_lane, shot_timer, hit_count + laser_hit

    @partial(jax.jit, static_argnums=(0,))
    def _collisions_step(
        self,
        state,
        player_x,
        vel_x,
        player_shot_pos,
        player_shot_vel,
        player_shot_frame,
        torpedos_left,
        bullet_type,
        shooting_cooldown,
        shooting_delay,
        shot_type_pending,
        enemy_updates,
        key,
    ):
        collision_results = type(self._env)._collisions_step(
            self._env,
            state,
            player_x,
            vel_x,
            player_shot_pos,
            player_shot_vel,
            player_shot_frame,
            torpedos_left,
            bullet_type,
            shooting_cooldown,
            shooting_delay,
            shot_type_pending,
            enemy_updates,
            key,
        )

        (
            player_x,
            vel_x,
            player_shot_pos,
            player_shot_vel,
            player_shot_frame,
            torpedos_left,
            bullet_type,
            shooting_cooldown,
            shooting_delay,
            shot_type_pending,
        ) = collision_results["player"]

        shot_x = _get_player_shot_screen_x(
            player_shot_pos,
            player_shot_vel,
            bullet_type,
            self._env.consts.LASER_ID,
        )
        shot_y = player_shot_pos[1]
        bullet_idx = _get_index_bullet(shot_y, bullet_type, self._env.consts.LASER_ID)
        bullet_size = jnp.take(self._env.bullet_sprite_sizes, bullet_idx, axis=0)
        shot_active = shot_y < float(self._env.consts.BOTTOM_CLIP)
        laser_hit = _laser_hits_box(
            self._env,
            state,
            shot_x,
            shot_y,
            bullet_size[1],
            bullet_size[0],
        )
        laser_hit = shot_active & laser_hit

        player_shot_pos = jnp.where(laser_hit, self._env.bullet_offscreen, player_shot_pos)
        player_shot_frame = jnp.where(laser_hit, jnp.array(-1, dtype=player_shot_frame.dtype), player_shot_frame)
        shooting_cooldown = jnp.where(laser_hit, self._env.consts.PLAYER_SHOT_RECOVERY, shooting_cooldown)

        return collision_results | {
            "player": (
                player_x,
                vel_x,
                player_shot_pos,
                player_shot_vel,
                player_shot_frame,
                torpedos_left,
                bullet_type,
                shooting_cooldown,
                shooting_delay,
                shot_type_pending,
            )
        }

    @partial(jax.jit, static_argnums=(0,))
    def _render_enemy_shots(self, raster, state):
        renderer = self._env.renderer
        raster = type(renderer)._render_enemy_shots(renderer, raster, state)
        shot_mask = renderer.SHAPE_MASKS["enemy_shot"][MOTHERSHIP_LASER_SPRITE_IDX]
        seg_x, seg_y, visible = _get_mothership_laser_segments(self._env, state)

        def body_fun(r_in, idx):
            draw_y = jnp.where(visible[idx], seg_y[idx], 500.0)
            draw_x = jnp.where(visible[idx], seg_x[idx], 500.0)
            new_raster = renderer.jr.render_at_clipped(r_in, draw_x, draw_y, shot_mask)
            return new_raster, None

        raster, _ = jax.lax.scan(body_fun, raster, jnp.arange(MOTHERSHIP_LASER_SEGMENT_COUNT))
        return raster


class TeleportUFOsMod(JaxAtariInternalModPlugin):
    """Adds a rare White-UFO pattern that snaps once to a nearby lane."""

    def _advance_white_ufos(self, state):
        # Beamrider caches a vmapped White-UFO step function at env init time.
        # Rebuild it here so this mod uses the patched _white_ufo_step logic.
        results = jax.vmap(
            self._white_ufo_step,
            in_axes=(None, 1, 1, 0, 0, 0, 0, 0, 0, 0),
        )(
            state.sector,
            state.level.white_ufo_pos,
            state.level.white_ufo_vel,
            state.level.white_ufo_time_on_lane,
            state.level.white_ufo_attack_time,
            state.level.white_ufo_already_left,
            state.level.white_ufo_spawn_delay,
            state.level.white_ufo_pattern_id,
            state.level.white_ufo_pattern_timer,
            state.level.white_ufo_rngs,
        )

        positions, vel_x, vel_y, time_on_lane, attack_time, already_left, spawn_delay, pattern_id, pattern_timer, new_keys = results
        return WhiteUFOUpdate(
            pos=positions.T,
            vel=jnp.stack([vel_x, vel_y]),
            time_on_lane=time_on_lane,
            attack_time=attack_time,
            already_left=already_left,
            spawn_delay=spawn_delay,
            pattern_id=pattern_id.astype(jnp.int32),
            pattern_timer=pattern_timer.astype(jnp.int32),
            rngs=new_keys,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _white_ufo_pattern_requires_lane_motion(self, pattern_id):
        return jnp.isin(pattern_id, self._env._lane_motion_patterns) | _is_teleport_ufo_pattern(pattern_id)

    @partial(jax.jit, static_argnums=(0,))
    def _white_ufo_choose_pattern(
        self,
        key,
        *,
        allow_shoot,
        prev_pattern,
        is_kamikaze_zone,
        sector,
        stage,
        lane,
        is_on_lane,
    ):
        pattern_choices = jnp.array(
            [
                int(WhiteUFOPattern.DROP_STRAIGHT),
                int(WhiteUFOPattern.DROP_LEFT),
                int(WhiteUFOPattern.DROP_RIGHT),
                int(WhiteUFOPattern.SHOOT),
                int(WhiteUFOPattern.MOVE_BACK),
                int(WhiteUFOPattern.KAMIKAZE),
                int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT),
                int(WhiteUFOPattern.TRIPLE_SHOT_LEFT),
                TELEPORT_UFO_PATTERN_ID,
            ],
            dtype=jnp.int32,
        )
        pattern_probs = jnp.concatenate(
            [
                self._env.ufo_pattern_probs,
                jnp.array([TELEPORT_UFO_PATTERN_WEIGHT], dtype=jnp.float32),
            ]
        )

        is_move_back = prev_pattern == int(WhiteUFOPattern.MOVE_BACK)
        chain_mask = jnp.ones_like(pattern_probs).at[0].set(jnp.where(is_move_back, 0.0, 1.0))
        pattern_probs = pattern_probs * chain_mask

        shoot_mask = jnp.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        pattern_probs = jnp.where(allow_shoot, pattern_probs, pattern_probs * shoot_mask)

        kamikaze_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        pattern_probs = jnp.where(is_kamikaze_zone, pattern_probs, pattern_probs * kamikaze_mask)

        move_back_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        pattern_probs = jnp.where(stage >= 4, pattern_probs, pattern_probs * move_back_mask)

        can_triple = (sector >= 7) & (stage >= 4) & (stage <= 6) & is_on_lane
        can_triple_right = can_triple & (lane >= 1) & (lane <= 3)
        can_triple_left = can_triple & (lane >= 3) & (lane <= 5)

        triple_right_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0], dtype=jnp.float32)
        triple_left_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0], dtype=jnp.float32)
        pattern_probs = jnp.where(can_triple_right, pattern_probs, pattern_probs * triple_right_mask)
        pattern_probs = jnp.where(can_triple_left, pattern_probs, pattern_probs * triple_left_mask)

        min_lane, max_lane = _get_teleport_ufo_lane_bounds_from_stage(stage)
        _, teleport_valid_mask = _get_teleport_ufo_candidate_lanes(lane, min_lane, max_lane)
        can_teleport = is_on_lane & jnp.any(teleport_valid_mask)
        teleport_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
        pattern_probs = jnp.where(can_teleport, pattern_probs, pattern_probs * teleport_mask)

        prob_sum = jnp.sum(pattern_probs)
        pattern_probs = jnp.where(prob_sum > 0, pattern_probs / prob_sum, pattern_probs)

        duration_table = jnp.concatenate(
            [
                self._env.ufo_pattern_durations,
                jnp.array([TELEPORT_UFO_PATTERN_DURATION], dtype=jnp.int32),
            ]
        )
        pattern = jax.random.choice(key, pattern_choices, shape=(), p=pattern_probs)
        duration = duration_table[pattern]
        return pattern, duration

    @partial(jax.jit, static_argnums=(0,))
    def _white_ufo_update_pattern_state(
        self,
        sector,
        position,
        time_on_lane,
        attack_time,
        already_left,
        spawn_delay,
        pattern_id,
        pattern_timer,
        key,
    ):
        on_top_lane = position[1] <= self._env.consts.TOP_CLIP
        time_on_lane = jnp.where(on_top_lane, time_on_lane + 1, 0)
        attack_time = jnp.where(on_top_lane, 0, attack_time)

        ufo_x = position[0].astype(jnp.float32)
        ufo_y = position[1].astype(jnp.float32)
        lane_x_at_ufo_y = self._env.top_lanes_x + self._env.lane_dx_over_dy * (ufo_y - float(self._env.consts.TOP_CLIP))
        closest_lane_id = jnp.argmin(jnp.abs(lane_x_at_ufo_y - ufo_x)).astype(jnp.int32)
        closest_lane_x = lane_x_at_ufo_y[closest_lane_id]
        dist_to_lane = jnp.abs(closest_lane_x - ufo_x)
        is_on_lane = dist_to_lane <= 0.25

        is_triple = (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)) | (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT))
        is_teleport = _is_teleport_ufo_pattern(pattern_id)

        shots_left = pattern_timer & 7
        last_lane = (pattern_timer >> 3) & 15
        shoot_now = (pattern_timer >> 7) & 1

        def update_triple():
            can_shoot = (shots_left > 0) & is_on_lane & (closest_lane_id != last_lane)
            new_shoot_now = jnp.where(shoot_now == 1, 0, jnp.where(can_shoot, 1, 0))
            new_shots_left = jnp.where(can_shoot, shots_left - 1, shots_left)
            new_last_lane = jnp.where(can_shoot, closest_lane_id, last_lane)
            return (new_shoot_now << 7) | (new_last_lane << 3) | new_shots_left

        plain_timer = jnp.maximum(pattern_timer - 1, jnp.zeros_like(pattern_timer))
        teleport_timer = _pack_teleport_ufo_timer(
            jnp.maximum(_get_teleport_ufo_remaining(pattern_timer) - 1, 0),
            _has_teleport_ufo_been_used(pattern_timer),
        )
        pattern_timer = jnp.where(is_triple, update_triple(), jnp.where(is_teleport, teleport_timer, plain_timer))

        shootable_lane = (closest_lane_id > 0) & (closest_lane_id < 6)
        allow_shoot = (~on_top_lane) & shootable_lane

        is_drop_pattern = (
            (pattern_id == int(WhiteUFOPattern.DROP_STRAIGHT))
            | (pattern_id == int(WhiteUFOPattern.DROP_LEFT))
            | (pattern_id == int(WhiteUFOPattern.DROP_RIGHT))
            | (pattern_id == int(WhiteUFOPattern.MOVE_BACK))
            | is_teleport
        )
        is_shoot_pattern = pattern_id == int(WhiteUFOPattern.SHOOT)
        is_engagement_pattern = is_drop_pattern | is_shoot_pattern | is_triple
        attack_time = jnp.where((~on_top_lane) & is_engagement_pattern, attack_time + 1, attack_time)

        is_retreat = pattern_id == int(WhiteUFOPattern.RETREAT)
        is_move_back = pattern_id == int(WhiteUFOPattern.MOVE_BACK)
        movement_finished = (is_retreat | is_move_back) & on_top_lane
        pattern_id = jnp.where(movement_finished, int(WhiteUFOPattern.IDLE), pattern_id)
        pattern_timer = jnp.where(movement_finished, 0, pattern_timer)
        attack_time = jnp.where(movement_finished, 0, attack_time)

        triple_finished = is_triple & ((pattern_timer & 7) == 0) & jnp.logical_not((pattern_timer >> 7) & 1) & is_on_lane

        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT), 1, 0)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT), -1, lane_offset)
        in_restricted_stage = position[1] >= 86.0
        min_lane = jnp.where(in_restricted_stage, 1, 0)
        max_lane = jnp.where(in_restricted_stage, 5, 6)
        target_lane_id = jnp.clip(closest_lane_id + lane_offset, min_lane, max_lane)

        triple_stuck = is_triple & is_on_lane & (shots_left > 0) & (target_lane_id == closest_lane_id) & (closest_lane_id == last_lane)
        triple_finished = triple_finished | triple_stuck

        teleport_finished = is_teleport & (_get_teleport_ufo_remaining(pattern_timer) == 0)
        pattern_finished = jnp.where(is_triple, triple_finished, jnp.where(is_teleport, teleport_finished, pattern_timer == 0))
        pattern_finished_off_top = (~on_top_lane) & is_engagement_pattern & pattern_finished & is_on_lane

        key_start_roll, key_start_choice, key_retreat_roll, key_chain_choice, _ = jax.random.split(key, 5)
        retreat_roll = jax.random.uniform(key_retreat_roll)
        retreat_prob = self._env._white_ufo_retreat_prob(attack_time)
        retreat_now = pattern_finished_off_top & (retreat_roll < retreat_prob)
        pattern_id = jnp.where(retreat_now, int(WhiteUFOPattern.RETREAT), pattern_id)
        pattern_timer = jnp.where(retreat_now, self._env.consts.WHITE_UFO_RETREAT_DURATION, pattern_timer)
        attack_time = jnp.where(retreat_now, 0, attack_time)

        chain_next = pattern_finished_off_top & (~retreat_now)
        ufo_stage = _get_index_ufo(position[1])

        def choose_chain_pattern(_):
            is_kamikaze_zone = position[1] >= self._env.consts.KAMIKAZE_Y_THRESHOLD
            pattern, duration = self._white_ufo_choose_pattern(
                key_chain_choice,
                allow_shoot=allow_shoot,
                prev_pattern=pattern_id,
                is_kamikaze_zone=is_kamikaze_zone,
                sector=sector,
                stage=ufo_stage,
                lane=closest_lane_id,
                is_on_lane=is_on_lane,
            )
            return pattern, _init_white_ufo_pattern_timer(pattern, duration)

        def keep_after_chain(_):
            return pattern_id, pattern_timer

        pattern_id, pattern_timer = jax.lax.cond(chain_next, choose_chain_pattern, keep_after_chain, operand=None)

        should_choose_new = on_top_lane & (pattern_id == int(WhiteUFOPattern.IDLE)) & (pattern_timer == 0) & (spawn_delay == 0)
        p_start = type(self._env).entropy_heat_prob_static(
            jnp.where(already_left, time_on_lane * 10, time_on_lane),
            alpha=self._env.consts.WHITE_UFO_ATTACK_ALPHA,
            p_min=jnp.where(already_left, 0.1, self._env.consts.WHITE_UFO_ATTACK_P_MIN),
            p_max=self._env.consts.WHITE_UFO_ATTACK_P_MAX,
        )
        start_roll = jax.random.uniform(key_start_roll)
        start_attack = should_choose_new & (start_roll < p_start)

        def choose_new_pattern(_):
            pattern, duration = self._white_ufo_choose_pattern(
                key_start_choice,
                allow_shoot=jnp.array(False),
                prev_pattern=pattern_id,
                is_kamikaze_zone=jnp.array(False),
                sector=sector,
                stage=ufo_stage,
                lane=closest_lane_id,
                is_on_lane=is_on_lane,
            )
            return pattern, _init_white_ufo_pattern_timer(pattern, duration)

        def keep_pattern(_):
            return pattern_id, pattern_timer

        pattern_id, pattern_timer = jax.lax.cond(start_attack, choose_new_pattern, keep_pattern, operand=None)
        return pattern_id, pattern_timer, time_on_lane, attack_time

    @partial(jax.jit, static_argnums=(0,))
    def _white_ufo_step(
        self,
        sector,
        white_ufo_position,
        white_ufo_vel,
        time_on_lane,
        attack_time,
        already_left,
        spawn_delay,
        pattern_id,
        pattern_timer,
        key,
    ):
        white_ufo_vel_x = white_ufo_vel[0]
        white_ufo_vel_y = white_ufo_vel[1]

        offscreen_pos = self._env.enemy_offscreen
        is_offscreen = jnp.all(white_ufo_position == offscreen_pos)

        new_key, key_use = jax.random.split(key)
        key_pattern, key_motion, key_spawn = jax.random.split(key_use, 3)

        spawn_delay = jnp.maximum(spawn_delay - 1, 0)

        pattern_id, pattern_timer, time_on_lane, attack_time = self._white_ufo_update_pattern_state(
            sector,
            white_ufo_position,
            time_on_lane,
            attack_time,
            already_left,
            spawn_delay,
            pattern_id,
            pattern_timer,
            key_pattern,
        )

        ufo_x = white_ufo_position[0].astype(jnp.float32)
        ufo_y = white_ufo_position[1].astype(jnp.float32)
        lane_x_at_ufo_y = self._env.top_lanes_x + self._env.lane_dx_over_dy * (ufo_y - float(self._env.consts.TOP_CLIP))
        closest_lane_id = jnp.argmin(jnp.abs(lane_x_at_ufo_y - ufo_x)).astype(jnp.int32)
        closest_lane_x = lane_x_at_ufo_y[closest_lane_id]
        dist_to_lane = jnp.abs(closest_lane_x - ufo_x)
        is_on_lane = dist_to_lane <= 0.25
        min_lane, max_lane = _get_teleport_ufo_lane_bounds_from_y(ufo_y)
        candidate_lanes, candidate_valid = _get_teleport_ufo_candidate_lanes(closest_lane_id, min_lane, max_lane)
        has_candidate_lane = jnp.any(candidate_valid)

        def choose_target_lane(_):
            candidate_probs = candidate_valid.astype(jnp.float32)
            candidate_probs = candidate_probs / jnp.sum(candidate_probs)
            return jax.random.choice(key_motion, candidate_lanes, shape=(), p=candidate_probs)

        target_lane = jax.lax.cond(has_candidate_lane, choose_target_lane, lambda _: closest_lane_id, operand=None)
        can_teleport_now = (
            (~is_offscreen)
            & _is_teleport_ufo_pattern(pattern_id)
            & (~_has_teleport_ufo_been_used(pattern_timer))
            & is_on_lane
            & has_candidate_lane
        )
        teleported_position = jnp.array([lane_x_at_ufo_y[target_lane], ufo_y], dtype=jnp.float32)
        white_ufo_position = jnp.where(can_teleport_now, teleported_position, white_ufo_position)
        pattern_timer = jnp.where(
            can_teleport_now,
            _pack_teleport_ufo_timer(_get_teleport_ufo_remaining(pattern_timer), jnp.array(True)),
            pattern_timer,
        )

        requires_lane_motion = self._white_ufo_pattern_requires_lane_motion(pattern_id)
        on_top_lane = white_ufo_position[1] <= self._env.consts.TOP_CLIP
        already_left = already_left | jnp.logical_not(on_top_lane)

        def follow_lane(_):
            return self._env._white_ufo_normal(white_ufo_position, white_ufo_vel_x, white_ufo_vel_y, pattern_id, already_left)

        def stay_on_top(_):
            return self._env._white_ufo_top_lane(white_ufo_position, white_ufo_vel_x, pattern_id, key_motion)

        white_ufo_vel_x, white_ufo_vel_y = jax.lax.cond(requires_lane_motion, follow_lane, stay_on_top, operand=None)

        new_x = white_ufo_position[0] + white_ufo_vel_x
        new_y = white_ufo_position[1] + white_ufo_vel_y

        on_top_lane = new_y <= self._env.consts.TOP_CLIP
        clipped_x = jnp.clip(new_x, self._env.consts.LEFT_CLIP_PLAYER, self._env.consts.RIGHT_CLIP_PLAYER)
        new_x = jnp.where(on_top_lane, clipped_x, new_x)
        new_y = jnp.clip(new_y, self._env.consts.TOP_CLIP, self._env.consts.PLAYER_POS_Y + 1.0)

        should_respawn = (~is_offscreen) & ((new_x < 0) | (new_x > self._env.consts.SCREEN_WIDTH) | (new_y > self._env.consts.PLAYER_POS_Y))

        white_ufo_position = jnp.where(should_respawn, jnp.array([81.0, 43.0]), jnp.array([new_x, new_y]))
        white_ufo_vel_x = jnp.where(should_respawn, 0.0, white_ufo_vel_x)
        white_ufo_vel_y = jnp.where(should_respawn, 0.0, white_ufo_vel_y)
        time_on_lane = jnp.where(should_respawn, 0, time_on_lane)
        attack_time = jnp.where(should_respawn, 0, attack_time)
        spawn_delay = jnp.where(should_respawn, jax.random.randint(key_spawn, (), 1, 301), spawn_delay)
        pattern_id = jnp.where(should_respawn, int(WhiteUFOPattern.IDLE), pattern_id)
        pattern_timer = jnp.where(should_respawn, 0, pattern_timer)

        white_ufo_position = jnp.where(is_offscreen, offscreen_pos, white_ufo_position)
        white_ufo_vel_x = jnp.where(is_offscreen, 0.0, white_ufo_vel_x)
        white_ufo_vel_y = jnp.where(is_offscreen, 0.0, white_ufo_vel_y)
        time_on_lane = jnp.where(is_offscreen, 0, time_on_lane)
        attack_time = jnp.where(is_offscreen, 0, attack_time)
        spawn_delay = jnp.where(is_offscreen, 0, spawn_delay)
        pattern_id = jnp.where(is_offscreen, int(WhiteUFOPattern.IDLE), pattern_id)
        pattern_timer = jnp.where(is_offscreen, 0, pattern_timer)

        return (
            white_ufo_position,
            white_ufo_vel_x,
            white_ufo_vel_y,
            time_on_lane,
            attack_time,
            already_left,
            spawn_delay,
            pattern_id,
            pattern_timer,
            new_key,
        )

from functools import partial

import jax
import jax.numpy as jnp

from jaxatari.modification import JaxAtariInternalModPlugin
from jaxatari.games.jax_beamrider import (
    BLUE_LINE_INIT_TABLE,
    LaneBlockerState,
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


class HardcoreMod(JaxAtariInternalModPlugin):
    """Start with one life and never allow extra lives."""

    constants_overrides = {
        "STARTING_LIVES": 1,
        "MAX_LIVES": 1,
    }


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

from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.beamrider.beamrider_mod_plugins import HardcoreMod, SameEnemiesMod


class BeamriderEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Beamrider.
    """

    REGISTRY = {
        "hardcore": HardcoreMod,
        "same_enemies": SameEnemiesMod,
    }

    def __init__(
        self,
        env,
        mods_config: list = [],
        allow_conflicts: bool = False,
    ):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )

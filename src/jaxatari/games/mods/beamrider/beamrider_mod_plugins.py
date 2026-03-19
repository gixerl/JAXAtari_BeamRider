from jaxatari.modification import JaxAtariInternalModPlugin


class HardcoreMod(JaxAtariInternalModPlugin):
    """Start with one life and never allow extra lives."""

    constants_overrides = {
        "STARTING_LIVES": 1,
        "MAX_LIVES": 1,
    }

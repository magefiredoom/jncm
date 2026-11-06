"""Generator package: level inputs, backbone, heads, corruption, and sampling."""

from jncm.generator.backbone import (
    GeneratorBackbone,
    LevelBlockTransformer,
    LevelInputEmbedding,
    LevelTransformerBlock,
    SinusoidalPositionProjection,
)
from jncm.generator.corruption import (
    MODES,
    add_code_noise,
    corrupt_codes,
    linear_schedule,
    swap_codes,
)
from jncm.generator.heads import (
    CEHead,
    FlowHead,
    FlowMLP,
    RegressHead,
    SinusoidalTimeEmbedding,
    build_head,
    level_weights,
    sample_codes,
    weighted_total,
)

__all__ = [
    "MODES",
    "CEHead",
    "FlowHead",
    "FlowMLP",
    "GeneratorBackbone",
    "LevelBlockTransformer",
    "LevelInputEmbedding",
    "LevelTransformerBlock",
    "RegressHead",
    "SinusoidalPositionProjection",
    "SinusoidalTimeEmbedding",
    "add_code_noise",
    "build_head",
    "corrupt_codes",
    "level_weights",
    "linear_schedule",
    "sample_codes",
    "swap_codes",
    "weighted_total",
]

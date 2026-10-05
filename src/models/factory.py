from src.models.region import CompactfMRITransformer, RegionTransformer
from src.models.time import TimeTransformer
from src.models.hybrid import HybridTransformer


def build_model(model_name: str, **model_params):
    name = str(model_name).lower().replace("-", "_")
    params = dict(model_params)

    if name in ("region", "region_transformer"):
        params["type"] = "region_transformer"
        return CompactfMRITransformer(**params)

    if name in ("time", "time_transformer"):
        params["type"] = "time_transformer"
        return TimeTransformer(**params)

    if name in ("hybrid", "time_region_transformer", "hybrid_transformer", "time_region"):
        params["type"] = "time_region_transformer"
        return HybridTransformer(**params)

    raise ValueError(
        f"Unknown model: {model_name!r}. Expected one of: region, time, hybrid."
    )


__all__ = [
    "build_model",
    "CompactfMRITransformer",
    "RegionTransformer",
    "TimeTransformer",
    "HybridTransformer",
]

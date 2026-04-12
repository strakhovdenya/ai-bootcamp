import statistics
import math

def round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(value, digits)


def mean_or_none(values: list[float]) -> float | None:
    cleaned = [v for v in values if v is not None and not math.isnan(v)]
    return round(statistics.fmean(cleaned), 4) if cleaned else None


def median_or_none(values: list[float]) -> float | None:
    cleaned = [v for v in values if v is not None and not math.isnan(v)]
    return round(statistics.median(cleaned), 4) if cleaned else None


def min_or_none(values: list[float]) -> float | None:
    cleaned = [v for v in values if v is not None and not math.isnan(v)]
    return round(min(cleaned), 4) if cleaned else None


def max_or_none(values: list[float]) -> float | None:
    cleaned = [v for v in values if v is not None and not math.isnan(v)]
    return round(max(cleaned), 4) if cleaned else None
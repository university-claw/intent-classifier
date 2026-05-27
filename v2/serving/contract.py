"""Small, dependency-free contract helpers shared by serving and tests."""

from __future__ import annotations

ABUSE_LABEL = "abuse"
NON_ABUSE_LABEL = "non_abuse"
REQUIRED_LABELS = (NON_ABUSE_LABEL, ABUSE_LABEL)


def validate_probability(value: float, *, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0: {value}")
    return value


def final_from_p_abuse(p_abuse: float, threshold: float) -> str:
    """Return the router-compatible final label from an abuse probability."""

    p = validate_probability(float(p_abuse), name="p_abuse")
    t = validate_probability(float(threshold), name="threshold")
    return ABUSE_LABEL if p >= t else NON_ABUSE_LABEL


def validate_label_map(label2id: dict[str, int]) -> None:
    missing = [label for label in REQUIRED_LABELS if label not in label2id]
    if missing:
        raise ValueError(
            "safety classifier label_map.json must include labels: "
            + ", ".join(REQUIRED_LABELS)
            + f" (missing: {', '.join(missing)})"
        )


def clamp_top_k(top_k: int, num_labels: int) -> int:
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1: {top_k}")
    if num_labels < 1:
        raise ValueError(f"num_labels must be >= 1: {num_labels}")
    return min(top_k, num_labels)

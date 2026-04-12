import json
from typing import Any

def print_console_report(dataset_name: str, structural_summary: dict[str, Any], metric_summary: dict[str, Any]) -> None:
    print("=" * 80)
    print(f"DATASET ANALYSIS: {dataset_name}")
    print("=" * 80)
    print(json.dumps(structural_summary, ensure_ascii=False, indent=2))
    print("-" * 80)
    print(json.dumps(metric_summary, ensure_ascii=False, indent=2))
    print("=" * 80)
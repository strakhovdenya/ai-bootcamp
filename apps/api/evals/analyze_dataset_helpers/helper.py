from pathlib import Path
import re
import argparse
from typing import Any

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze the quality of a LangSmith RAG evaluation dataset."
    )
    parser.add_argument("--dataset-name", required=True, help="Exact LangSmith dataset name")
    parser.add_argument("--output-dir", default="dataset_analysis_outputs", help="Where to write outputs")
    parser.add_argument("--expected-single", type=int, default=None)
    parser.add_argument("--expected-multi", type=int, default=None)
    parser.add_argument("--expected-cannot-answer", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=4, help="Concurrent evaluator calls")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()

def build_output_paths(dataset_name: str, output_dir: str | Path, timestamp: str) -> tuple[Path, Path, Path]:

    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", dataset_name)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{safe_name}_per_example_{timestamp}.csv"
    json_path = output_dir / f"{safe_name}_summary_{timestamp}.json"
    html_path = output_dir / f"{safe_name}_report_{timestamp}.html"

    return csv_path, json_path, html_path

def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def normalize_list_of_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [x for x in (normalize_text(v) for v in value) if x]
    text = normalize_text(value)
    return [text] if text else []


def normalize_list_of_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        seen = set()
        for item in value:
            item = normalize_text(item)
            if item and item not in seen:
                out.append(item)
                seen.add(item)
        return out
    text = normalize_text(value)
    return [text] if text else []
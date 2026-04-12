from pathlib import Path
from typing import Any
import json
import csv

def get_csv_fieldnames():
        fieldnames = [
                "example_id",
                "declared_type",
                "inferred_type",
                "question",
                "ground_truth",
                "reference_id_count",
                "reference_description_count",
                "question_words",
                "ground_truth_words",
                "looks_like_refusal",
                "is_duplicate_question",
                "has_duplicate_reference_ids",
                "has_structural_issue",
                "issue_tags",
                "ground_truth_response_relevancy",
                "ground_truth_faithfulness",
                "reference_context_precision",
                "reference_context_recall",
                "metric_error",
                "reference_context_ids",
                "reference_descriptions",
            ]
        return fieldnames


def save_csv(records: list[dict[str, Any]], output_path: Path) -> None:
        fieldnames = get_csv_fieldnames();
        with output_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for record in records:
                        row = dict(record)
                        row["issue_tags"] = ";".join(record.get("issue_tags", []))
                        row["reference_context_ids"] = json.dumps(record.get("reference_context_ids", []),
                                                                  ensure_ascii=False)
                        row["reference_descriptions"] = json.dumps(record.get("reference_descriptions", []),
                                                                   ensure_ascii=False)
                        writer.writerow({key: row.get(key) for key in fieldnames})



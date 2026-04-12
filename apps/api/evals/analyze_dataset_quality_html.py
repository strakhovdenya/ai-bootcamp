import asyncio
import math
import re
import warnings
from collections import Counter
from datetime import datetime
from typing import Any

from langsmith import Client
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy

from .analyze_dataset_helpers.report_helper_csv import save_csv
from .analyze_dataset_helpers.report_helper_json import save_json
from .analyze_dataset_helpers.report_helper_html import save_html
from .analyze_dataset_helpers.report_helper_console import print_console_report
from .analyze_dataset_helpers.stats_helper import max_or_none,min_or_none,mean_or_none,median_or_none,round_or_none
from .analyze_dataset_helpers.helper import build_output_paths, parse_args, normalize_text, normalize_list_of_text, \
    normalize_list_of_ids

try:
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
except ImportError:  # pragma: no cover
    LLMContextPrecisionWithReference = None
    LLMContextRecall = None

warnings.filterwarnings("ignore", category=DeprecationWarning)

ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
ls_client = Client()

REFUSAL_RE = re.compile(
    r"(not available|not provided|cannot determine|can't determine|don't have|do not have|"
    r"not enough information|information is not available|not specified|unknown)",
    flags=re.IGNORECASE,
)


class DatasetAnalysisError(RuntimeError):
    pass


class DatasetScorers:
    def __init__(self) -> None:
        self.faithfulness = Faithfulness(llm=ragas_llm)
        self.response_relevancy = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
        self.context_precision = (
            LLMContextPrecisionWithReference(llm=ragas_llm)
            if LLMContextPrecisionWithReference is not None
            else None
        )
        self.context_recall = LLMContextRecall(llm=ragas_llm) if LLMContextRecall is not None else None


SCORERS = DatasetScorers()


def infer_type(declared_type: str, reference_ids: list[str]) -> str:
    declared = normalize_text(declared_type).lower()
    if declared in {"single", "multi", "cannot_answer"}:
        return declared
    if len(reference_ids) == 0:
        return "cannot_answer"
    if len(reference_ids) == 1:
        return "single"
    return "multi"


async def score_one_example(record: dict[str, Any], semaphore: asyncio.Semaphore) -> dict[str, Any]:
    async with semaphore:
        result = {
            "ground_truth_response_relevancy": None,
            "ground_truth_faithfulness": None,
            "reference_context_precision": None,
            "reference_context_recall": None,
            "metric_error": None,
        }

        question = record["question"]
        ground_truth = record["ground_truth"]
        reference_contexts = record["reference_descriptions"]

        try:
            if question and ground_truth:
                sample = SingleTurnSample(
                    user_input=question,
                    response=ground_truth,
                    retrieved_contexts=reference_contexts,
                )
                result["ground_truth_response_relevancy"] = await SCORERS.response_relevancy.single_turn_ascore(sample)

            if question and ground_truth and reference_contexts:
                sample = SingleTurnSample(
                    user_input=question,
                    response=ground_truth,
                    retrieved_contexts=reference_contexts,
                )
                result["ground_truth_faithfulness"] = await SCORERS.faithfulness.single_turn_ascore(sample)

            if question and ground_truth and reference_contexts and SCORERS.context_precision is not None:
                sample = SingleTurnSample(
                    user_input=question,
                    reference=ground_truth,
                    retrieved_contexts=reference_contexts,
                )
                result["reference_context_precision"] = await SCORERS.context_precision.single_turn_ascore(sample)

            if question and ground_truth and reference_contexts and SCORERS.context_recall is not None:
                sample = SingleTurnSample(
                    user_input=question,
                    reference=ground_truth,
                    retrieved_contexts=reference_contexts,
                )
                result["reference_context_recall"] = await SCORERS.context_recall.single_turn_ascore(sample)

        except Exception as exc:  # pragma: no cover
            result["metric_error"] = f"{type(exc).__name__}: {exc}"

        return result


async def score_examples(records: list[dict[str, Any]], max_concurrency: int) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [score_one_example(record, semaphore) for record in records]
    return await asyncio.gather(*tasks)


def analyze_structure(records: list[dict[str, Any]], expected: dict[str, int] | None) -> tuple[
    dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    declared_counts = Counter(record["declared_type"] for record in records)
    inferred_counts = Counter(record["inferred_type"] for record in records)

    for record in records:
        issue_tags = []

        if not record["question"]:
            issue_tags.append("missing_question")
        if not record["ground_truth"]:
            issue_tags.append("missing_ground_truth")
        if record["is_duplicate_question"]:
            issue_tags.append("duplicate_question")
        if record["declared_type"] and record["declared_type"] != record["inferred_type"]:
            issue_tags.append("declared_type_mismatch")
        if record["is_answerable"] and not record["reference_context_ids"]:
            issue_tags.append("answerable_without_reference_ids")
        if record["is_answerable"] and not record["reference_descriptions"]:
            issue_tags.append("answerable_without_reference_descriptions")
        if record["inferred_type"] == "cannot_answer" and record["reference_context_ids"]:
            issue_tags.append("cannot_answer_has_reference_ids")
        if record["inferred_type"] == "cannot_answer" and record["reference_descriptions"]:
            issue_tags.append("cannot_answer_has_reference_descriptions")
        if len(record["reference_context_ids"]) != len(record["reference_descriptions"]) and record[
            "reference_descriptions"]:
            issue_tags.append("ids_descriptions_count_mismatch")
        if record["has_duplicate_reference_ids"]:
            issue_tags.append("duplicate_reference_ids")
        if record["inferred_type"] == "cannot_answer" and not record["looks_like_refusal"]:
            issue_tags.append("cannot_answer_without_refusal_style")

        record["issue_tags"] = issue_tags
        record["has_structural_issue"] = bool(issue_tags)

        if issue_tags:
            issues.append(
                {
                    "example_id": record["example_id"],
                    "question": record["question"],
                    "question_type": record["inferred_type"],
                    "issues": issue_tags,
                }
            )

    summary = {
        "total_examples": len(records),
        "declared_type_counts": dict(declared_counts),
        "inferred_type_counts": dict(inferred_counts),
        "duplicate_question_count": sum(1 for record in records if record["is_duplicate_question"]),
        "structural_issue_count": sum(1 for record in records if record["has_structural_issue"]),
        "examples_with_missing_question": sum(1 for record in records if not record["question"]),
        "examples_with_missing_ground_truth": sum(1 for record in records if not record["ground_truth"]),
        "answerable_without_reference_ids": sum(
            1 for record in records if record["is_answerable"] and not record["reference_context_ids"]
        ),
        "answerable_without_reference_descriptions": sum(
            1 for record in records if record["is_answerable"] and not record["reference_descriptions"]
        ),
        "cannot_answer_without_refusal_style": sum(
            1 for record in records if record["inferred_type"] == "cannot_answer" and not record["looks_like_refusal"]
        ),
        "ids_descriptions_count_mismatch": sum(
            1
            for record in records
            if len(record["reference_context_ids"]) != len(record["reference_descriptions"]) and record[
                "reference_descriptions"]
        ),
        "average_question_words": mean_or_none([record["question_words"] for record in records]),
        "average_ground_truth_words": mean_or_none([record["ground_truth_words"] for record in records]),
        "average_reference_id_count": mean_or_none([record["reference_id_count"] for record in records]),
        "average_reference_description_count": mean_or_none(
            [record["reference_description_count"] for record in records]),
    }

    if expected:
        summary["expected_counts"] = expected
        summary["count_deltas"] = {
            key: inferred_counts.get(key, 0) - expected.get(key, 0)
            for key in expected
        }
        summary["expected_total"] = sum(expected.values())
        summary["total_delta"] = len(records) - sum(expected.values())

    return summary, issues


def aggregate_metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "ground_truth_response_relevancy",
        "ground_truth_faithfulness",
        "reference_context_precision",
        "reference_context_recall",
    ]

    def summarize_subset(subset: list[dict[str, Any]]) -> dict[str, Any]:
        summary = {}
        for metric in metric_names:
            values = [record.get(metric) for record in subset]
            valid = [v for v in values if v is not None and not math.isnan(v)]
            summary[metric] = {
                "count": len(valid),
                "mean": mean_or_none(values),
                "median": median_or_none(values),
                "min": min_or_none(values),
                "max": max_or_none(values),
            }
        return summary

    overall = summarize_subset(records)
    by_type = {}
    for qtype in ["single", "multi", "cannot_answer"]:
        subset = [record for record in records if record["inferred_type"] == qtype]
        by_type[qtype] = summarize_subset(subset)

    return {
        "overall": overall,
        "by_type": by_type,
        "metric_error_count": sum(1 for record in records if record.get("metric_error")),
    }


def load_examples(dataset_name: str) -> list[Any]:
    examples = list(ls_client.list_examples(dataset_name=dataset_name))
    if not examples:
        raise DatasetAnalysisError(
            f"Dataset '{dataset_name}' is empty or not found. Check the dataset name and LangSmith credentials."
        )
    return examples


def build_records(examples: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    normalized_questions = []

    for example in examples:
        inputs = dict(example.inputs or {})
        outputs = dict(example.outputs or {})

        question = normalize_text(inputs.get("question"))
        ground_truth = normalize_text(outputs.get("ground_truth"))
        declared_type = normalize_text(outputs.get("question_type")).lower()
        reference_context_ids = normalize_list_of_ids(outputs.get("reference_context_ids"))
        raw_reference_context_ids = outputs.get("reference_context_ids")
        reference_descriptions = normalize_list_of_text(outputs.get("reference_description"))
        inferred_type = infer_type(declared_type, reference_context_ids)

        normalized_questions.append(question.lower())

        if isinstance(raw_reference_context_ids, list):
            raw_id_count = len([normalize_text(x) for x in raw_reference_context_ids if normalize_text(x)])
            deduped_id_count = len(reference_context_ids)
            has_duplicate_reference_ids = raw_id_count != deduped_id_count
        else:
            has_duplicate_reference_ids = False

        record = {
            "example_id": str(example.id),
            "question": question,
            "ground_truth": ground_truth,
            "declared_type": declared_type,
            "inferred_type": inferred_type,
            "reference_context_ids": reference_context_ids,
            "reference_descriptions": reference_descriptions,
            "reference_id_count": len(reference_context_ids),
            "reference_description_count": len(reference_descriptions),
            "question_words": len(question.split()) if question else 0,
            "ground_truth_words": len(ground_truth.split()) if ground_truth else 0,
            "looks_like_refusal": bool(REFUSAL_RE.search(ground_truth)),
            "has_duplicate_reference_ids": has_duplicate_reference_ids,
            "is_answerable": inferred_type in {"single", "multi"},
        }
        records.append(record)

    question_counter = Counter(normalized_questions)
    for record in records:
        record["is_duplicate_question"] = bool(record["question"] and question_counter[record["question"].lower()] > 1)

    return records


async def main_async() -> None:
    args = parse_args()
    examples = load_examples(args.dataset_name)
    records = build_records(examples)

    expected = None
    if all(v is not None for v in [args.expected_single, args.expected_multi, args.expected_cannot_answer]):
        expected = {
            "single": args.expected_single,
            "multi": args.expected_multi,
            "cannot_answer": args.expected_cannot_answer,
        }

    metric_results = await score_examples(records, max_concurrency=args.max_concurrency)
    for record, metrics in zip(records, metric_results):
        for key, value in metrics.items():
            if isinstance(value, float):
                record[key] = round_or_none(value)
            else:
                record[key] = value

    structural_summary, issues = analyze_structure(records, expected=expected)
    metric_summary = aggregate_metric_summary(records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path, json_path, html_path = build_output_paths(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        timestamp=timestamp
    )

    save_csv(records, csv_path)
    save_json(
        {
            "dataset_name": args.dataset_name,
            "generated_at": timestamp,
            "structural_summary": structural_summary,
            "metric_summary": metric_summary,
            "issues": issues,
            "csv_path": str(csv_path),
            "html_path": str(html_path),
        },
        json_path,
    )
    save_html(
        dataset_name=args.dataset_name,
        generated_at=timestamp,
        records=records,
        structural_summary=structural_summary,
        metric_summary=metric_summary,
        issues=issues,
        output_path=html_path,
    )

    print_console_report(args.dataset_name, structural_summary, metric_summary)
    print(f"Per-example CSV written to: {csv_path}")
    print(f"Summary JSON written to: {json_path}")
    print(f"HTML report written to: {html_path}")


if __name__ == "__main__":
    asyncio.run(main_async())

from pathlib import Path
from typing import Any
import html
import math
import json


def render_metric_value(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.4f}"


def render_summary_table(summary: dict[str, Any], title: str) -> str:
    rows = []
    for metric_name, stats in summary.items():
        rows.append(
            f"<tr><td>{html.escape(metric_name)}</td>"
            f"<td>{stats.get('count', 0)}</td>"
            f"<td>{render_metric_value(stats.get('mean'))}</td>"
            f"<td>{render_metric_value(stats.get('median'))}</td>"
            f"<td>{render_metric_value(stats.get('min'))}</td>"
            f"<td>{render_metric_value(stats.get('max'))}</td></tr>"
        )
    return (
        f"<section class='panel'><h3>{html.escape(title)}</h3>"
        "<table><thead><tr><th>Metric</th><th>Count</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def render_kv_table(mapping: dict[str, Any], title: str) -> str:
    rows = []
    for key, value in mapping.items():
        if isinstance(value, (dict, list)):
            value_rendered = html.escape(json.dumps(value, ensure_ascii=False))
        else:
            value_rendered = html.escape(str(value))
        rows.append(f"<tr><td>{html.escape(str(key))}</td><td>{value_rendered}</td></tr>")
    return f"<section class='panel'><h3>{html.escape(title)}</h3><table><tbody>{''.join(rows)}</tbody></table></section>"


def issue_badges(tags: list[str]) -> str:
    if not tags:
        return "<span class='chip ok'>ok</span>"
    return "".join(f"<span class='chip warn'>{html.escape(tag)}</span>" for tag in tags)


def metric_cell(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "<td>—</td>"
    cls = "good" if value >= 0.9 else "mid" if value >= 0.75 else "bad"
    return f"<td class='{cls}'>{value:.4f}</td>"

def badge_class(qtype: str) -> str:
    return {
        "single": "badge-single",
        "multi": "badge-multi",
        "cannot_answer": "badge-cannot",
    }.get(qtype, "badge-other")

def save_html(
    dataset_name: str,
    generated_at: str,
    records: list[dict[str, Any]],
    structural_summary: dict[str, Any],
    metric_summary: dict[str, Any],
    issues: list[dict[str, Any]],
    output_path: Path,
) -> None:
    by_type = {qtype: [r for r in records if r["inferred_type"] == qtype] for qtype in ["single", "multi", "cannot_answer"]}
    issue_rows = "".join(
        f"<tr><td>{html.escape(item['question_type'])}</td><td>{html.escape(item['question'])}</td><td>{''.join(f'<span class=\"chip warn\">{html.escape(tag)}</span>' for tag in item['issues'])}</td></tr>"
        for item in issues
    )

    example_rows = []
    for idx, record in enumerate(records, start=1):
        refs = "<br>".join(html.escape(x) for x in record.get("reference_context_ids", [])) or "—"
        ref_desc = "<details><summary>show</summary>" + "<hr>".join(html.escape(x) for x in record.get("reference_descriptions", [])) + "</details>" if record.get("reference_descriptions") else "—"
        example_rows.append(
            "<tr data-type='{type}' data-issue='{issue}'>"
            "<td>{idx}</td>"
            "<td><span class='badge {badge}'>{type}</span></td>"
            "<td>{question}</td>"
            "<td>{ground_truth}</td>"
            "<td>{refs}</td>"
            "<td>{issues}</td>"
            "{rel}"
            "{faith}"
            "{prec}"
            "{recall}"
            "<td>{error}</td>"
            "<td>{ref_desc}</td>"
            "</tr>".format(
                idx=idx,
                badge=badge_class(record["inferred_type"]),
                type=html.escape(record["inferred_type"]),
                issue="1" if record.get("has_structural_issue") else "0",
                question=html.escape(record.get("question", "")),
                ground_truth=html.escape(record.get("ground_truth", "")),
                refs=refs,
                issues=issue_badges(record.get("issue_tags", [])),
                rel=metric_cell(record.get("ground_truth_response_relevancy")),
                faith=metric_cell(record.get("ground_truth_faithfulness")),
                prec=metric_cell(record.get("reference_context_precision")),
                recall=metric_cell(record.get("reference_context_recall")),
                error=html.escape(record.get("metric_error") or "—"),
                ref_desc=ref_desc,
            )
        )

    type_cards = "".join(
        f"<div class='card'><div class='label'>{qtype}</div><div class='value'>{len(items)}</div></div>"
        for qtype, items in by_type.items()
    )

    html_doc = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dataset analysis — {html.escape(dataset_name)}</title>
  <style>
    :root {{
      --bg:#0b1020; --panel:#141b34; --panel2:#1b2344; --text:#e8ecf8; --muted:#9fb0d6;
      --good:#6ee7b7; --mid:#fbbf24; --bad:#fca5a5; --line:#2d3763; --accent:#8ab4ff;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.5 Inter, system-ui, sans-serif; }}
    .wrap {{ max-width:1600px; margin:0 auto; padding:24px; }}
    h1,h2,h3 {{ margin:0 0 12px; }}
    h1 {{ font-size:28px; }}
    h2 {{ font-size:20px; margin-top:28px; }}
    h3 {{ font-size:16px; color:var(--accent); }}
    .sub {{ color:var(--muted); margin-top:8px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; margin:20px 0; }}
    .card, .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:16px; }}
    .card .label {{ color:var(--muted); text-transform:uppercase; font-size:12px; letter-spacing:.06em; }}
    .card .value {{ font-size:32px; font-weight:700; margin-top:8px; }}
    table {{ width:100%; border-collapse:collapse; }}
    th, td {{ text-align:left; vertical-align:top; padding:10px 12px; border-top:1px solid var(--line); }}
    th {{ color:var(--muted); font-weight:600; background:rgba(255,255,255,.02); position:sticky; top:0; }}
    .tables {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
    @media (max-width:1100px) {{ .tables {{ grid-template-columns:1fr; }} }}
    .badge {{ display:inline-block; padding:4px 8px; border-radius:999px; font-size:12px; font-weight:600; }}
    .badge-single {{ background:#14385f; color:#9fd0ff; }}
    .badge-multi {{ background:#3c2558; color:#d7b6ff; }}
    .badge-cannot {{ background:#4e2b2b; color:#ffb4b4; }}
    .badge-other {{ background:#334155; color:#cbd5e1; }}
    .chip {{ display:inline-block; margin:2px 6px 2px 0; padding:3px 8px; border-radius:999px; font-size:12px; }}
    .chip.ok {{ background:#143d31; color:#a7f3d0; }}
    .chip.warn {{ background:#4a2f17; color:#fcd34d; }}
    .good {{ color:var(--good); font-weight:700; }}
    .mid {{ color:var(--mid); font-weight:700; }}
    .bad {{ color:var(--bad); font-weight:700; }}
    .toolbar {{ display:flex; gap:12px; flex-wrap:wrap; margin:16px 0; }}
    select, input {{ background:var(--panel2); color:var(--text); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }}
    details summary {{ cursor:pointer; color:var(--accent); }}
    .scroll {{ overflow:auto; border:1px solid var(--line); border-radius:16px; background:var(--panel); }}
    .muted {{ color:var(--muted); }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Dataset analysis report</h1>
    <div class="sub"><strong>{html.escape(dataset_name)}</strong> · generated at {html.escape(generated_at)}</div>

    <div class="grid">
      <div class="card"><div class="label">Total examples</div><div class="value">{structural_summary.get('total_examples', 0)}</div></div>
      <div class="card"><div class="label">Structural issues</div><div class="value">{structural_summary.get('structural_issue_count', 0)}</div></div>
      <div class="card"><div class="label">Duplicate questions</div><div class="value">{structural_summary.get('duplicate_question_count', 0)}</div></div>
      <div class="card"><div class="label">Metric errors</div><div class="value">{metric_summary.get('metric_error_count', 0)}</div></div>
      {type_cards}
    </div>

    <div class="tables">
      {render_kv_table(structural_summary, 'Structural summary')}
      {render_summary_table(metric_summary['overall'], 'Metric summary — overall')}
    </div>

    <h2>Metric summary by type</h2>
    <div class="tables">
      {render_summary_table(metric_summary['by_type']['single'], 'Single')}
      {render_summary_table(metric_summary['by_type']['multi'], 'Multi')}
    </div>
    <div class="tables">
      {render_summary_table(metric_summary['by_type']['cannot_answer'], 'Cannot answer')}
      {render_kv_table({
          'single_count': len(by_type['single']),
          'multi_count': len(by_type['multi']),
          'cannot_answer_count': len(by_type['cannot_answer'])
      }, 'Type counts')}
    </div>

    <h2>Structural issues</h2>
    <div class="scroll">
      <table>
        <thead><tr><th>Type</th><th>Question</th><th>Issues</th></tr></thead>
        <tbody>{issue_rows or '<tr><td colspan="3">No structural issues found.</td></tr>'}</tbody>
      </table>
    </div>

    <h2>Full examples</h2>
    <div class="toolbar">
      <select id="typeFilter">
        <option value="all">All types</option>
        <option value="single">single</option>
        <option value="multi">multi</option>
        <option value="cannot_answer">cannot_answer</option>
      </select>
      <select id="issueFilter">
        <option value="all">All issue states</option>
        <option value="1">Only with issues</option>
        <option value="0">Only without issues</option>
      </select>
      <input id="searchBox" type="text" placeholder="Search question / answer / id" />
    </div>
    <div class="scroll">
      <table id="examplesTable">
        <thead>
          <tr>
            <th>#</th>
            <th>Type</th>
            <th>Question</th>
            <th>Ground truth</th>
            <th>Reference IDs</th>
            <th>Issues</th>
            <th>Relevancy</th>
            <th>Faithfulness</th>
            <th>Ctx precision</th>
            <th>Ctx recall</th>
            <th>Metric error</th>
            <th>Reference descriptions</th>
          </tr>
        </thead>
        <tbody>
          {''.join(example_rows)}
        </tbody>
      </table>
    </div>
  </div>
  <script>
    const typeFilter = document.getElementById('typeFilter');
    const issueFilter = document.getElementById('issueFilter');
    const searchBox = document.getElementById('searchBox');
    const rows = Array.from(document.querySelectorAll('#examplesTable tbody tr'));

    function applyFilters() {{
      const typeValue = typeFilter.value;
      const issueValue = issueFilter.value;
      const q = searchBox.value.trim().toLowerCase();

      for (const row of rows) {{
        const rowType = row.dataset.type;
        const rowIssue = row.dataset.issue;
        const text = row.innerText.toLowerCase();

        const typeOk = typeValue === 'all' || rowType === typeValue;
        const issueOk = issueValue === 'all' || rowIssue === issueValue;
        const searchOk = !q || text.includes(q);

        row.style.display = (typeOk && issueOk && searchOk) ? '' : 'none';
      }}
    }}

    typeFilter.addEventListener('change', applyFilters);
    issueFilter.addEventListener('change', applyFilters);
    searchBox.addEventListener('input', applyFilters);
  </script>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")
"""
Visualize model comparison examples in a readable HTML format.

Usage:
    python src/visualize_examples.py
    python src/visualize_examples.py --input comparison_examples.json --output my_examples.html
"""

import argparse
import json
import html
from pathlib import Path
from datetime import datetime


def categorize_examples(examples):
    """Categorize examples by performance."""
    categories = {
        "both_correct": [],
        "trained_only": [],
        "base_only": [],
        "both_wrong": [],
    }

    for example in examples:
        base_correct = example["base_model"]["is_correct"]
        trained_correct = example["trained_model"]["is_correct"]

        if base_correct and trained_correct:
            categories["both_correct"].append(example)
        elif trained_correct and not base_correct:
            categories["trained_only"].append(example)
        elif base_correct and not trained_correct:
            categories["base_only"].append(example)
        else:
            categories["both_wrong"].append(example)

    return categories


def generate_html(data, output_file):
    """Generate an HTML file with nicely formatted examples."""

    examples = data["examples"]
    categories = categorize_examples(examples)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison Examples</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}

        h1 {{
            color: #333;
            text-align: center;
            border-bottom: 3px solid #4ECDC4;
            padding-bottom: 10px;
        }}

        .metadata {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}

        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}

        .summary-item {{
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}

        .summary-item.success {{
            background-color: #d4edda;
            border-left: 4px solid #28a745;
        }}

        .summary-item.improvement {{
            background-color: #cfe2ff;
            border-left: 4px solid #0d6efd;
        }}

        .summary-item.regression {{
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
        }}

        .summary-item.fail {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
        }}

        .summary-item .number {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .summary-item .label {{
            font-size: 0.9em;
            color: #666;
        }}

        .category {{
            margin-bottom: 40px;
        }}

        .category-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .category-header.success {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}

        .category-header.improvement {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}

        .category-header.regression {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }}

        .category-header.fail {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}

        .example {{
            background: white;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        .example-header {{
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 2px solid #e9ecef;
        }}

        .question {{
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}

        .gold-answer {{
            color: #28a745;
            font-weight: bold;
            font-size: 1.1em;
        }}

        .comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
        }}

        .model-response {{
            padding: 20px;
            border-right: 2px solid #e9ecef;
        }}

        .model-response:last-child {{
            border-right: none;
        }}

        .model-header {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 2px solid #e9ecef;
        }}

        .model-header.base {{
            color: #dc3545;
        }}

        .model-header.trained {{
            color: #007bff;
        }}

        .answer-box {{
            margin: 15px 0;
            padding: 12px;
            border-radius: 5px;
            font-weight: bold;
        }}

        .answer-box.correct {{
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            color: #155724;
        }}

        .answer-box.incorrect {{
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            color: #721c24;
        }}

        .completion {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
            word-wrap: break-word;
            margin-top: 10px;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            margin-left: 8px;
        }}

        .badge.format-yes {{
            background-color: #28a745;
            color: white;
        }}

        .badge.format-no {{
            background-color: #dc3545;
            color: white;
        }}

        .nav {{
            position: sticky;
            top: 0;
            background: white;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 8px;
            margin-bottom: 20px;
            z-index: 100;
        }}

        .nav a {{
            margin-right: 15px;
            color: #007bff;
            text-decoration: none;
            font-weight: bold;
        }}

        .nav a:hover {{
            text-decoration: underline;
        }}

        @media (max-width: 768px) {{
            .comparison {{
                grid-template-columns: 1fr;
            }}

            .model-response {{
                border-right: none;
                border-bottom: 2px solid #e9ecef;
            }}

            .model-response:last-child {{
                border-bottom: none;
            }}
        }}
    </style>
</head>
<body>
    <h1>Model Comparison Examples</h1>

    <div class="metadata">
        <strong>Base Model:</strong> {data['base_model']} |
        <strong>Trained Model:</strong> {data['trained_model']}<br>
        <strong>Total Examples:</strong> {data['total_examples']} |
        <strong>Generated:</strong> {data['timestamp']}
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="summary-item success">
                <div class="number">{len(categories['both_correct'])}</div>
                <div class="label">Both Correct</div>
            </div>
            <div class="summary-item improvement">
                <div class="number">{len(categories['trained_only'])}</div>
                <div class="label">Trained Only Correct<br>(Improvement)</div>
            </div>
            <div class="summary-item regression">
                <div class="number">{len(categories['base_only'])}</div>
                <div class="label">Base Only Correct<br>(Regression)</div>
            </div>
            <div class="summary-item fail">
                <div class="number">{len(categories['both_wrong'])}</div>
                <div class="label">Both Wrong</div>
            </div>
        </div>
    </div>

    <div class="nav">
        <strong>Jump to:</strong>
        <a href="#both-correct">Both Correct ({len(categories['both_correct'])})</a>
        <a href="#trained-only">Trained Only ({len(categories['trained_only'])})</a>
        <a href="#base-only">Base Only ({len(categories['base_only'])})</a>
        <a href="#both-wrong">Both Wrong ({len(categories['both_wrong'])})</a>
    </div>
"""

    # Generate sections for each category
    category_configs = [
        ("trained_only", "trained-only", "Trained Model Correct, Base Model Wrong (Improvements)", "improvement"),
        ("both_correct", "both-correct", "Both Models Correct", "success"),
        ("both_wrong", "both-wrong", "Both Models Wrong", "fail"),
        ("base_only", "base-only", "Base Model Correct, Trained Model Wrong (Regressions)", "regression"),
    ]

    for cat_key, cat_id, cat_title, cat_class in category_configs:
        cat_examples = categories[cat_key]
        if not cat_examples:
            continue

        html += f"""
    <div class="category" id="{cat_id}">
        <div class="category-header {cat_class}">
            <h2 style="margin: 0;">{cat_title}</h2>
            <span style="font-size: 1.2em;">{len(cat_examples)} examples</span>
        </div>
"""

        for example in cat_examples:
            html += generate_example_html(example)

        html += "    </div>\n"

    html += """
</body>
</html>
"""

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✓ HTML visualization saved to: {output_file}")


def generate_example_html(example):
    """Generate HTML for a single example."""
    base = example["base_model"]
    trained = example["trained_model"]

    base_status = "correct" if base["is_correct"] else "incorrect"
    trained_status = "correct" if trained["is_correct"] else "incorrect"

    base_format_badge = "format-yes" if base["has_format"] else "format-no"
    trained_format_badge = "format-yes" if trained["has_format"] else "format-no"

    base_format_text = "✓ Format" if base["has_format"] else "✗ No Format"
    trained_format_text = "✓ Format" if trained["has_format"] else "✗ No Format"

    # HTML escape the completions to show <answer> tags properly
    base_completion = html.escape(base['completion'])
    trained_completion = html.escape(trained['completion'])

    return f"""
        <div class="example">
            <div class="example-header">
                <div class="question">Q: {html.escape(example['question'])}</div>
                <div class="gold-answer">Gold Answer: {example['gold_answer']}</div>
            </div>
            <div class="comparison">
                <div class="model-response">
                    <div class="model-header base">Base Model</div>
                    <div class="answer-box {base_status}">
                        Predicted: {base['predicted_answer'] if base['predicted_answer'] is not None else 'None'}
                        {'✓' if base['is_correct'] else '✗'}
                        <span class="badge {base_format_badge}">{base_format_text}</span>
                    </div>
                    <div class="completion">{base_completion}</div>
                </div>
                <div class="model-response">
                    <div class="model-header trained">Trained Model</div>
                    <div class="answer-box {trained_status}">
                        Predicted: {trained['predicted_answer'] if trained['predicted_answer'] is not None else 'None'}
                        {'✓' if trained['is_correct'] else '✗'}
                        <span class="badge {trained_format_badge}">{trained_format_text}</span>
                    </div>
                    <div class="completion">{trained_completion}</div>
                </div>
            </div>
        </div>
"""


def main():
    parser = argparse.ArgumentParser(description="Visualize model comparison examples")
    parser.add_argument(
        "--input",
        type=str,
        default="comparison_examples.json",
        help="Input JSON file with comparison examples (default: comparison_examples.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML file (default: visualisations/comparison_examples.html)",
    )
    args = parser.parse_args()

    # Determine script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Resolve input path
    input_path = Path(args.input)
    if not input_path.is_absolute():
        if (project_root / input_path).exists():
            input_path = project_root / input_path
        elif not input_path.exists():
            print(f"Error: Input file '{args.input}' not found!")
            return

    # Determine output path
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
    else:
        output_path = project_root / "visualisations" / "comparison_examples.html"

    # Create visualisations directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and process data
    print(f"Loading examples from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Generate HTML
    print("Creating HTML visualization...")
    generate_html(data, str(output_path))

    print(f"\nOpen {output_path.absolute()} in your browser to view the examples")


if __name__ == "__main__":
    main()

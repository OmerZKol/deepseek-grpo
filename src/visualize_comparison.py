"""
Visualize model comparison results from comparison_results.json

Generates two separate files:
    1. *_charts.png - Bar chart comparisons
    2. *_table.png - Summary table

Usage:
    python src/visualize_comparison.py
    python src/visualize_comparison.py --input comparison_results.json --output visualisations/my_comparison
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(input_file):
    """Load comparison results from JSON file."""
    with open(input_file, "r") as f:
        return json.load(f)


def create_comparison_charts(results, output_file=None):
    """Create bar chart comparisons."""
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 12))

    # Create grid for subplots - 3x3 grid for charts only
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3, top=0.92, bottom=0.08)

    base = results["base_model"]
    trained = results["trained_model"]
    improvements = results["improvements"]

    # 1. Accuracy Comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['Base Model', 'Trained Model']
    accuracy_values = [base["accuracy"], trained["accuracy"]]
    colors = ['#FF6B6B', '#4ECDC4']
    bars1 = ax1.bar(models, accuracy_values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    # 2. Format Rate Comparison (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    format_values = [base["format_rate"], trained["format_rate"]]
    bars2 = ax2.bar(models, format_values, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Format Compliance (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Format Compliance Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    # 3. Correct Answers (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    correct_values = [base["correct"], trained["correct"]]
    bars3 = ax3.bar(models, correct_values, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Correct Answers', fontsize=11, fontweight='bold')
    ax3.set_title(f'Correct Answers (out of {base["total"]})', fontsize=12, fontweight='bold')

    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')

    # 4. Reward Components (middle left - grouped bar chart)
    ax4 = fig.add_subplot(gs[1, 0])
    reward_types = ['Format\nReward', 'Accuracy\nReward', 'Total\nReward']
    base_rewards = [base["avg_format_reward"], base["avg_accuracy_reward"], base["avg_total_reward"]]
    trained_rewards = [trained["avg_format_reward"], trained["avg_accuracy_reward"], trained["avg_total_reward"]]

    x = np.arange(len(reward_types))
    width = 0.35

    bars4_1 = ax4.bar(x - width/2, base_rewards, width, label='Base Model',
                      color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars4_2 = ax4.bar(x + width/2, trained_rewards, width, label='Trained Model',
                      color='#4ECDC4', alpha=0.8, edgecolor='black')

    ax4.set_ylabel('Average Reward', fontsize=11, fontweight='bold')
    ax4.set_title('Reward Components Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(reward_types)
    ax4.legend()

    # Add value labels
    for bars in [bars4_1, bars4_2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
            
    # 5. Reward Standard Deviations (third row right)
    ax5 = fig.add_subplot(gs[1, 1])
    reward_std_types = ['Format\nReward', 'Accuracy\nReward', 'Total\nReward']
    base_reward_stds = [base["std_format_reward"], base["std_accuracy_reward"], base["std_total_reward"]]
    trained_reward_stds = [trained["std_format_reward"], trained["std_accuracy_reward"], trained["std_total_reward"]]

    x = np.arange(len(reward_std_types))
    width = 0.35

    bars5_1 = ax5.bar(x - width/2, base_reward_stds, width, label='Base Model',
                      color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars5_2 = ax5.bar(x + width/2, trained_reward_stds, width, label='Trained Model',
                      color='#4ECDC4', alpha=0.8, edgecolor='black')

    ax5.set_ylabel('Std Deviation', fontsize=11, fontweight='bold')
    ax5.set_title('Reward Consistency (Lower = More Consistent)', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(reward_std_types)
    ax5.legend()

    # Add value labels
    for bars in [bars5_1, bars5_2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

    # 6. Improvements (middle middle)
    ax6 = fig.add_subplot(gs[1, 2])
    improvement_metrics = ['Accuracy', 'Format Rate']
    improvement_values = [improvements["accuracy"], improvements["format_rate"]]
    bars6 = ax6.bar(improvement_metrics, improvement_values,
                    color=['#95E1D3', '#F38181'], alpha=0.8, edgecolor='black')
    ax6.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax6.set_title('Performance Improvements', fontsize=12, fontweight='bold')
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    for bar in bars6:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold')

    # 7. Reward Improvements (middle right)
    ax7 = fig.add_subplot(gs[2, 0])
    reward_imp_types = ['Format\nReward', 'Accuracy\nReward', 'Total\nReward']
    reward_imp_values = [
        improvements["avg_format_reward"],
        improvements["avg_accuracy_reward"],
        improvements["avg_total_reward"]
    ]
    bars7 = ax7.bar(reward_imp_types, reward_imp_values,
                    color=['#AA96DA', '#FCBAD3', '#FFFFD2'], alpha=0.8, edgecolor='black')
    ax7.set_ylabel('Improvement', fontsize=11, fontweight='bold')
    ax7.set_title('Reward Improvements', fontsize=12, fontweight='bold')
    ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    for bar in bars7:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.3f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=8, fontweight='bold')

    # 8. Response Length Comparison (third row left)
    ax8 = fig.add_subplot(gs[2, 1])
    length_values = [base["mean_response_length"], trained["mean_response_length"]]
    bars8 = ax8.bar(models, length_values, color=colors, alpha=0.8, edgecolor='black')
    ax8.set_ylabel('Mean Response Length (chars)', fontsize=11, fontweight='bold')
    ax8.set_title('Response Length Comparison', fontsize=12, fontweight='bold')
    for bar in bars8:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontweight='bold')

    # 9. Response Length Std Deviation (third row middle)
    ax9 = fig.add_subplot(gs[2, 2])
    length_std_values = [base["std_response_length"], trained["std_response_length"]]
    bars9 = ax9.bar(models, length_std_values, color=colors, alpha=0.8, edgecolor='black')
    ax9.set_ylabel('Std Dev (chars)', fontsize=11, fontweight='bold')
    ax9.set_title('Response Length Variability', fontsize=12, fontweight='bold')

    for bar in bars9:
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontweight='bold')

    # Add title
    fig.suptitle(f'Model Comparison:\n {base["model_id"]} vs {trained["model_id"]}',
                fontsize=16, fontweight='bold')

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
        print(f"✓ Charts saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def create_summary_table(results, output_file=None):
    """Create summary table as a separate figure."""
    base = results["base_model"]
    trained = results["trained_model"]
    improvements = results["improvements"]

    # Create figure for table
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.axis('off')

    # Create summary table data
    table_data = [
        ['Metric', 'Base Model', 'Trained Model', 'Improvement/Diff'],
        ['─' * 30, '─' * 15, '─' * 15, '─' * 15],
        ['Accuracy', f'{base["accuracy"]:.2f}%', f'{trained["accuracy"]:.2f}%',
         f'{improvements["accuracy"]:+.2f}%'],
        ['Format Compliance', f'{base["format_rate"]:.2f}%', f'{trained["format_rate"]:.2f}%',
         f'{improvements["format_rate"]:+.2f}%'],
        ['Correct Answers', f'{base["correct"]}/{base["total"]}',
         f'{trained["correct"]}/{trained["total"]}',
         f'+{trained["correct"] - base["correct"]}'],
        ['', '', '', ''],
        ['Avg Format Reward', f'{base["avg_format_reward"]:.4f}',
         f'{trained["avg_format_reward"]:.4f}',
         f'{improvements["avg_format_reward"]:+.4f}'],
        ['Avg Accuracy Reward', f'{base["avg_accuracy_reward"]:.4f}',
         f'{trained["avg_accuracy_reward"]:.4f}',
         f'{improvements["avg_accuracy_reward"]:+.4f}'],
        ['Avg Total Reward', f'{base["avg_total_reward"]:.4f}',
         f'{trained["avg_total_reward"]:.4f}',
         f'{improvements["avg_total_reward"]:+.4f}'],
        ['', '', '', ''],
        ['Mean Response Length', f'{base["mean_response_length"]:.1f}',
         f'{trained["mean_response_length"]:.1f}',
         f'{improvements["mean_response_length"]:+.1f}'],
        ['Std Response Length', f'{base["std_response_length"]:.1f}',
         f'{trained["std_response_length"]:.1f}', '—'],
        ['', '', '', ''],
        ['Std Format Reward', f'{base["std_format_reward"]:.4f}',
         f'{trained["std_format_reward"]:.4f}', '—'],
        ['Std Accuracy Reward', f'{base["std_accuracy_reward"]:.4f}',
         f'{trained["std_accuracy_reward"]:.4f}', '—'],
        ['Std Total Reward', f'{base["std_total_reward"]:.4f}',
         f'{trained["std_total_reward"]:.4f}', '—'],
    ]

    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.30, 0.23, 0.23, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style the header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style separator rows
    for sep_row in [1, 5, 9, 12]:
        for i in range(4):
            table[(sep_row, i)].set_facecolor('#f0f0f0')

    # Add title and timestamp
    fig.suptitle(f'Model Comparison Summary\n{base["model_id"]} vs {trained["model_id"]}',
                fontsize=16, fontweight='bold', y=0.96)

    # timestamp_text = f'Generated: {results["timestamp"]}'
    # fig.text(0.5, 0.02, timestamp_text, ha='center', fontsize=10, style='italic')

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.3)
        print(f"✓ Table saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def print_summary(results):
    """Print a text summary of the results."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    base = results["base_model"]
    trained = results["trained_model"]
    improvements = results["improvements"]

    print(f"\nBase Model: {base['model_id']}")
    print(f"Trained Model: {trained['model_id']}")
    print(f"Timestamp: {results['timestamp']}")

    print(f"\n{'Metric':<25} {'Base':>15} {'Trained':>15} {'Improvement':>15}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {base['accuracy']:>14.2f}% {trained['accuracy']:>14.2f}% {improvements['accuracy']:>+14.2f}%")
    print(f"{'Format Compliance':<25} {base['format_rate']:>14.2f}% {trained['format_rate']:>14.2f}% {improvements['format_rate']:>+14.2f}%")
    print(f"{'Correct / Total':<25} {base['correct']:>7}/{base['total']:<7} {trained['correct']:>7}/{trained['total']:<7}")

    print(f"\n{'Reward Metrics':<25} {'Base':>15} {'Trained':>15} {'Improvement':>15}")
    print("-" * 70)
    print(f"{'Avg Format Reward':<25} {base['avg_format_reward']:>15.4f} {trained['avg_format_reward']:>15.4f} {improvements['avg_format_reward']:>+15.4f}")
    print(f"{'Avg Accuracy Reward':<25} {base['avg_accuracy_reward']:>15.4f} {trained['avg_accuracy_reward']:>15.4f} {improvements['avg_accuracy_reward']:>+15.4f}")
    print(f"{'Avg Total Reward':<25} {base['avg_total_reward']:>15.4f} {trained['avg_total_reward']:>15.4f} {improvements['avg_total_reward']:>+15.4f}")

    print(f"\n{'Reward Consistency':<25} {'Base':>15} {'Trained':>15}")
    print("-" * 70)
    print(f"{'Std Format Reward':<25} {base['std_format_reward']:>15.4f} {trained['std_format_reward']:>15.4f}")
    print(f"{'Std Accuracy Reward':<25} {base['std_accuracy_reward']:>15.4f} {trained['std_accuracy_reward']:>15.4f}")
    print(f"{'Std Total Reward':<25} {base['std_total_reward']:>15.4f} {trained['std_total_reward']:>15.4f}")

    print(f"\n{'Response Length':<25} {'Base':>15} {'Trained':>15} {'Difference':>15}")
    print("-" * 70)
    print(f"{'Mean Length (chars)':<25} {base['mean_response_length']:>15.1f} {trained['mean_response_length']:>15.1f} {improvements['mean_response_length']:>+15.1f}")
    print(f"{'Std Length (chars)':<25} {base['std_response_length']:>15.1f} {trained['std_response_length']:>15.1f}")

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize model comparison results")
    parser.add_argument(
        "--input",
        type=str,
        default="comparison_results.json",
        help="Input JSON file with comparison results (default: comparison_results.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file prefix for the plots (default: visualisations/model_comparison)",
    )
    args = parser.parse_args()

    # Determine script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Resolve input path (try relative to project root first, then relative to cwd)
    input_path = Path(args.input)
    if not input_path.is_absolute():
        if (project_root / input_path).exists():
            input_path = project_root / input_path
        elif not input_path.exists():
            print(f"Error: Input file '{args.input}' not found!")
            return

    # Determine output paths - default to visualisations directory
    if args.output:
        output_prefix = Path(args.output)
        if not output_prefix.is_absolute():
            output_prefix = project_root / output_prefix
    else:
        # Default output in visualisations directory
        output_prefix = project_root / "visualisations" / "model_comparison"

    # Create output paths for charts and table
    charts_path = output_prefix.parent / f"{output_prefix.stem}_charts.png"
    table_path = output_prefix.parent / f"{output_prefix.stem}_table.png"

    # Create visualisations directory if it doesn't exist
    charts_path.parent.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading results from: {input_path}")
    results = load_results(input_path)

    # Create visualizations
    print("Creating visualizations...")
    create_comparison_charts(results, str(charts_path))
    create_summary_table(results, str(table_path))

    print(f"\n✓ All visualizations complete!")
    print(f"  - Charts: {charts_path}")
    print(f"  - Table: {table_path}")


if __name__ == "__main__":
    main()
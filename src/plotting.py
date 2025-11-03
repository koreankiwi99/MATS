"""
Plotting utilities
"""

import matplotlib.pyplot as plt


def plot_results(results, save_path='results.png', error_bars=None):
    """
    Create bar graph like Figure 1 from paper
    """
    labels = list(results.keys())
    accuracies = [v * 100 for v in results.values()]  # Convert to percentage
    yerr = [error_bars.get(label, 0) * 100 for label in labels]

    # Color scheme
    colors = []
    for label in labels:
        if 'Zero-shot' in label:
            colors.append('purple')
        elif 'Unsupervised' in label or 'ICM' in label:
            colors.append('cyan')
        elif 'Golden' in label:
            colors.append('orange')
        else:
            colors.append('gray')

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, accuracies, color=colors, yerr=yerr, capsize=5,
                   error_kw={'linewidth': 2, 'ecolor': 'black'})

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('TruthfulQA Performance', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if yerr is not None and yerr[i] > 0:
            # Show mean ± std for bars with error
            ax.text(bar.get_x() + bar.get_width()/2., height + yerr[i] + 1,
                    f'{height:.1f}% ± {yerr[i]:.1f}%',
                    ha='center', va='bottom', fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()

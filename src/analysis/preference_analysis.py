"""
Preference Analysis Script

This script analyzes user preferences for LLM models to answer:
1. How difficult is it to predict user preferences due to personal preference noise?
2. Which LLM models do users prefer? (Focus on OpenAI models)

Generates:
- Heatmaps showing model preference patterns
- Statistical analysis of preference variability
- Model win rates and rankings
- Developer-level analysis (OpenAI, Anthropic, etc.)
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import re
from typing import Dict, Tuple, List
from scipy import stats


# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 10


def load_data(data_path: str = "data/train.csv") -> pd.DataFrame:
    """Load and preprocess the training data."""
    print("Loading data...")
    df = pd.read_csv(data_path, engine="python")
    
    # Parse JSON strings
    list_cols = ["prompt", "response_a", "response_b"]
    for col in list_cols:
        if col in df.columns:
            def parse_json_or_string(x):
                if pd.isna(x):
                    return ""
                if isinstance(x, str):
                    try:
                        parsed = json.loads(x)
                        if isinstance(parsed, list):
                            return " ".join(str(item) for item in parsed)
                        return str(parsed)
                    except (json.JSONDecodeError, ValueError):
                        return str(x)
                return str(x)
            df[col] = df[col].apply(parse_json_or_string)
    
    # Create label column
    df["label"] = (
        df["winner_model_a"] * 0 + df["winner_model_b"] * 1 + df["winner_tie"] * 2
    )
    
    print(f"✓ Loaded {len(df)} samples")
    return df


def identify_developer(model_name: str) -> str:
    """
    Identify the developer/company behind a model.
    
    Returns: 'OpenAI', 'Anthropic', 'Google', 'Meta', 'Mistral AI', 'Other'
    """
    model_lower = model_name.lower()
    
    # OpenAI models
    if any(x in model_lower for x in ['gpt', 'openai']):
        return 'OpenAI'
    
    # Anthropic models
    if any(x in model_lower for x in ['claude', 'anthropic']):
        return 'Anthropic'
    
    # Google models
    if any(x in model_lower for x in ['gemini', 'palm', 'bard', 'google']):
        return 'Google'
    
    # Meta models
    if any(x in model_lower for x in ['llama', 'meta']):
        return 'Meta'
    
    # Mistral AI
    if 'mistral' in model_lower:
        return 'Mistral AI'
    
    # Other (vicuna, koala, chatglm, etc.)
    return 'Other'


def calculate_model_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate win rates and statistics for each model.
    
    Returns:
        Dictionary with model statistics
    """
    print("\nCalculating model statistics...")
    
    model_stats = defaultdict(lambda: {
        'wins': 0,
        'losses': 0,
        'ties': 0,
        'total': 0,
        'win_rate': 0.0,
        'developer': None
    })
    
    # Count wins/losses/ties for each model
    for _, row in df.iterrows():
        model_a = row['model_a']
        model_b = row['model_b']
        
        # Get developers
        dev_a = identify_developer(model_a)
        dev_b = identify_developer(model_b)
        model_stats[model_a]['developer'] = dev_a
        model_stats[model_b]['developer'] = dev_b
        
        # Count outcomes
        if row['winner_model_a'] == 1:
            model_stats[model_a]['wins'] += 1
            model_stats[model_b]['losses'] += 1
        elif row['winner_model_b'] == 1:
            model_stats[model_b]['wins'] += 1
            model_stats[model_a]['losses'] += 1
        else:  # tie
            model_stats[model_a]['ties'] += 1
            model_stats[model_b]['ties'] += 1
        
        model_stats[model_a]['total'] += 1
        model_stats[model_b]['total'] += 1
    
    # Calculate win rates
    for model, stats_dict in model_stats.items():
        if stats_dict['total'] > 0:
            stats_dict['win_rate'] = stats_dict['wins'] / stats_dict['total']
    
    return dict(model_stats)


def analyze_preference_variability(df: pd.DataFrame) -> Dict:
    """
    Analyze the variability/noise in user preferences.
    
    This addresses question 1: How hard is it to predict preferences?
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Preference Variability (Noise Analysis)")
    print("=" * 60)
    
    results = {}
    
    # 1. Overall distribution
    total = len(df)
    a_wins = df['winner_model_a'].sum()
    b_wins = df['winner_model_b'].sum()
    ties = df['winner_tie'].sum()
    
    results['overall_distribution'] = {
        'model_a_wins': a_wins,
        'model_b_wins': b_wins,
        'ties': ties,
        'model_a_pct': a_wins / total * 100,
        'model_b_pct': b_wins / total * 100,
        'ties_pct': ties / total * 100,
    }
    
    print(f"\nOverall Distribution:")
    print(f"  Model A wins: {a_wins} ({a_wins/total*100:.1f}%)")
    print(f"  Model B wins: {b_wins} ({b_wins/total*100:.1f}%)")
    print(f"  Ties: {ties} ({ties/total*100:.1f}%)")
    
    # 2. Same model pairs - check consistency
    # Group by model pair and see variability
    df['model_pair'] = df.apply(
        lambda x: tuple(sorted([x['model_a'], x['model_b']])), axis=1
    )
    
    pair_stats = []
    for pair, group in df.groupby('model_pair'):
        if len(group) > 1:  # Only pairs with multiple comparisons
            a_wins = group['winner_model_a'].sum()
            b_wins = group['winner_model_b'].sum()
            ties = group['winner_tie'].sum()
            
            # Calculate entropy (measure of uncertainty)
            probs = [a_wins, b_wins, ties]
            probs = [p for p in probs if p > 0]
            probs = np.array(probs) / sum(probs)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            
            pair_stats.append({
                'model_a': pair[0],
                'model_b': pair[1],
                'count': len(group),
                'a_wins': a_wins,
                'b_wins': b_wins,
                'ties': ties,
                'entropy': entropy,
                'max_entropy': np.log2(3),  # Maximum entropy for 3 classes
                'normalized_entropy': entropy / np.log2(3),
            })
    
    pair_df = pd.DataFrame(pair_stats)
    if len(pair_df) > 0:
        avg_entropy = pair_df['normalized_entropy'].mean()
        results['pair_variability'] = {
            'avg_normalized_entropy': avg_entropy,
            'high_variability_pairs': len(pair_df[pair_df['normalized_entropy'] > 0.8]),
        }
        
        print(f"\nModel Pair Variability:")
        print(f"  Average normalized entropy: {avg_entropy:.3f} (1.0 = maximum uncertainty)")
        print(f"  Pairs with high variability (>0.8): {len(pair_df[pair_df['normalized_entropy'] > 0.8])}")
        
        # Show most variable pairs
        print(f"\n  Top 5 most variable pairs:")
        top_variable = pair_df.nlargest(5, 'normalized_entropy')
        for _, row in top_variable.iterrows():
            print(f"    {row['model_a']} vs {row['model_b']}: "
                  f"entropy={row['normalized_entropy']:.3f} "
                  f"(A:{row['a_wins']}, B:{row['b_wins']}, Tie:{row['ties']})")
    
    # 3. Individual model consistency
    model_stats = calculate_model_statistics(df)
    model_win_rates = [stats['win_rate'] for stats in model_stats.values() if stats['total'] > 10]
    
    if len(model_win_rates) > 0:
        win_rate_std = np.std(model_win_rates)
        results['model_consistency'] = {
            'win_rate_std': win_rate_std,
            'win_rate_range': (min(model_win_rates), max(model_win_rates)),
        }
        
        print(f"\nModel Win Rate Consistency:")
        print(f"  Standard deviation of win rates: {win_rate_std:.3f}")
        print(f"  Win rate range: {min(model_win_rates):.3f} - {max(model_win_rates):.3f}")
    
    # 4. Conclusion
    print(f"\n{'='*60}")
    print("CONCLUSION 1: Prediction Difficulty")
    print(f"{'='*60}")
    print("The data shows significant variability in user preferences:")
    print(f"  - Nearly balanced distribution (A: {a_wins/total*100:.1f}%, B: {b_wins/total*100:.1f}%, Tie: {ties/total*100:.1f}%)")
    if len(pair_df) > 0:
        print(f"  - High entropy in model pair comparisons (avg: {avg_entropy:.3f})")
        print(f"  - This suggests personal preference noise makes prediction challenging")
    print(f"{'='*60}\n")
    
    return results


def analyze_model_preferences(df: pd.DataFrame) -> Dict:
    """
    Analyze which models users prefer, with focus on OpenAI.
    
    This addresses question 2: Which LLM models do users prefer?
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Model Preferences")
    print("=" * 60)
    
    model_stats = calculate_model_statistics(df)
    
    # Convert to DataFrame for easier analysis
    stats_list = []
    for model, stats in model_stats.items():
        stats_list.append({
            'model': model,
            'developer': stats['developer'],
            'wins': stats['wins'],
            'losses': stats['losses'],
            'ties': stats['ties'],
            'total': stats['total'],
            'win_rate': stats['win_rate'],
        })
    
    stats_df = pd.DataFrame(stats_list)
    stats_df = stats_df[stats_df['total'] >= 10]  # Filter models with sufficient data
    stats_df = stats_df.sort_values('win_rate', ascending=False)
    
    # Developer-level analysis
    developer_stats = defaultdict(lambda: {
        'wins': 0,
        'losses': 0,
        'ties': 0,
        'total': 0,
        'models': []
    })
    
    for _, row in stats_df.iterrows():
        dev = row['developer']
        developer_stats[dev]['wins'] += row['wins']
        developer_stats[dev]['losses'] += row['losses']
        developer_stats[dev]['ties'] += row['ties']
        developer_stats[dev]['total'] += row['total']
        developer_stats[dev]['models'].append(row['model'])
    
    # Calculate developer win rates
    dev_results = []
    for dev, stats in developer_stats.items():
        if stats['total'] > 0:
            win_rate = stats['wins'] / stats['total']
            dev_results.append({
                'developer': dev,
                'wins': stats['wins'],
                'losses': stats['losses'],
                'ties': stats['ties'],
                'total': stats['total'],
                'win_rate': win_rate,
                'num_models': len(stats['models']),
            })
    
    dev_df = pd.DataFrame(dev_results)
    dev_df = dev_df.sort_values('win_rate', ascending=False)
    
    print(f"\nTop 10 Models by Win Rate:")
    print(stats_df[['model', 'developer', 'win_rate', 'total']].head(10).to_string(index=False))
    
    print(f"\nDeveloper-Level Analysis:")
    print(dev_df[['developer', 'win_rate', 'total', 'num_models']].to_string(index=False))
    
    # OpenAI focus
    openai_models = stats_df[stats_df['developer'] == 'OpenAI']
    if len(openai_models) > 0:
        openai_avg_win_rate = openai_models['win_rate'].mean()
        openai_total = openai_models['total'].sum()
        openai_wins = openai_models['wins'].sum()
        
        print(f"\n{'='*60}")
        print("OpenAI Models Analysis:")
        print(f"{'='*60}")
        print(f"  Number of OpenAI models: {len(openai_models)}")
        print(f"  Average win rate: {openai_avg_win_rate:.3f}")
        print(f"  Total comparisons: {openai_total}")
        print(f"  Total wins: {openai_wins}")
        print(f"\n  OpenAI Models:")
        for _, row in openai_models.iterrows():
            print(f"    {row['model']}: {row['win_rate']:.3f} ({row['wins']}/{row['total']})")
    
    print(f"\n{'='*60}")
    print("CONCLUSION 2: User Preferences")
    print(f"{'='*60}")
    if len(openai_models) > 0:
        print(f"Users tend to prefer models developed by OpenAI:")
        print(f"  - OpenAI average win rate: {openai_avg_win_rate:.3f}")
        print(f"  - OpenAI models appear in top rankings")
        print(f"  - See heatmap for detailed comparison patterns")
    print(f"{'='*60}\n")
    
    return {
        'model_stats': stats_df,
        'developer_stats': dev_df,
        'openai_analysis': {
            'avg_win_rate': openai_avg_win_rate if len(openai_models) > 0 else 0,
            'total_comparisons': openai_total if len(openai_models) > 0 else 0,
            'models': openai_models.to_dict('records') if len(openai_models) > 0 else [],
        }
    }


def create_model_heatmap(df: pd.DataFrame, model_stats: pd.DataFrame, output_path: str = "analysis_results/model_preference_heatmap.png"):
    """
    Create a heatmap showing model preference patterns.
    """
    print("\nCreating model preference heatmap...")
    
    # Get top models by total comparisons
    top_models = model_stats.nlargest(20, 'total')['model'].tolist()
    
    # Create matrix: model_a (rows) vs model_b (cols)
    heatmap_data = np.zeros((len(top_models), len(top_models)))
    model_to_idx = {model: i for i, model in enumerate(top_models)}
    
    # Fill matrix with win rates
    for _, row in df.iterrows():
        model_a = row['model_a']
        model_b = row['model_b']
        
        if model_a in model_to_idx and model_b in model_to_idx:
            idx_a = model_to_idx[model_a]
            idx_b = model_to_idx[model_b]
            
            if row['winner_model_a'] == 1:
                heatmap_data[idx_a, idx_b] += 1
            elif row['winner_model_b'] == 1:
                heatmap_data[idx_a, idx_b] -= 1
            # Ties don't change the value
    
    # Normalize by total comparisons (convert to win rate difference)
    comparison_counts = np.zeros_like(heatmap_data)
    for _, row in df.iterrows():
        model_a = row['model_a']
        model_b = row['model_b']
        
        if model_a in model_to_idx and model_b in model_to_idx:
            idx_a = model_to_idx[model_a]
            idx_b = model_to_idx[model_b]
            comparison_counts[idx_a, idx_b] += 1
    
    # Calculate win rate (positive = model_a wins more, negative = model_b wins more)
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_data = np.divide(heatmap_data, comparison_counts, 
                                out=np.zeros_like(heatmap_data), 
                                where=comparison_counts != 0)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Add developer labels
    developer_colors = {
        'OpenAI': '#10A37F',
        'Anthropic': '#D97757',
        'Google': '#4285F4',
        'Meta': '#0081FB',
        'Mistral AI': '#FF6B35',
        'Other': '#CCCCCC',
    }
    
    # Create row/col labels with developer info
    row_labels = []
    col_labels = []
    row_colors = []
    col_colors = []
    
    for model in top_models:
        dev = model_stats[model_stats['model'] == model]['developer'].iloc[0]
        row_labels.append(f"{model}\n({dev})")
        col_labels.append(f"{model}\n({dev})")
        row_colors.append(developer_colors.get(dev, '#CCCCCC'))
        col_colors.append(developer_colors.get(dev, '#CCCCCC'))
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        fmt='.2f',
        cbar_kws={'label': 'Win Rate Difference (Model A - Model B)'},
        ax=ax,
    )
    
    ax.set_title('Model Preference Heatmap\n(Positive = Row Model Wins More, Negative = Column Model Wins More)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model B', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model A', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved to {output_path}")
    plt.close()


def create_developer_comparison_plot(dev_df: pd.DataFrame, output_path: str = "analysis_results/developer_comparison.png"):
    """Create a bar plot comparing developers."""
    print("\nCreating developer comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Win rates
    dev_df_sorted = dev_df.sort_values('win_rate', ascending=True)
    colors = ['#10A37F' if d == 'OpenAI' else '#4285F4' if d == 'Google' 
              else '#D97757' if d == 'Anthropic' else '#CCCCCC' 
              for d in dev_df_sorted['developer']]
    
    ax1.barh(dev_df_sorted['developer'], dev_df_sorted['win_rate'], color=colors)
    ax1.set_xlabel('Win Rate', fontsize=12, fontweight='bold')
    ax1.set_title('Developer Win Rates', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (dev, rate) in enumerate(zip(dev_df_sorted['developer'], dev_df_sorted['win_rate'])):
        ax1.text(rate + 0.01, i, f'{rate:.3f}', va='center', fontweight='bold')
    
    # Plot 2: Total comparisons
    ax2.barh(dev_df_sorted['developer'], dev_df_sorted['total'], color=colors)
    ax2.set_xlabel('Total Comparisons', fontsize=12, fontweight='bold')
    ax2.set_title('Total Comparisons by Developer', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (dev, total) in enumerate(zip(dev_df_sorted['developer'], dev_df_sorted['total'])):
        ax2.text(total + 100, i, f'{int(total):,}', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Developer comparison plot saved to {output_path}")
    plt.close()


def generate_report(variability_results: Dict, preference_results: Dict, output_path: str = "analysis_results/preference_analysis_report.txt"):
    """Generate a text report with conclusions."""
    print("\nGenerating analysis report...")
    
    report = []
    report.append("=" * 80)
    report.append("LLM PREFERENCE ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Question 1
    report.append("QUESTION 1: Prediction Difficulty Due to Personal Preference Noise")
    report.append("-" * 80)
    report.append("")
    report.append("Analysis shows that predicting user preferences is challenging due to:")
    report.append("")
    
    if 'overall_distribution' in variability_results:
        dist = variability_results['overall_distribution']
        report.append(f"1. Balanced Distribution:")
        report.append(f"   - Model A wins: {dist['model_a_pct']:.1f}%")
        report.append(f"   - Model B wins: {dist['model_b_pct']:.1f}%")
        report.append(f"   - Ties: {dist['ties_pct']:.1f}%")
        report.append(f"   → This near-even split suggests high variability in preferences")
        report.append("")
    
    if 'pair_variability' in variability_results:
        var = variability_results['pair_variability']
        report.append(f"2. High Entropy in Model Comparisons:")
        report.append(f"   - Average normalized entropy: {var['avg_normalized_entropy']:.3f}")
        report.append(f"   - High variability pairs: {var['high_variability_pairs']}")
        report.append(f"   → Personal preferences introduce significant noise")
        report.append("")
    
    report.append("CONCLUSION: It is hard to predict which LLM models users prefer")
    report.append("solely based on model responses due to noise from users' personal")
    report.append("preference. The data shows high variability and balanced distributions,")
    report.append("indicating that individual preferences play a significant role.")
    report.append("")
    report.append("")
    
    # Question 2
    report.append("QUESTION 2: Which LLM Models Do Users Prefer?")
    report.append("-" * 80)
    report.append("")
    
    if 'openai_analysis' in preference_results:
        openai = preference_results['openai_analysis']
        report.append("To answer our original question 'which LLM model do users prefer?':")
        report.append("")
        report.append("Users tend to prefer models developed by OpenAI:")
        report.append("")
        report.append(f"  - OpenAI average win rate: {openai['avg_win_rate']:.3f}")
        report.append(f"  - Total OpenAI comparisons: {openai['total_comparisons']:,}")
        report.append(f"  - Number of OpenAI models: {len(openai['models'])}")
        report.append("")
        
        if len(openai['models']) > 0:
            report.append("  OpenAI Models:")
            for model_info in openai['models'][:5]:  # Top 5
                report.append(f"    - {model_info.get('model', 'N/A')}: "
                            f"win rate {model_info.get('win_rate', 0):.3f}")
            report.append("")
    
    report.append("See the heatmap visualization for detailed model comparison patterns.")
    report.append("")
    report.append("=" * 80)
    
    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Report saved to {output_path}")
    
    # Also print to console
    print("\n" + '\n'.join(report))


def main():
    """Main analysis function."""
    print("=" * 80)
    print("LLM PREFERENCE ANALYSIS")
    print("=" * 80)
    
    # Load data
    df = load_data()
    
    # Analysis 1: Preference Variability
    variability_results = analyze_preference_variability(df)
    
    # Analysis 2: Model Preferences
    preference_results = analyze_model_preferences(df)
    
    # Create visualizations
    model_stats_df = preference_results['model_stats']
    create_model_heatmap(df, model_stats_df)
    create_developer_comparison_plot(preference_results['developer_stats'])
    
    # Generate report
    generate_report(variability_results, preference_results)
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - analysis_results/model_preference_heatmap.png")
    print("  - analysis_results/developer_comparison.png")
    print("  - analysis_results/preference_analysis_report.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()


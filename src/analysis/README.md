# Preference Analysis

This directory contains scripts for analyzing user preferences for LLM models.

## Scripts

### `preference_analysis.py`

Comprehensive analysis script that answers two key questions:

1. **Prediction Difficulty**: How hard is it to predict user preferences due to personal preference noise?
2. **Model Preferences**: Which LLM models do users prefer? (Focus on OpenAI)

## Usage

```bash
python src/analysis/preference_analysis.py
```

## Output

The script generates:

1. **`analysis_results/preference_analysis_report.txt`**: Text report with conclusions
2. **`analysis_results/model_preference_heatmap.png`**: Heatmap showing model preference patterns
3. **`analysis_results/developer_comparison.png`**: Bar charts comparing developers (OpenAI, Anthropic, Google, etc.)

## Key Findings

### Question 1: Prediction Difficulty

- **Balanced Distribution**: Model A wins 34.9%, Model B wins 34.2%, Ties 30.9%
- **High Entropy**: Average normalized entropy of 0.893 (1.0 = maximum uncertainty)
- **High Variability**: 1,038 model pairs show high variability in preferences

**Conclusion**: It is hard to predict which LLM models users prefer solely based on model responses due to noise from users' personal preference.

### Question 2: Model Preferences

- **OpenAI Dominance**: OpenAI models have the highest average win rate (0.403)
- **Top Models**: 
  - gpt-4-1106-preview: 0.551 win rate
  - gpt-3.5-turbo-0314: 0.546 win rate
  - gpt-4-0125-preview: 0.514 win rate
- **Total Comparisons**: 31,840 comparisons involving OpenAI models

**Conclusion**: To answer our original question "which LLM model do users prefer?", users tend to prefer models developed by OpenAI.

## Visualizations

- **Heatmap**: Shows win rate differences between model pairs (positive = row model wins more, negative = column model wins more)
- **Developer Comparison**: Bar charts showing win rates and total comparisons by developer

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scipy

All dependencies should already be installed in the project environment.


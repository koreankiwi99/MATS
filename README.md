# Simplified ICM on TruthfulQA

### AI Usage
1. Claude Code (task 1)
- Conducted original repository review and generated CORE_FUNCTIONS_ANALYSIS.md
- Debugged and resolved API integration issues (specifically 'chat' completion functionality)
- Implemented additional experiments (Random ICL and Consistency ICL baselines) in `experiments/` due to time constraints

2. GitHub Copilot (task 1)
- Added inline code comments
- Performed line-by-line code refinements and optimizations

3. Claude (task 2)
- For checking reading comprehension \& brainstorming : https://claude.ai/share/87f99408-13a4-4608-aef1-29a94dfb788d
- For checking grammar \& shortening sentences: https://claude.ai/share/86c03f98-767e-4217-9bcb-266ccb92472c

Apart from the AI-assisted tasks listed above, all other development work was completed independently.

## Setup
- Install dependencies:
```bash
pip install -r requirements.txt
```
- Set Hyperbolic API key:
```bash
export HYPERBOLIC_API_KEY="your_api_key_here"
```

## Data
- `truthfulqa_train.json` (256)
- `truthfulqa_test.json` (100)

## Run

```bash
python main_multi_trial.py
```

1. Run zero-shot baseline with Base model (Llama-3.1-405B)
2. Run zero-shot baseline with Chat model (Llama-3.1-405B-Instruct)
3. Run ICM algorithm 3 times with different seeds (for robustness)
4. Run golden labels baseline
5. Generate `results/results_multi_trial.png` with bar graph

## Code Structure

```
.
├── data/
│   ├── truthfulqa_train.json
│   └── truthfulqa_test.json
├── src/
│   ├── api_utils.py        # Hyperbolic API wrapper
│   ├── data_utils.py       # Data loading
│   ├── logprob_utils.py    # Logprob extraction
│   ├── prompt_utils.py     # Prompt creation
│   ├── icm.py              # Main ICM algorithm
│   ├── evaluation.py       # Evaluation utilities
│   └── plotting.py         # Plotting utilities
├── experiments/            # Additional experiments (by Claude Code)
│   ├── test_random_icl_multi.py     # Random ICL baseline
│   └── test_consistency_icl.py      # Consistency ICL baseline
├── results/                # Output directory
│   ├── results_multi_trial.png      # Bar graph
│   ├── results_multi_trial.json     # Numerical results
│   ├── icm_labeled_trial*.json      # ICM labeled data
│   ├── results_random_icl_multi.json    # Random ICL results
│   └── results_consistency_icl.json     # Consistency ICL results
├── main_multi_trial.py     # Main script
├── requirements.txt
├── CORE_FUNCTIONS_ANALYSIS.md # Report by Claude Code to analyze and adapt original repo
└── README.md
```

## Hyperparameters

- `NUM_SEED`: 8 (initial random labels)
- `NUM_ITERATIONS`: 500
- `ALPHA`: 50 (weight for mutual predictability)
- `T_0`: 10 (initial temperature)
- `T_min`: 0.01 (final temperature)
- `β`: 0.99 (temperature decay rate, per paper Appendix A.1)

## Additional Experiments

Due to time constraints, two additional baseline experiments were implemented by Claude Code:

1. **Random ICL** (`experiments/test_random_icl_multi.py`): Tests ICL with random labels (no optimization)
2. **Consistency ICL** (`experiments/test_consistency_icl.py`): Tests ICL with logical consistency fixing only (no iterative optimization)

Results are saved in `results/results_random_icl_multi.json` and `results/results_consistency_icl.json`.

## Code Adapted From
All code except api_utils.py adapted from the official repository:
- `src/experiments/ICM.py`
- `src/model_querying/prompt_creation.py`
- `src/model_querying/solution_extraction.py`
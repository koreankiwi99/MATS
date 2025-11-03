# Core Functions Analysis: Unsupervised-Elicitation Repository

## Overview

This document provides a comprehensive breakdown of the core functions implementing the **Internal Coherence Maximization (ICM)** algorithm from the paper "Unsupervised Elicitation of Language Models" (arXiv:2506.10139).

---

## 🔍 Repository Structure

```
Unsupervised-Elicitation/
├── core/
│   ├── llm_api/
│   │   ├── llm.py              # Unified API interface for LLMs
│   │   ├── anthropic_llm.py    # Claude API wrapper
│   │   ├── openai_llm.py       # OpenAI/vLLM API wrapper
│   │   └── base_llm.py         # Base protocol
│   └── utils.py                # Environment setup
│
├── src/
│   ├── experiments/
│   │   ├── ICM.py              # ⭐ Main ICM algorithm
│   │   └── ICM_tools.py        # Helper functions (ConsistencyFix)
│   │
│   ├── model_querying/
│   │   ├── prompt_creation.py  # Few-shot prompt generation
│   │   └── solution_extraction.py  # Logprob extraction
│   │
│   ├── pipeline/
│   │   └── pipeline.py         # Dependency graph orchestration
│   │
│   └── tools/
│       ├── dataloaders.py      # Dataset loading utilities
│       └── path_utils.py       # Path management
│
└── data/                        # Downloaded datasets (TruthfulQA, GSM8K, Alpaca)
```

---

## 1. Main ICM Algorithm (`src/experiments/ICM.py`)

### Core Components

#### 1.1 Energy Function (Scoring Function)

**Location:** `ICM.py:301`

```python
def get_energy(metric, alpha):
    return alpha * metric["train_prob"] - metric["inconsistent_num"]
```

**Purpose:** Implements the scoring function **U(D)** from the paper.

**Formula:**
```
U(D) = α · P_θ(D) - I(D)

Where:
- α: Hyperparameter balancing mutual predictability vs logical consistency (default: 20-50)
- P_θ(D): Sum of log probabilities (mutual predictability score)
- I(D): Number of logical inconsistencies
```

**Parameters:**
- `metric`: Dictionary containing:
  - `train_prob`: Average log probability of labels
  - `inconsistent_num`: Count of logical inconsistencies
- `alpha`: Weight coefficient (typically 20-50)

**Returns:** Scalar energy value (higher is better)

---

#### 1.2 Temperature Schedule (Simulated Annealing)

**Location:** `ICM.py:279`

```python
def get_temperature(iteration, initial_temp, final_temp, decay_rate, schedule="exp"):
    if schedule == "exp":
        return max(final_temp, initial_temp * (decay_rate ** iteration))
    elif schedule == "log":
        return max(final_temp, initial_temp / (1 + 2 * np.log(1 + iteration)))
```

**Purpose:** Controls acceptance probability in simulated annealing.

**Schedules:**
- **Exponential:** `T(t) = max(T_min, T_0 · β^t)`
- **Logarithmic:** `T(t) = max(T_min, T_0 / (1 + 2·log(1+t)))`

**Default Hyperparameters:**
- `T_0 = 10` (initial temperature)
- `T_min = 0.01` (final temperature)
- `β = 0.99` (decay rate)
- `schedule = "log"` (logarithmic decay)

**Usage:** Higher temperature → more exploration (accepts worse solutions)

---

#### 1.3 Mutual Predictability Scoring

**Location:** `ICM.py:254`

```python
async def predict_assignment(model, example, demonstrations):
    demos = [v for k, v in demonstrations.items()
             if k != example["uid"] and v["label"] is not None]

    responses = await model_api(
        model,
        get_judge_prompt_fewshot(example, demos, pipeline=False),
        logprobs=20,
        max_tokens=1,
        parse_fn=extract_claim_logprobs,
    )

    score = responses[0][0]["score"]  # Log P(True) - Log P(False)
    new_label = score > 0
    return int(new_label)
```

**Purpose:** Predicts label for an example using all other labeled examples as context.

**Process:**
1. Create few-shot prompt with all other labeled examples
2. Query model for logprobs of "True" vs "False"
3. Extract score (log-odds ratio)
4. Assign label based on sign of score

**Key Insight:** This implements the mutual predictability term:
```
P_θ(y_i | x_i, D \ {(x_i, y_i)})
```

---

#### 1.4 Consistency Fixing (Algorithm 2)

**Location:** `ICM.py:77`

```python
def fix_inconsistency(demonstrations, cur_metric, name, alpha, iter=0, K=20):
    if cur_metric["inconsistent_num"] == 0:
        return demonstrations, cur_metric

    best_metric = cur_metric
    best_assignment = assignment

    for k in range(K):
        # 1. Find inconsistent pairs
        pipeline = propose_consistencyfix(...)
        results = asyncio.run(pipeline.run())
        decisions = results["decisions"]

        # 2. Try all consistent label combinations
        for decision_id, decision in enumerate(decisions.values()):
            for score in [0, 1]:
                tmp_assignment = update_assign_based_on_decision(assignment, decision)
                tmp_metric = evaluate(tmp_assignment)

                # 3. Pick best combination
                if get_energy(tmp_metric, alpha) >= get_energy(best_metric, alpha):
                    best_metric = tmp_metric
                    best_assignment = tmp_assignment
                    break

        if best_metric["inconsistent_num"] == 0:
            break

    return demonstrations, best_metric
```

**Purpose:** Implements **Algorithm 2: ConsistencyFix** from the paper.

**Process:**
1. Find inconsistent pairs using logical rules
2. Enumerate all consistent label combinations
3. Select combination maximizing U(D)
4. Iterate until no inconsistencies remain (max K=20 iterations)

**Important:** This runs after EVERY label change, not just initialization.

---

#### 1.5 Main ICM Loop

**Location:** `ICM.py:427`

```python
def main(args):
    # Load data and initialize
    train, fewshot_ids = load_data(args)
    demonstrations, unlabeled_ids, whole_ids, seed_ids = initialize(train, fewshot_ids, args)

    for iter in range(args.K):  # K = 3000 iterations
        # Step 1: Weighted sampling (prioritize consistency groups)
        candidates_ids = whole_ids
        weights = [1 for _ in range(len(candidates_ids))]
        for i in candidates_ids:
            if i in cur_pool:
                same_group = [j for j in candidates_ids
                             if demonstrations[j]["consistency_id"] == demonstrations[i]["consistency_id"]]
                for j in same_group:
                    if j not in cur_pool:
                        weights[j] = 100  # 100× weight for unlabeled in same group

        example_id = random.choices(candidates_ids, k=1, weights=weights)[0]

        # Step 2: Predict new label
        new_label = asyncio.run(predict_assignment(model, demonstrations[example_id], cur_pool))

        # Step 3: If label changed, fix inconsistencies and evaluate
        if demonstrations[example_id]["label"] != new_label:
            tmp_demonstrations = deepcopy(demonstrations)
            tmp_demonstrations[example_id]["label"] = new_label

            tmp_demonstrations, _ = fix_inconsistency(tmp_demonstrations, ...)
            metric = evaluate(tmp_demonstrations)

            # Step 4: Simulated annealing acceptance
            T = get_temperature(flip_cnt, args.initial_T, args.final_T, args.decay)
            ΔE = get_energy(metric, args.alpha) - get_energy(cur_metric, args.alpha)
            accept_prob = exp(ΔE / T)

            if random.random() < accept_prob:
                demonstrations = tmp_demonstrations  # Accept
                cur_metric = metric
                flip_cnt += 1
```

**Algorithm Summary:**
```
Initialize with K=8 randomly labeled examples
For 3000 iterations:
    1. Sample example (weighted toward consistency groups)
    2. Predict label using mutual predictability
    3. Fix logical inconsistencies
    4. Calculate ΔU = U(new) - U(current)
    5. Accept with probability min(1, exp(ΔU / T))
Return final labeled dataset
```

---

### Data Loading and Templates

**Location:** `ICM.py:321`

#### TruthfulQA Template
```python
template = """Question: {question}
Claim: {answer}
I think this claim is """
```

#### GSM8K Template
```python
template = """Question: {question}
Claim: {answer}
I think this claim is """
```

#### Alpaca Template
```python
template = """Human: {question}
Response 1: {choice}
Response 2: {choice_2}
Claim: Response 1 is more helpful and harmless than Response 2.
I think this claim is """
```

**Consistency Groups:**
- **TruthfulQA/GSM8K:** `GROUP_SIZE = 4` (multiple answers per question)
- **Alpaca:** `GROUP_SIZE = 2` (pairwise comparisons)

---

## 2. ICM Helper Tools (`src/experiments/ICM_tools.py`)

### 2.1 Inconsistency Detection

**Location:** `ICM_tools.py:68`

```python
def pick_two_inconsistent_claims(data):
    consistency_groups = {}
    for claim in data.values():
        cid = claim["consistency_id"]
        if cid not in consistency_groups:
            consistency_groups[cid] = []
        consistency_groups[cid].append(claim)

    inconsistent_pairs = {}
    for group in consistency_groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                # Type 1: Contradiction (different answers both labeled True)
                if (group[i]['consistency_key'] != group[j]['consistency_key']) and \
                   (group[i]['label'] == group[j]['label'] == 1):
                    inconsistent_pairs[len(inconsistent_pairs)] = {
                        "claim_1": group[i],
                        "claim_2": group[j],
                        "type": "contradiction",
                    }

                # Type 2: Implication (same answer, different labels)
                elif (group[i]["consistency_key"] == group[j]["consistency_key"]) and \
                     (group[i]["label"] != group[j]["label"]):
                    inconsistent_pairs[len(inconsistent_pairs)] = {
                        "claim_1": group[i],
                        "claim_2": group[j],
                        "type": "implication",
                    }

    return inconsistent_pairs
```

**Inconsistency Types:**

1. **Contradiction:**
   - Example: "5+5=10 is True" AND "5+5=8 is True"
   - Rule: For same question, different answers can't both be True

2. **Implication:**
   - Example: "5+5=10 is True" AND "5+5=10 is False"
   - Rule: Same claim can't have different labels

**Returns:** Dictionary of inconsistent pairs with metadata

---

### 2.2 Label Update Based on Decisions

**Location:** `ICM_tools.py:49`

```python
def update_assign_based_on_decision(data, decision):
    if decision["type"] == "contradiction":
        if decision["score"] > 0:  # Claim 1 > Claim 2
            data[decision["claim_1"]["uid"]]["label"] = 1
            data[decision["claim_2"]["uid"]]["label"] = 0
        else:
            data[decision["claim_1"]["uid"]]["label"] = 0
            data[decision["claim_2"]["uid"]]["label"] = 1
    else:  # implication
        if decision["score"] > 0:  # Both True
            data[decision["claim_1"]["uid"]]["label"] = 1
            data[decision["claim_2"]["uid"]]["label"] = 1
        else:  # Both False
            data[decision["claim_1"]["uid"]]["label"] = 0
            data[decision["claim_2"]["uid"]]["label"] = 0
    return data
```

**Purpose:** Resolves inconsistencies by updating labels based on model's preference.

---

### 2.3 Metric Calculation

**Location:** `ICM_tools.py:27`

```python
def calculate_accuracy(train_data, inconsistent_pairs):
    return {
        "train_accuracy": np.mean([i["label"] == i["vanilla_label"]
                                   for i in train_data.values()]),
        "train_prob": np.mean([i["score"] if i["label"] == 1 else -i["score"]
                               for i in train_data.values()]),
        "train_size": len(train_data),
        "inconsistent_num": len(inconsistent_pairs),
        "train_predict_distribution": Counter([i["label"] for i in train_data.values()]),
        "train_label_distribution": Counter([i["vanilla_label"] for i in train_data.values()]),
    }
```

**Metrics:**
- `train_accuracy`: Agreement with ground truth (for evaluation only, not used in algorithm)
- `train_prob`: Average mutual predictability score
- `inconsistent_num`: Number of logical inconsistencies
- `train_*_distribution`: Label distribution stats

---

## 3. Prompt Creation & Logprob Extraction

### 3.1 Few-Shot Prompt Construction

**Location:** `src/model_querying/prompt_creation.py:31`

```python
def get_judge_prompt_fewshot(example, demonstrations=None, pipeline=True):
    if demonstrations is None:
        demonstrations = list(example["demonstration"].values())

    prompt = ""
    for i in demonstrations:
        prompt += i['prompt']
        prompt += "True" if i["label"] else "False"
        prompt += "\n\n"

    prompt += example['prompt']

    return Prompt(prompt) if pipeline else prompt
```

**Example Output:**
```
Question: What is 2+2?
Claim: 2+2=4
I think this claim is True

Question: What is 5+5?
Claim: 5+5=10
I think this claim is True

Question: What is 3+3?
Claim: 3+3=7
I think this claim is
```

**Purpose:** Creates in-context learning prompt with all labeled examples as demonstrations.

**Key Design:** Leave-one-out approach - when scoring example i, use all other N-1 examples as context.

---

### 3.2 Logprob Extraction

**Location:** `src/model_querying/solution_extraction.py:18`

```python
def get_yes_no_diff_logprobs(logprobs):
    eps = 1e-5
    prob_sums = {False: eps, True: eps}

    for k, v in logprobs.items():
        o = get_yes_no(k)  # Check if token contains "true" or "false"
        if o is None:
            continue
        prob_sums[o] += math.exp(v)  # Accumulate probabilities

    if prob_sums[False] == eps and prob_sums[True] == eps:
        return 0
    else:
        return math.log(prob_sums[True]) - math.log(prob_sums[False])
```

**Purpose:** Extracts log-odds ratio from model's top-K logprobs.

**Process:**
1. Get top-K tokens and their logprobs (K=20)
2. Classify each token as "True", "False", or neither
3. Sum probabilities: `P(True) = Σ exp(logprob)` for all "True" tokens
4. Return log-odds: `log P(True) - log P(False)`

**Why This Works:**
- Handles variations: "True", "true", "TRUE", " True", etc.
- Robust to tokenization differences across models
- Aggregates over multiple tokens expressing same concept

**Helper Function:**
```python
def get_yes_no(x):
    x = x.lower()
    y = "true" in x
    n = "false" in x
    if y == n:  # Both or neither
        return None
    return y
```

---

## 4. LLM API Layer (`core/llm_api/llm.py`)

### 4.1 ModelAPI Class

**Location:** `llm.py:28`

```python
@attrs.define()
class ModelAPI:
    anthropic_num_threads: int = 2
    openai_fraction_rate_limit: float = 0.99
    organization: str = "NYU_ORG"
    print_prompt_and_response: bool = False

    running_cost: float = 0
    model_timings: dict[str, list[float]] = {}
```

**Purpose:** Unified interface for multiple LLM APIs (Anthropic Claude, OpenAI, vLLM).

**Features:**
- **Rate limiting:** Respects API quotas
- **Caching:** Saves/loads responses to disk
- **Cost tracking:** Accumulates API costs
- **Timing:** Tracks latency metrics
- **Async:** Parallel requests via asyncio

---

### 4.2 Main Call Method

**Location:** `llm.py:111`

```python
async def __call__(
    self,
    model_ids: Union[str, list[str]],
    prompt: Union[list[dict[str, str]], str],
    max_tokens: int,
    logprobs: int = None,  # Top-K logprobs (K=20 for ICM)
    parse_fn: Callable = None,  # Post-processing function
    use_cache: bool = True,
    **kwargs,
) -> list[LLMResponse]:
```

**Workflow:**
```
1. Check cache (if use_cache=True)
2. Route to appropriate backend:
   - AnthropicChatModel for Claude
   - OpenAIBaseModel for Llama (via vLLM)
   - OpenAIChatModel for GPT
3. Execute API call(s) asynchronously
4. Apply parse_fn (e.g., extract_claim_logprobs)
5. Save to cache
6. Return parsed responses
```

**Model Routing:**
```python
def model_id_to_class(model_id: str):
    if model_id in BASE_MODELS:
        return self._openai_base  # For vLLM-deployed models
    elif model_id in GPT_CHAT_MODELS:
        return self._openai_chat
    elif model_id in ANTHROPIC_MODELS:
        return self._anthropic_chat
```

**Caching Strategy:**
```python
# Cache key = prompt + model + hyperparameters
save_path = f"./cache/{hash(prompt)}_{model}_{max_tokens}.json"

# Load from cache
if use_cache and os.path.exists(save_path):
    responses = json.load(save_path)

# Save to cache
with open(save_path, "w") as f:
    json.dump(responses, f, indent=2)
```

---

### 4.3 Response Format

```python
{
    "response": {
        "completion": "True",  # Generated text
        "logprobs": [
            {
                "True": -0.5,  # log P(token)
                "False": -2.3,
                " true": -1.2,
                ...
            }
        ],
        "cost": 0.00015,  # API cost in USD
        "model_id": "claude-3-haiku-20240307",
        "api_duration": 0.523,  # API call time
        "duration": 0.689,  # Total time
    },
    "metadata": {
        "uid": 42,  # Example ID
        ...
    },
    "score": 2.8  # After parse_fn (log-odds ratio)
}
```

---

## 5. Pipeline Orchestration (`src/pipeline/pipeline.py`)

### 5.1 Task Dependency Graph

**Location:** `pipeline.py:29`

```python
class Task:
    def __init__(self, name, func, use_cache, dependencies=[]):
        self.name = name
        self.func = func
        self.dependencies = dependencies  # Parent tasks
        self.dependents = []  # Child tasks
        self.result = None

    async def execute(self, results):
        if self.result is None:
            dep_results = [results[dep.name] for dep in self.dependencies]
            self.result = await self.func(*dep_results, use_cache=self.use_cache)
        return self.result
```

**Purpose:** Represents a node in the computational DAG.

---

### 5.2 Pipeline Class

**Location:** `pipeline.py:84`

```python
class Pipeline:
    def __init__(self, config):
        self.steps = []  # List of Task objects
        self.results = {}  # Task name → result mapping
        self.model_api = ModelAPI(...)
        self.file_sem = asyncio.BoundedSemaphore(config.num_open_files)

    def add_load_data_step(self, name, dataloader_fn, data_location, dependencies=[]):
        task = Task(name, dataloader_fn, use_cache, dependencies)
        self.steps.append(task)
        return task

    def add_query_step(self, name, model, prompt_fn, parse_fn, dependencies=[]):
        # Create async function that queries model
        async def call(data, use_cache, index):
            return await query_model(self.model_api, data, prompt_fn, parse_fn, ...)

        task = Task(name, call, use_cache, dependencies)
        self.steps.append(task)
        return task

    def add_transformation_step(self, name, transform_fn, dependencies=[]):
        task = Task(name, transform_fn, use_cache, dependencies)
        self.steps.append(task)
        return task

    def add_eval_step(self, name, eval_fn, dependencies=[]):
        task = Task(name, eval_fn, use_cache=False, dependencies)
        self.steps.append(task)
        return task
```

---

### 5.3 Example Pipeline

```python
pipeline = Pipeline(config)

# Step 1: Load data
data = pipeline.add_load_data_step("load", load_assignments, assignment)

# Step 2: Add demonstrations (leave-one-out)
demos = pipeline.add_transformation_step("add_demos", add_train_demonstrations,
                                          dependencies=[data])

# Step 3: Query model for logprobs
preds = pipeline.add_query_step("get_preds", model, get_judge_prompt_fewshot,
                                extract_claim_logprobs, dependencies=[demos])

# Step 4: Find inconsistencies
incons = pipeline.add_transformation_step("pick_incons", pick_two_inconsistent_claims,
                                           dependencies=[data])

# Step 5: Evaluate
metrics = pipeline.add_eval_step("evaluate", calculate_accuracy,
                                 dependencies=[preds, incons])

# Execute
results = asyncio.run(pipeline.run())
print(results["evaluate"])  # Access metrics
```

**Execution:** Topological sort → parallel execution of independent tasks.

---

## 6. Complete ICM Workflow

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                           │
│    - Load dataset (TruthfulQA/GSM8K/Alpaca)                │
│    - Sample batch (default: 256 examples)                  │
│    - Randomly label K=8 examples                           │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. MAIN LOOP (3000 iterations)                             │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 2a. Weighted Sampling                                │  │
│  │     - 100× weight for unlabeled in same group       │  │
│  └──────────────────────────────────────────────────────┘  │
│                             ↓                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 2b. Mutual Predictability                           │  │
│  │     - Few-shot prompt with N-1 examples             │  │
│  │     - Extract logprobs                              │  │
│  │     - Predict label                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                             ↓                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 2c. ConsistencyFix (if label changed)               │  │
│  │     - Find inconsistent pairs                       │  │
│  │     - Try all consistent combinations               │  │
│  │     - Pick best by U(D)                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                             ↓                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 2d. Simulated Annealing                             │  │
│  │     - Calculate ΔU = U(new) - U(current)           │  │
│  │     - Accept with prob = exp(ΔU / T)               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. OUTPUT                                                   │
│    - Labeled dataset (labels stored in demonstrations)     │
│    - Save to log file                                      │
└─────────────────────────────────────────────────────────────┘
```

---

### Detailed Scoring Flow

```
Input: Dataset D with partial labels

1. For each labeled example (xᵢ, yᵢ):
   ┌─────────────────────────────────────────────────┐
   │ Prompt = All other examples + (xᵢ, ?)          │
   │ Score  = log P(True | prompt) - log P(False)   │
   │ Store score in example metadata                 │
   └─────────────────────────────────────────────────┘

2. Calculate mutual predictability:
   ┌─────────────────────────────────────────────────┐
   │ P_θ(D) = Σᵢ log P(yᵢ | xᵢ, D\{(xᵢ,yᵢ)})       │
   │        = mean(score if label=True else -score) │
   └─────────────────────────────────────────────────┘

3. Find logical inconsistencies:
   ┌─────────────────────────────────────────────────┐
   │ For each consistency group (same question):     │
   │   - Contradiction: Different answers both True  │
   │   - Implication: Same answer different labels   │
   │ Count total inconsistencies: I(D)              │
   └─────────────────────────────────────────────────┘

4. Compute final score:
   ┌─────────────────────────────────────────────────┐
   │ U(D) = α · P_θ(D) - I(D)                       │
   │      = 50 · 0.82 - 3                           │
   │      = 38                                       │
   └─────────────────────────────────────────────────┘
```

---

## 7. Key Hyperparameters

### Command-Line Arguments

```bash
python ICM.py \
  --testbed truthfulQA \        # Dataset: {truthfulQA, gsm8k, alpaca}
  --model meta-llama/Llama-3.1-70B \
  --alpha 50 \                   # Balance parameter
  --batch_size 256 \             # Examples per batch
  --num_seed 8 \                 # Initial random labels
  --K 3000 \                     # Max iterations
  --consistency_fix_K 10 \       # Max ConsistencyFix iterations
  --initial_T 10 \               # Initial temperature
  --final_T 0.01 \               # Final temperature
  --decay 0.99 \                 # Decay rate
  --scheduler log \              # {exp, log}
  --seed 27565976                # Random seed
```

### Critical Hyperparameters

| Parameter | Default | Purpose | Impact |
|-----------|---------|---------|--------|
| `alpha` | 50 | Balance predictability vs consistency | Higher α → more weight on mutual predictability |
| `num_seed` | 8 | Initial random labels | Too low → poor initialization; too high → noisy start |
| `K` | 3000 | Main loop iterations | More iterations → better convergence but slower |
| `consistency_fix_K` | 10 | ConsistencyFix iterations | Higher → more thorough but expensive |
| `initial_T` | 10 | Starting temperature | Higher → more exploration early |
| `final_T` | 0.01 | Ending temperature | Lower → more exploitation late |
| `scheduler` | log | Temperature decay | Log slower than exp decay |

### Performance Metrics

From paper (Table 3):
- **Average forward passes per datapoint:** 2-3
- **Total cost:** ~2-4× cost of single forward pass through dataset
- **Caching benefit:** ~50% reduction in API calls

---

## 8. Implementation Details

### 8.1 Weighted Sampling Strategy

```python
# Base weight = 1 for all examples
weights = [1] * len(whole_ids)

# For each labeled example
for i in labeled_ids:
    # Find all examples in same consistency group
    same_group = [j for j in whole_ids
                  if demonstrations[j]["consistency_id"] == demonstrations[i]["consistency_id"]]

    # Boost weight 100× for unlabeled examples in same group
    for j in same_group:
        if j not in labeled_ids:
            weights[j] = 100
```

**Rationale:** Prioritize labeling examples that share logical relationships with already-labeled examples to maximize leverage of consistency constraints.

---

### 8.2 Caching Strategy

**Cache Directory Structure:**
```
prompt_history/
├── iterative-truth-assign-iter-0-truthfulQA/
│   ├── get_train_preds/
│   │   ├── 0.json
│   │   ├── 1.json
│   │   └── ...
│   └── evaluate.json
├── consistencyfix-iter-5-truthfulQA/
│   └── ...
└── ...
```

**Cache Key:** `{pipeline_name}/{step_name}/{example_id}.json`

**Why Critical:**
- ICM makes ~3 forward passes per datapoint
- 3000 iterations × 256 examples = potential 768K API calls
- Caching reduces to ~2× original dataset size

---

### 8.3 Consistency Rules

#### For TruthfulQA/GSM8K (Multiple Choice)

```python
consistency_id = question_hash  # Same question
consistency_key = answer_text   # Different answers

# Rule 1: Different answers to same question can't both be True
if (answer1 != answer2) and (label1 == True) and (label2 == True):
    INCONSISTENT (contradiction)

# Rule 2: Same answer must have same label
if (answer1 == answer2) and (label1 != label2):
    INCONSISTENT (implication)
```

#### For Alpaca (Pairwise Comparisons)

```python
consistency_id = question_hash
consistency_key = "A>B" or "B>A"

# Rule 1: Antisymmetry
if (claim = "A>B") and (label1 == True) and (label2 == True):
    where claim2 = "B>A"
    INCONSISTENT (contradiction)

# Rule 2: At least one must be True (comparative tasks)
if (label1 == False) and (label2 == False):
    where claim1 = "A>B" and claim2 = "B>A"
    INCONSISTENT (special case)
```

---

## 9. Production Deployment Notes

### From Paper (Section 4.4)

For production-scale deployment (400K examples):

1. **Two-stage process:**
   ```
   Stage 1: ICM labels 6K examples
   Stage 2: Train RM on 6K → label remaining 394K
   ```

2. **Iterative fine-tuning:**
   ```
   - Split data into batches of 256
   - For batch j:
       1. Use model M_{j-1} to label batch j
       2. Fine-tune M_{j-1} on batches 1..j → M_j
       3. Repeat
   ```

3. **Infrastructure requirements:**
   - vLLM for model serving
   - Prefix caching enabled
   - 40 parallel threads for Anthropic API
   - Async I/O for caching

---

## 10. Comparison to Paper Pseudocode

### Algorithm 1 from Paper

```
1: Randomly label K examples; D ← {(x₁,y₁),...,(xₖ,yₖ)}
2: D ← consistencyfix(D)
3: for n = 1, ..., N do
4:     T ← max(Tₘᵢₙ, T₀/(1+β·log(n)))
5:     Sample xᵢ ~ {x₁,...,xₙ}
6:     ŷᵢ ← arg max P_θ(y|xᵢ, D\{(xᵢ,yᵢ)})
7:     D̂ ← D ∪ {(xᵢ,ŷᵢ)}
8:     D̂ ← consistencyfix(D̂)
9:     Δ ← U(D̂) - U(D)
10:    if Δ > 0 then
11:        D ← D̂
12:    else
13:        if random() < exp(Δ/T) then
14:            D ← D̂
```

### Implementation Mapping

| Paper Line | Code Location | Notes |
|------------|---------------|-------|
| Line 1 | `ICM.py:406` | `random_init_labels = [1]*(K//2) + [0]*(K//2)` |
| Line 2 | `ICM.py:463` | Initial `fix_inconsistency()` call |
| Line 3 | `ICM.py:447` | `for _ in tqdm(range(args.K))` |
| Line 4 | `ICM.py:526` | `get_temperature()` with log schedule |
| Line 5 | `ICM.py:481` | `random.choices(candidates_ids, weights=weights)` |
| Line 6 | `ICM.py:484` | `predict_assignment()` |
| Line 7 | `ICM.py:494` | `tmp_demonstrations[example_id]["label"] = new_label` |
| Line 8 | `ICM.py:503` | `fix_inconsistency()` |
| Line 9 | `ICM.py:533` | `exp((get_energy(metric) - get_energy(cur_metric)) / T)` |
| Lines 10-14 | `ICM.py:535` | Simulated annealing acceptance |

**Key Differences:**
- Implementation uses **weighted sampling** (not in paper pseudocode)
- Implementation **batches** dataset into chunks (for memory efficiency)
- Implementation includes **extensive caching** and **async I/O**

---

## 11. Common Issues and Debugging

### Issue 1: All Labels Converge to Same Value

**Symptom:** All examples labeled True (or False)

**Cause:** `alpha` too high relative to consistency penalties

**Fix:** Reduce `alpha` from 50 to 20-30

---

### Issue 2: No Inconsistencies Fixed

**Symptom:** `inconsistent_num` stays constant

**Cause:** `consistency_fix_K` too low or consistency rules not matching data

**Debug:**
```python
# Add to fix_inconsistency()
print(f"Found {len(inconsistent_pairs)} inconsistencies")
print("Sample:", list(inconsistent_pairs.values())[:3])
```

---

### Issue 3: Very Slow Execution

**Symptom:** <1 iteration per minute

**Causes:**
1. No caching enabled
2. Prefix caching not enabled on vLLM
3. Too many API calls per iteration

**Fixes:**
1. Check `use_cache=True` in all pipeline steps
2. Enable vLLM prefix caching: `--enable-prefix-caching`
3. Reduce `consistency_fix_K` from 10 to 5

---

### Issue 4: API Rate Limits

**Symptom:** Frequent 429 errors

**Fix:**
```python
# Reduce parallelism
pipeline_config = PipelineConfig(
    ...,
    anthropic_num_threads=10,  # Down from 40
    openai_fraction_rate_limit=0.8,  # Down from 0.99
)
```

---

## 12. Extension Ideas

### 12.1 Multi-Task Learning

Label multiple datasets jointly to share consistency constraints:

```python
train = load_truthfulqa() + load_gsm8k()
# Shared consistency: "Mathematical correctness" concept
```

---

### 12.2 Active Learning

Instead of random initialization, use uncertainty sampling:

```python
# Initialize with most uncertain examples
scores = [zero_shot_score(x) for x in train]
uncertain_ids = argsort(abs(scores))[:8]  # Closest to decision boundary
```

---

### 12.3 Hierarchical Consistency

Add multi-level consistency rules:

```python
# Level 1: Within same question
# Level 2: Across questions with same topic
# Level 3: Global consistency (e.g., all math should be correct)
```

---

## Summary

This codebase implements a production-ready version of the ICM algorithm with:

✅ **Efficient caching** to minimize API calls
✅ **Async execution** for parallelism
✅ **Flexible API backends** (Anthropic, OpenAI, vLLM)
✅ **Robust logprob extraction** handling tokenization variations
✅ **Weighted sampling** to leverage consistency constraints
✅ **Extensive logging** for debugging
✅ **Scalability** to 400K+ examples

The implementation closely follows the paper's algorithms while adding practical optimizations for deployment at scale.

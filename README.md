# ECE-143-Group-23  
Welcome to the team repository!
To manage these large dataset files efficiently within Git, we will use Git Large File Storage (Git LFS).
**It is mandatory for all team members to set up Git LFS before cloning or working with the data files.** Failure to do so will result in errors when pushing/pulling large files.

## Setup Instructions: Git LFS
- **Download and install Git LFS:** https://git-lfs.com/
- Alternatively, using a package manager:
    - macOS (Homebrew): `brew install git-lfs`
    - Windows (Chocolatey): `choco install git-lfs`

- Make sure to run `git lfs install` from within your local clone of the repo

- After that, you can commit and push normally

# Repo Structure
```
ECE-143-Group-23/
├── analysis_results/ # Images and report generated from analyzing model
│
├── data/ # Raw CSVs 
│
├── notebooks/ # Jupyter notebook
│ └── Preference_Analysis.ipynb # Notebook showing visualizations of model evaluation
│ └── data_visualization.ipynb  # Notebook showing visualizations of dataset
│
├── src/ # All Python scripts
│ │
│ ├── analysis/ # LLM Model Preference Analysis Code
│ │ ├── README.md                  # Explains findings from analysis script
│ │ └── preference_analysis.py     # Script analyzing model preference in the dataset (from notebook)
│ │
│ ├── models/ # Model training and evaluation
│ │ ├── deberta_test.py            # Main training script
│ │ ├── deberta_test_v2.py         # Training script with similarity features implemented
│ │ ├── deberta_with_similarity.py # Module extending the DeBERTa model to incorporate similarity features
│ │ ├── model_evaluation.py        # Comprehensive evaluation script 
│ │ └── similarity_features.py     # Module to calculate similarity features
│ │
│ ├── preprocessing/ # Data preprocessing code 
│ │ ├── data_preprocessing.py       # Main preprocessing script (from notebook)
│ │ └── data_preprocessing_funcs.py # Helper Functions
│ │
│ └── visualization/ # Visualization code 
│   └── data_visualization.py      # Main data visualization script (from notebook)
│
├── tf-logs/trunc_2048_run/ # TensorBoard logs for generating images  
│
├── visual_plot/ # Generated Images
│
├── .gitattributes 
├── .gitignore 
├── environment.yml  
└── README.md 
```

# Third-Party Modules:
- Numerical / Data Libraries
  - numpy
  - pandas
  - scipy.stats
- scikit-learn
  - sklearn.model_selection
  - sklearn.metrics
- PyTorch
  - torch
  - torch.nn.functional
  - torch.utils.data
  - torch.utils.tensorboard
- Transformers (Hugging Face)
  - transformers
    - AutoTokenizer
    - AutoModelForSequenceClassification
    - Trainer
    - TrainingArguments
    - TrainerCallback
    - set_seed
- Visualization libraries
  - matplotlib.pyplot
  - seaborn

# How to Run the Code

## 1. Navigate to the root of the repository
  Open a terminal and move into the top-level project folder
  ```bash
  cd ECE-143-Group-23
  ```

## 2. Environment Setup

### Option A: Using Conda (Recommended)
```bash
# If the environment already exists, you can either:
# - Update it: conda env update -n ece143 -f environment.yml --prune
# - Or remove and recreate (⚠️ WARNING: This will delete the existing environment):
conda env remove -n ece143
conda env create -f environment.yml
conda activate ece143
```

### Option B: Using pip and venv
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate pandas numpy scikit-learn scipy matplotlib seaborn tensorboard tqdm
```

## 3. Verify Data Files
```bash
ls data/train.csv data/test.csv
```
Both files should exist. If not, make sure Git LFS is set up and the files are downloaded.

## 4. Run the Model Training Script
From the root of the repository (with the environment activated):
```bash
python src/models/deberta_test.py
```

**What the script does automatically:**
1. **Data Loading & Preprocessing** (automatic):
   - Loads `data/train.csv` and `data/test.csv`
   - Parses JSON strings in prompt/response columns
   - Converts one-hot encoded labels to single label column
   - Cleans missing data
   - Splits training data into train/validation sets

2. **Model Initialization**:
   - Downloads DeBERTa-v3-base model from Hugging Face
   - Initializes tokenizer and model
   - Sets up training configuration

3. **Training**:
   - Trains for 3 epochs (configurable)
   - Saves checkpoints during training
   - Logs metrics to TensorBoard

4. **Evaluation**:
   - Evaluates on validation set
   - Prints confusion matrix and accuracy metrics

5. **Prediction & Submission**:
   - Generates predictions on test set
   - Creates `submission_test.csv` file

**Note:** 
- Training will take several hours depending on your hardware (GPU recommended)
- The script handles all data preprocessing automatically - no separate preprocessing step needed
- Model checkpoints are saved to `./llm_preference_model_smart/`
- TensorBoard logs are saved to `./tf-logs/trunc_2048_run/`

## 5. (Optional) Evaluate Trained Model
After training, you can run comprehensive evaluation:
```bash
python src/models/model_evaluation.py --model_path ./llm_preference_model_smart/checkpoint-XXX
```

This will generate:
- Detailed metrics (precision, recall, F1-score per class)
- Confusion matrix visualization
- Per-class performance plots
- Error analysis CSV
- JSON evaluation report

## 6. (Optional) View Training Progress with TensorBoard
If you want to monitor training in real-time:
```bash
tensorboard --logdir ./tf-logs/trunc_2048_run
```
Then open your browser to `http://localhost:6006`

# Dataset Description

The `data/` directory contains the raw CSV files used for training, testing, and submission.

## `train.csv`
- **Purpose**: Contains the training data with user preferences for LLM responses.
- **Rows**: 57,477
- **Columns**:
    - `id`: Unique identifier for each conversation session.
    - `model_a`, `model_b`: Names of the two LLM models being compared.
    - `prompt`: A JSON string representing a list of prompts in a multi-turn conversation. Example: `["prompt1", "prompt2"]`.
    - `response_a`, `response_b`: JSON strings representing lists of responses from `model_a` and `model_b` respectively, corresponding to the prompts. Example: `["response_a1", "response_a2"]`.
    - `winner_model_a`, `winner_model_b`, `winner_tie`: One-hot encoded labels indicating the user's preference.
        - `winner_model_a = 1`: Model A is preferred.
        - `winner_model_b = 1`: Model B is preferred.
        - `winner_tie = 1`: User found both responses equally good (a tie).

## `test.csv`
- **Purpose**: Contains the test data for which predictions need to be generated.
- **Rows**: 3
- **Columns**: `id`, `prompt`, `response_a`, `response_b` (similar to `train.csv` but without winner labels).

## `sample_submission.csv`
- **Purpose**: Provides the expected format for the submission file to Kaggle.
- **Columns**: `id`, `winner_model_a`, `winner_model_b`, `winner_tie`. The values should be probabilities.

## Important Notes on Data Format:
- The `prompt`, `response_a`, and `response_b` columns are stored as **JSON strings** that represent lists of strings. For example, a cell might contain `'["Hello", "How are you?"]'`. The preprocessing step in the model script handles parsing these into actual Python lists and then joining them into a single string for the model input.
- The dataset represents **multi-turn conversations**. When `prompt`, `response_a`, and `response_b` contain multiple elements, they correspond to different turns in the same conversation. The model script concatenates these turns into a single sequence, preserving the conversational context.
- The `deberta_test.py` script directly loads and preprocesses these CSVs, handling the JSON parsing and label conversion internally.

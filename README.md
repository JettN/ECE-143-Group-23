# ECE-143-Group-23  
Welcome to the team repository!
To manage these large dataset files efficiently within Git, we will use Git Large File Storage (Git LFS).
**It is mandatory for all team members to set up Git LFS before cloning or working with the data files.** Failure to do so will result in errors when pushing/pulling large files.

Currently testing "microsoft/deberta-v3-base" model  
Change to "microsoft/deberta-v3-small" for smaller model  
Set PROTOTYPE_FRAC to change the proportion of training set used  
Working on testing and tuning  

# File Structure
```
ECE-143-Group-23/
├── saved_deberta_model_debug/                         # Debug snapshot of DeBERTa model (KEEP???)
│
├── tf-logs/trunc_2048_run                             # Tensorboard Records (KEEP???)
│   └── events.out.tfevents...
│
├── data/                                              # CSV files containing data sets
│
├── visual_plot/                                       # PNG images of data visualizations
│   └── data-visuzalization-for-ece143.ipynb           # Jupyter notebook showing all visualizations generated
│
├── Modules/                                           # Different Components for the code
│
├── .gitignore                                         # Git ignore rules
├── .gitattributes                                     # Indicates which files belong to Git LFS
├── _____.py                                           # File containing model to run
└── README.md                                          # Project documentation
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

## Environment Config
```
conda env create -f environment.yml  
conda activate ece143  
  ```
## Setup Instructions: Git LFS
- **Download and install Git LFS:** https://git-lfs.com/
- Alternatively, using a package manager:
    - macOS (Homebrew): `brew install git-lfs`
    - Windows (Chocolatey): `choco install git-lfs`

- Make sure to run `git lfs install` from within your local clone of the repo

- After that, you can commit and push normally

## Running the Code:
- **TODO**


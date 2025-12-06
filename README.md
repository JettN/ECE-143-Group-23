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

# File Structure
```
ECE-143-Group-23/ 
├── data/ # Raw CSVs 
│
├── notebooks/ # Jupyter notebook
│ └── data_visualization.ipynb # Notebook showing visualizations 
│
├── src/ # All Python scripts 
│ │
│ ├── models/ # Model  
│ │ └── deberta_test.py 
│ │
│ ├── preprocessing/ # Data preprocessing code 
│ │ ├── data_preprocessing.py # main preprocessing script 
│ │ └── data_preprocessing_funcs.py
│ │
│ └── visualization/ # Visualization code 
│   └── data_visualization.py 
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
  `cd ECE-143-Group-23`

## 2. Environment Config
  Setup and activate the environment
  ```
  conda env remove -n ece143
  conda env create -f environment.yml  
  conda activate ece143  
  ```

## 3. Run the Model Script:
  From the root of the repository, run:
  `python src/models/deberta_test.py`

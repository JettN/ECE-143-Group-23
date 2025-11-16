# ECE-143-Group-23  
Welcome to the team repository!
To manage these large dataset files efficiently within Git, we will use Git Large File Storage (Git LFS).
**It is mandatory for all team members to set up Git LFS before cloning or working with the data files.** Failure to do so will result in errors when pushing/pulling large files.

Currently testing "microsoft/deberta-v3-base" model  
Change to "microsoft/deberta-v3-small" for smaller model  
Set PROTOTYPE_FRAC to change the proportion of training set used  
Working on testing and tuning  

# Environment Config
```
conda env create -f environment.yml  
conda activate ece143  
  ```
# Setup Instructions: Git LFS
- **Download and install Git LFS:** https://git-lfs.com/
- Alternatively, using a package manager:
    - macOS (Homebrew): `brew install git-lfs`
    - Windows (Chocolatey): `choco install git-lfs`

- Make sure to run `git lfs install` from within your local clone of the repo
- After that, you can commit and push normally
# Prerequisites

## 1. Software Requirements

Ensure the following software and libraries are installed on your system:

Python: Version 3.10 or higher

pip: Python package manager

## 2. Files in the Project

The following files are included in the project:

main.py: The main Python script

requirements.txt: The file listing all required Python libraries

Any additional data files required by the program

# Steps to Reproduce the Output

## 1. Clone or Download the Project

Clone the repository or download the project files to your local machine:
```
git clone https://github.com/buithangart04/traveling_thief_problem_aco-.git
cd cs2
```

## 2. Set Up the Environment (skip this part if you already have pymoo and numpy libraries in your environment)

It is recommended to use a virtual environment to manage dependencies:

### Step 1: Create a Virtual Environment
```
python -m venv venv
```

### Step 2: Activate the Virtual Environment

On Windows:
```
venv\Scripts\activate
```
On macOS/Linux:
```
source venv/bin/activate
```
### Step 3: Install Dependencies

Install the required Python libraries using pip:
```
pip install -r requirements.txt
```
## 3. Choose instances to run
Go to the main function and change the instances you want to run by changing `instance_2_run` variable' value. You can also change other hyperparameter values.

## 4. Run the Program

Execute the main script to reproduce the outputs:
```
python main.py
```
# Output Location

The output files (including .hv, .f, .x) will be saved in the `results/instance` directory specified in the program (Please note that the output files have 'Group16_`instance`' formats, not the ones in the optimization folder).

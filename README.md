# Project Setup and Execution Guide

This guide provides step-by-step instructions on how to set up and run this project within an Anaconda environment using the Spyder IDE.

## Prerequisites

- Anaconda distribution installed on your machine.
- Basic familiarity with conda environments and Spyder IDE.

## 1. Creating a Virtual Environment

First, open your Anaconda prompt and execute the following command to create a new virtual environment:

```bash
conda create -n Deep_Learning python=3.10
```


Activate your newly created environment:

```bash
conda activate Deep_Learning
```

## 2. Installing Required Packages

With your environment activated, navigate to the directory containing your project's `requirements.txt` file. Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## 3. Verifying PyTorch Installation

To ensure PyTorch has been installed correctly in your environment, run the following command in your spyder console:

```bash
import torch; print(torch.__version__)
```

## 4. Running the Preprocessing Script

Before running the main project file, you need to execute the preprocessing script. In Spyder, open the `preprocessing.py` file while your virtual environment is activated. You can run the script by pressing F5 or clicking the run button.

## 5. Running the Main Project File

Finally, to execute the main project logic, open the `main.py` file in Spyder within the same virtual environment. Run the script by pressing F5 or using the run button, just like you did for the preprocessing script.

By following these steps, you should be able to set up your environment and run the project successfully within Spyder.
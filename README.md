# Tutorial for Traffic Time-series Prediction

This tutorial is designed to addressing the challenge of **multi-step forecasting** in **traffic time series data**. The key features of this project include:

1. Sliding windows for sampling traffic sequence data, which enhances the model's ability to understand and predict future trends effectively.
2. A comprehensive suite of deep learning models - RNN, GRU, LSTM, and Transformer - to facilitate robust modeling, training, and evaluation of forecasting performances.

If you find this project useful, please consider giving it a star :star: — it helps!

## Project Setup and Execution Guide

This guide provides step-by-step instructions on how to set up and run this project within an **Anaconda environment** using the **Spyder IDE**.

### **0. Prerequisites**

- Anaconda distribution installed on your machine.
- Basic familiarity with conda environments and Spyder IDE.

### **1. Create and Activate a Virtual Environment:**

Open your **Anaconda prompt** (check it in the Start Menu). Use the following command to navigate to the root directory of this project.

```bash
cd <path\to\your\file>
```

Execute the following command to create a new conda environment:

```bash
conda create -n Deep_Learning python=3.10
```

Activate your newly created environment:

```bash
conda activate Deep_Learning
```

Once the environment is activated, you will see the following prompt indicating that the environment is in use:

```bash
(Deep_Learning) <path\to\your\file> :
```

### 2. Installing Required Packages

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

Upon successful installation of the packages, you would typically see a series of messages indicating the successful installation of each package. For instance, the console might display something like this at the end of the installation process:

````bash
Successfully installed package1-version package2-version  ...
````

### 3. Installing and Verifying Pytorch

PyTorch is the key package for our project, renowned as one of the most popular Python libraries for deep learning. The installation commands can vary based on the operating system and computing platform you're using. To find the specific installation command that fits your setup, please visit [PyTorch&#39;s official installation guide](https://pytorch.org/get-started/locally/).

For the Anaconda environment provided by our school, there are two recommended installation commands:

For installations without GPU support, use:

````bash
pip3 install torch
````

This is the **recommended option** if you do not require GPU acceleration.

For installations with GPU support, if your system has a compatible NVIDIA GPU and you wish to leverage it for accelerated computing, use:

````bash
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
````

After installing PyTorch, you should verify the installation to ensure everything is set up correctly. Using the following command:

````bash
python -c "import torch; print(torch.__version__)"
````

You will be able to see the version of torch package like:

````bash
2.2.1+cu118
````

or

````bash
2.2.1+cpu
````

### 4. Start the Project in Spyder

There are two main approaches to working with Spyder for running your Python project. Here's how you can proceed with either:

**Launching Spyder from the Anaconda Prompt:**
Once the environment is active and the root is navigated to this project, you can launch Spyder directly from the prompt by typing:

````bash
spyder
````

**Launching Spyder from the Anaconda Navigator:**

* Open the Anaconda Navigator and select the `Home` tab.
* Launch Spyder from the Navigator. This approach may not set your desired working directory automatically.
* Once Spyder opens, you can set the root directory by navigating to `File > Open` and selecting your project's root directory.
* To ensure Spyder is using the correct interpreter from your environment, go to `Tools > Preferences > Python interpreter > <Path to anaconda> / Anaconda3/envs/Deep_Learning` and select the Python interpreter associated with your `Deep_Learning` environment. This might require navigating to your environment's directory and selecting the python executable.

### 5. Running the Files

Before running the main project file, you need to execute the preprocessing script. In Spyder, open the `preprocessing.py` file while your virtual environment is activated. You can run the script by pressing F5 or clicking the run button.

Finally, to execute the main project logic, open the `main.py` file in Spyder within the same virtual environment. Run the script by pressing F5 or using the run button.

---

Thank you for making it to the end! If you found this helpful, please don't forget to give us a star :star:!
------------------------------------------------------------------------------------------------------




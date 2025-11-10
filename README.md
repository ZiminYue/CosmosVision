# 🌌 Cosmos Vision

**Cosmos Vision** is a computer-vision-based interactive program designed to explore the relationship between bodily motion, macro-perspective visualization, and identity perception.  
It forms part of the *Cosmos Vision* research project, developed as part of MSc Creative Making: Advanced Final Project thesis at UAL.



💡 Notes

The project was tested with Python 3.10, CuPy 12.3, and CUDA 11.8 on Windows 11.

---

## 🛠️ Installation & Run (Windows, using Miniconda)

Follow the steps below to set up and run the project:



### 1. Install Miniconda

Download and install **[Miniconda](https://docs.conda.io/en/latest/miniconda.html)** for Windows.

During installation, make sure to select:  
> ✅ *Add Miniconda to my PATH environment variable*

 

### 2. Create and activate a new environment

```bash
conda create -n cosmosvision python=3.10
conda activate cosmosvision
```

(You can replace cosmosvision with any environment name you prefer.)



### 3. Install GPU-related dependencies

```
conda install -c conda-forge cupy cudatoolkit=11.8
```

💡 This step installs CuPy for CUDA 11.8. Make sure your GPU supports CUDA 11.8 and your drivers are up to date.



### 4. Install other dependencies

```
pip install -r requirements.txt
```



### 5. Run the program

```
python main.py
```

Then the program will launch and display the interactive cosmos simulation window!

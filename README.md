# 🌌 Cosmos Vision

**Cosmos Vision** is a computer-vision-based interactive program designed to explore the relationship between bodily motion, macro-perspective visualization, and identity perception.  
It forms part of the *Cosmos Vision* research project, developed as part of **MSc Creative Making: Advanced Final Project** thesis at UAL.



# 💡 Notes

The project was tested with Python 3.10, CuPy 12.3, and CUDA 11.8 on Windows 11.


---


# 📂 Project Structure

```
CosmosVision_main/
├── main.py                
├── core.py                 
├── background.py           
├── sph_module.py
├── lighting.py             
├── audio/ (optional)       # Folder for music files
│      └─ *.mp3, *.wav, *.ogg, *.flac
├── postcards/ (auto-generated)   # Saved postcards
├── requirements.txt        
└── README.md
```


---


## 🛠️ Installation & Run (Windows, using Miniconda)

Please follow the steps below to set up and run the project:



### 1. Download the Project

You can either **clone** the repository via GitHub Desktop:

```
git clone https://github.com/YourUsername/CosmosVision_main.git
```

Or download it as a ZIP file from GitHub and extract it manually.





### 2. Navigate to the Project Folder

Open `command prompt` and run the following code:

`cd /d "path-to-project-folder\CosmosVision_main"`

(Replace "path-to-project-folder" with the actual folder path on your computer.)





### 3. Install Miniconda

Download and install **[Miniconda](https://docs.conda.io/en/latest/miniconda.html)** for Windows.

During installation, make sure to select:  
> ✅ *Add Miniconda to my PATH environment variable*


 

### 4. Create and activate a new environment

```bash
conda create -n cosmosvision python=3.10
conda activate cosmosvision
```

(You can replace "cosmosvision" here with any environment name you prefer.)




### 5. Install GPU-related dependencies

```
conda install -c conda-forge cupy cudatoolkit=11.8
```

💡 This step installs CuPy for CUDA 11.8. Make sure your GPU supports CUDA 11.8 and your drivers are up to date.




### 6. Install other dependencies

```
pip install -r requirements.txt
```




### 7. Run the program

```
python main.py
```

Then the program will launch and display the interactive cosmos simulation window!

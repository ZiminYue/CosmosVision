# 🌌 Cosmos Vision

**Cosmos Vision** is a computer-vision-based interactive program designed to explore the relationship between bodily motion, macro-perspective visualization, and identity perception. It forms part of the *Cosmos Vision: Reimagining Identity from a Macro Perspective with Computer Vision* research project, developed for **MSc Creative Making: Advanced Final Project** at UAL.

---

## ❤ Acknowledgement

This project was developed based on Narcis Calin's **[Galaxy Engine](https://github.com/NarcisCalin/Galaxy-Engine)** and Angel Uriot's **[Galaxy simulation](https://github.com/angeluriot/Galaxy_simulation/)**, with their C++ physics and visualization frameworks converted and extended in Python under the assistance of LLMs (ChatGPT and Claude).

Example music files are royalty-free tracks sourced from Pixabay.


---

## 💡 Notes

The program was tested with Python 3.10, CuPy 12.3, and CUDA 11.8 on Windows 11.

The `main` branch contains the complete version of the project. For detailed development process and experimental works, see the `tech-works` branch.

---


## 📂 Project Structure

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
└── requirements.txt        

```


---


## 🛠️ Installation & Run (Windows, using Miniconda)

Please follow the steps below to set up and run the project:



### 1. Install Miniconda

Download and install **[Miniconda](https://www.anaconda.com/download/success)** for Windows.

During installation, make sure to select:  
> ✅ *Add installation to my PATH environment variable*




#
### 2. Download the project

The program files are contained within the `\CosmosVision_main` folder. You can either **clone** the repository or **download it as a ZIP file** from GitHub and extract it manually.




#
### 3. Navigate to the project folder

Open `Command Prompt` and run the following code:

`cd /d "path-to-project-folder\CosmosVision_main"`

Replace "path-to-project-folder" with the actual folder path on your computer.




#
### 4. Create and activate a new environment

Run the following commands, one after the other:

```
conda create -n cosmosvision python=3.10
conda activate cosmosvision
```

You can replace "cosmosvision" here with any environment name you prefer.




#
### 5. Install GPU-related dependencies

Run the following command:

```
conda install -c conda-forge cupy cudatoolkit=11.8
```

💡 This step installs CuPy for CUDA 11.8. Make sure your GPU supports CUDA 11.8 and your drivers are up to date.




#
### 6. Install other dependencies

Run the following command:

```
pip install -r requirements.txt
```



#
### 7. Run the program

Run the following command:

```
python main.py
```

Then the program will launch and display the interactive cosmos simulation window!


---


## 🎞️ Videos

(UAL access required)

**[Demo Video](https://ual.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=1e02f889-573d-435f-ac9d-b393016601d5)** 

**[Presentation Video](https://ual.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=4af72ee5-c2ba-4477-9e6c-b39600d395bb)** 



### June 26

- Tested **Galaxy Engine** from https://github.com/NarcisCalin/Galaxy-Engine (C++, only avaliable for Windows) and built the executable exe file
  
*Notes*:
1. For building it in **VS Code**, need to download and install **MinGW-w64** from WinLibs (https://www.winlibs.com/)
2. Install the plugins that VS Code recommanded (for C++)
3. Add a line to the environment variables (`Path-to-MinGW\mingw64\bin`)
4. In the file `yaml.cmake`: Edit `GIT_TAG 0.8.0` -> `GIT_TAG master`
5. Run the following code in terminal (one by one)
 ```bash
   mkdir build
   cd build
   cmake .. -G "MinGW Makefiles" -DBUILD_AUDIO=OFF -DSUPPORT_AUDIO=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
   cmake --build .
```
6. Ensure the `/Textures`,`/Shaders`,`/fonts`,`/Sounds`folders in `/build` (I was trapped by missing `/Sounds`)
7. Then the `GalaxyEngine.exe` in `/build` should be executable
   
- Started considering how may I combine it with Python code



### June 27

- Discussed with LLMs about possible development directions:

Plan 1: Use Python and Mediapipe for pose detection (like I did before), then pass the data to the C++-based system for visual output

Plan 2: Convert the C++ system to Python

Plan 3: Rebuild the visual part in Python, after figuring out how the C++ system calculates physics like gravity.

Plan 4: Maybe explore a different visual style altogether—something like the pixel-based planet generators (e.g., https://github.com/Deep-Fold/PixelPlanets).


*Notes*: Since my familiarity with the languages is like `JavaScript ≥ Python > C++`, guess Plan 2 might be tough. Kinda want to test out Plan 4 though.


### July 15-16

- Conducted Plan 3 with the help of ChatGPT, converted some calculation fomulas to Python.

### July 17-18
- Converted `physics.cpp` and `morton.cpp` fomulas to Python (`core.py`).
- Created `tech work` branch on GitHub for test files

### July 19
- Converted `quadtree.cpp` to Python (`quadtree.py`).
- Created `test_animation.py` for animated test output.

### July 20
- Finished converting the rest files in `/physics` folder.
- Created a simple interactive test demo.
  
### August 4-6
- Tried to generate animated galaxies with existing code functions (see `tech work` branch).

### August 7
- Succeeded in creating a spinning spiral galaxy, integrating code from https://towardsdatascience.com/create-3-d-galactic-art-with-matplotlib-a7534148a319/ (see `tech work` branch). Adjustment required, though.

### August 15
- Adjusted code for generating spinning spiral galaxy, and tested the collision of multiple galaxies (Physical calculation effect is not obvious, and the animation is SLOW).

### August 18
- Reached out to a technician to check the code implementation.

### August 23-24
- Replaced the Matplotlib-based rendering with VisPy under LLMs' assistance, allowing the render of a large number of particles much more smoothly. 
- Edited the code for a long time, finally get the galaxy animated again (with lagging).
- Highlighted potentially useful parameters for later development

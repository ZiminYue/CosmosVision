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
- Succeeded in creating a spinning spiral galaxy, integrating code from `https://towardsdatascience.com/create-3-d-galactic-art-with-matplotlib-a7534148a319/` (see `tech work` branch). Adjustment required, though.

### August 15
- Adjusted code for generating spinning spiral galaxy, and tested the collision of multiple galaxies (Physical calculation effect is not obvious, and the animation is SLOW).

### August 18
- Reached out to a technician to check the code implementation.

### August 23-24
- Replaced the Matplotlib-based rendering with VisPy under LLMs' assistance, allowing the render of a large number of particles much more smoothly. 
- Edited the code for a long time, finally get the galaxy animated again (with lagging).
- Highlighted potentially useful parameters for later development

### August 25
- MAJOR UPDATE: Switched the system to GPU using CuPy, the animation can run smoothly with all existing modules and more particles now!

### August 27
- Created a file for producing background stars and integrated it with the main galaxy simulation, producing rather ideal effects in the visual output

### August 29
- Started designing the input -> output mapping system

### August 30
- Finished the initial mapping system, started input test with MediaPipe

### August 31
- Connected the input and the output
- Adjusted the parameters for better effects

### September 1
- Created version for the collision of multiple galaxies
- Invited two of my roommates (both are UAL students) for testing and collected some feedback

### September 2
- Following one of the tester's suggestions, added auto-cameras for dynamic galaxy cinematography.

### September 18
- Following one of the tester's suggestions, added guiding text on the canvas screen.
- Made the canvas window pop up as full screen for better UX

### September 19
- Made some adjustments to the guiding text

### September 21
- Converted the new file from **Galaxy Engine**, adding extra lighting effect to the galaxies

### September 22
- Attempted to package the project into an .exe file for portability, but after hours of trying PyInstaller and Nuitka, it still didn’t work.
- But verified the feasibility of remote testing via a video call.

### October 6
- Found that adding some music could improve immersion during testing, downloaded a royalty-free "cosmos" music from Pixabay (https://pixabay.com/music/ambient-floating-in-space-full-soundtrack-272331/) and planned to make it play with the code (not via other media player software).
  
### October 7
- Fixed the bug preventing mouse interaction with the scene in "manual camera mode".
- Improved the guiding messages.

### October 9
- Improved guiding messages.
- Added an auto music playing feature.

### October 20
- Cleaned the code, removed invalid key controls.
- Improved guiding messages.
- Added multilingual options.
- Added screenshot export feature.
- Added debug info for camera.


## Python Version

Current Version: Python 3.12.7

### July 18

1. Converted `physics.cpp` into `core.py`.
- Test output-01: ![Description](TestOutput/0718.png)
  
2. Converted and integrated `morton.cpp` into `core.py`, implemented Z-order curve spatial hashing for cache-friendly particle neighbor queries.
- Test output-02: ![Description](TestOutput/0718-02.png)

### July 19

1. Converted `quadtree.cpp` into `quadtree.py` and integrated it into `core.py`, enabling faster gravitational force calculations by approximating distant particle clusters, thus improving simulation performance for large numbers of particles.

2. Added `test_animation.py` for 2-dimentional animation output (`test.py` is not avaliable in current version).
- Test output-03: ![Description](TestOutput/0719.gif)

### July 20

1. Converted `SPH.cpp` into `sph_module.py` and integrated it into `core.py`, to compute fluid particle interactions through density and pressure calculations, enhancing the fluid dynamics realism in the galaxy simulation.
- Test output-04: ![Description](TestOutput/0720.gif)

2. Converted `slingshot.cpp` into `slingshot.py` and integrated it into `core.py`, enabling interactively launch particles by clicking and dragging, enabling real-time testing of motion dynamics and parameter effects.
- Test output-05: ![Description](TestOutput/0720-02.gif)
(Note: As pygame is used, run `python test.py` in terminal when want to have a try! `test.py` can be replaced with `test_animation.py` `test_slingshot.py` or `test_slingshot_boundary.py`)

### August 6
Made adjustment in `core.py` and created `test_glaxyShape.py`, trying to create galaxy-like visual output with current code (but the output looks kind of weird right now).
- Test output-06: ![Description](TestOutput/0806.gif)

### August 7
1. Made adjustment in `core.py` and `test_glaxyShape.py`, still trying to create galaxy-like visual output with current code. Integrated code from https://towardsdatascience.com/create-3-d-galactic-art-with-matplotlib-a7534148a319/ to make the distribution of the particles more galaxy-like. 

But this version cannot generate animation correctly and ChatGPT failed to fix it (It asked me to repeatedly fix some pieces of the code and later changed it back) :(
- Test output-07: ![Description](TestOutput/0807.png)

2. Then I asked Claude to analyze and fix the bugs in `core.py` and `test_glaxyShape.py`. Finally we have a spinning spiral galaxy! (Yet still looks a bit weird as time goes by. However it's good to see something like this so far!)
- Test output-08: ![Description](TestOutput/0807-02.gif)

### August 15
1. Made adjustment in `core.py` and `test_glaxyShape.py`, fixed the spinning direction.
- Test output-09: ![Description](TestOutput/0815.gif)

2. Made adjustment in `core.py`, enabling SPH module. Then created `test_multiple_galaxies.py` to see the visual effects produced by the collision of multiple galaxies. However the animation is VERY SLOW (the gif here is 500% of original speed).
- Test output-10: ![Description](TestOutput/0815-02.gif)

### August 23
Made adjustment in `core.py` and created `vispy_demo.py`, replacing the Matplotlib-based rendering with VisPy (`pip install Vispy` needed). Added a cloud of particles in the center of the galaxy.

Good news:
Now it is possible to smoothly render 3D output with thousands of practicles.

Bad news:
It should be animated, but everything's frozen in this version.
- Test output-11: ![Description](TestOutput/0823.gif)

### August 24
1. Made adjustment in `core.py` and `vispy_demo.py`, improved the visual output, but still not moving at all xp
- Test output-12: ![Description](TestOutput/0824.gif)
  
2. Made adjustment in `core.py` and created `test_physics.py`, `test_vispy_singleThread.py`, trying to check why the animation is not working. (When running `test_vispy_singleThread.py`, the galaxy starts moving, but could only move a liitle per many seconds)

3. Updated `core.py`, `sph_module.py` and `vispy_demo.py`, highlighting potentially useful parameters. Finally able to generate animated (but lagging) galaxy again!
- Test output-13: ![Description](TestOutput/0824-02.gif)

### August 24
Updated `vispy_demo.py` with loading time detecting code, found that the lagging was caused by SPH module.
- Debug info: ![Description](Debugs/0825-SPHLagging.png)

### August 25
1. ❕ Major update: Switch the simulation to GPU using CuPy, updating `core.py` and `vispy_demo.py`
   
   Steps Taken:
   
   (1) Update core dependencies:
   
   ```
   conda install numpy=1.25 scipy=1.11
   ```
   
   (2) Install GPU support:
   
   ```
   conda install -c conda-forge cupy=13.6 cudatoolkit=11.8
   ```
   
   (3) Modify current code
   
   - Added GPU flags and buffers (in `core.py`):
   ```
   self.use_gpu = False
   self.gpu_n2_limit = 3000
   self._gpu_ready = False
   self._d_pos, self._d_vel, self._d_mass = None, None, None
   ```
   
   - Implemented `_gpu_build_from_particles()` to upload particle data to GPU, and `_gpu_push_to_particles()` to fetch results back to CPU for rendering.
    
   (4) GPU N² update

   - `_update_gpu_n2(self, dt)` computes pairwise gravity on GPU with central mass, damping, and velocity/boundary constraints.

   - Particle positions/velocities are updated in float32 for memory efficiency.
     
   - Update in main `update()`
     ```
     if self.use_gpu and (N <= self.gpu_n2_limit):
          self._update_gpu_n2(dt)
      else:
          self._update_cpu(dt)
      ```
   (5) Demo update
   
   - Added `galaxy.use_gpu = True` and `galaxy.use_sph = True`
     
-  Test output-14 (The animation is very smooth now!): ![Description](TestOutput/0825.gif)

2. Created another version of `test_multiple_galaxies.py` to see the effects (I've also tried to let two galaxies from in different planes collide, but after the hit, particles went strange ways)
-  Test output-15 (The animation is very smooth now!): ![Description](TestOutput/0825-02.gif)

### August 27
Created `background.py` and `main.py`, and updated `core.py` to integrate the background stars with the galaxy simulation. Adjusted parameters to control star number, color, brightness, and motion speed. Now the visual output part can produce a rather ideal cosmic animated scene.
-  Test output-16: ![Description](TestOutput/0827.gif)

### August 30
Finished mapping design and created `motion_input.py` to test MediaPipe functions and adjust parameters
-  Test input-01: ![Description](TestOutput/0830.gif)

### August 31
Connected input and output with `main.py` and `motion_input.py`, deleted "stability" parameter for better effects.
-  Test input-02 & output-17: ![Description](TestOutput/0831.gif)

### September 1
1. Created `main_multiGalaxies.py` to see the visual effects produced by the collision of multiple galaxies.
2. Asked two of my friends to test this version and collected some feedback
-  Test input-03 & output-18: ![Description](TestOutput/0901.gif)

### September 2
Created `main_multiGalaxies_camera.py`, adding auto-camera (Update based on tester feedback) with multiple orbit patterns (and keeping manual control mode) for dynamic galaxy cinematography.
-  Test output-19: ![Description](TestOutput/0902.gif)

### September 18
Updated `main_multiGalaxies_camera.py`, making the canvas window pop up as full screen and adding guiding text (Update based on tester feedback).
-  Test output-20: ![Description](TestOutput/0918.gif)

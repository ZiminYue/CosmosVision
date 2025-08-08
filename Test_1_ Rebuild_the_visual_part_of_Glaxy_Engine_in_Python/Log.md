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
Make adjustment in `core.py` and create `test_glaxyShape.py`, trying to create galaxy-like visual output with current code (but the output looks kind of weird right now).
- Test output-06: ![Description](TestOutput/0806.gif)

### August 7
1. Make adjustment in `core.py` and `test_glaxyShape.py`, still trying to create galaxy-like visual output with current code. Integrated code from https://towardsdatascience.com/create-3-d-galactic-art-with-matplotlib-a7534148a319/ to make the distribution of the particles more galaxy-like. 

But this version cannot generate animation correctly and ChatGPT failed to fix it (It asked me to repeatedly fix some pieces of the code and later changed it back) :(
- Test output-07: ![Description](TestOutput/0807.png)

2. Then I asked Claude to analyze and fix the bugs in `core.py` and `test_glaxyShape.py`. Finally we have a spinning spiral galaxy! (Yet still looks a bit weird as time goes by. However it's good to see something like this so far!)
- Test output-08: ![Description](TestOutput/0807-02.gif)

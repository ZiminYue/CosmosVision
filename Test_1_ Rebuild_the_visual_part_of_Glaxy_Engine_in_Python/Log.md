### July 18

1. Converted `physics.cpp` into `core.py`.
- Test output-01: ![Description](TestOutput/0718.png)
  
2. Converted and integrated `morton.cpp` into `core.py`, implemented Z-order curve spatial hashing for cache-friendly particle neighbor queries.
- Test output-02: ![Description](TestOutput/0718-02.png)

### July 19

1. Converted `quadtree.cpp` into `quadtree.py` and integrated it into `core.py`, enabling faster gravitational force calculations by approximating distant particle clusters, thus improving simulation performance for large numbers of particles.

2. Added `test_animation.py` for 2-dimentional animation output (`test.py` is not avaliable in current version).
- Test output-03: ![Description](TestOutput/0719.gif)

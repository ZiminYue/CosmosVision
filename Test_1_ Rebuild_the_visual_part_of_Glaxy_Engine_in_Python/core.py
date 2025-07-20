import numpy as np
from sph_module import SPHModule
from slingshot import particle_slingshot
import matplotlib.pyplot as plt

class Particle:
    """
    A simple particle object representing a star or mass point.
    Each particle has a position, mass, velocity, and net force.
    """
    def __init__(self, pos, mass, index=None):
        self.pos = np.array(pos, dtype=np.float64)   # 2D position
        self.mass = mass                             # Scalar mass
        self.index = index                           # Optional index ID
        self.force = np.zeros(2)                     # Net force acting on the particle
        self.vel = np.zeros(2)                       # Current velocity
         # SPH 
        self.density = 0.0
        self.pressure = 0.0
        self.pressure_force = np.zeros(2) 


class QuadtreeNode:
    """
    A node in the quadtree structure for spatial partitioning and Barnes-Hut force approximation.
    """
    max_leaf_particles = 10      # Maximum number of particles in a leaf node
    min_leaf_size = 1e-3         # Smallest allowable node size (prevents infinite subdivision)

    def __init__(self, center, half_size, start_index, end_index, particles):
        self.center = np.array(center, dtype=float)   # Center of the current quadrant
        self.half_size = half_size                    # Half the width/height of the node
        self.start_index = start_index                # Start index in the particle list
        self.end_index = end_index                    # End index in the particle list
        self.particles = particles                    # List of particle objects in this region

        self.mass = 0.0                               # Total mass of this node
        self.center_of_mass = np.zeros(2)             # Center of mass for all particles in this node
        self.sub_grids = []                           # Children (subdivided) nodes

        if (end_index - start_index) <= QuadtreeNode.max_leaf_particles or half_size * 2 <= QuadtreeNode.min_leaf_size:
            self.compute_leaf_mass()
        else:
            self.subdivide()
            self.compute_internal_mass()

    def compute_leaf_mass(self):
        """
        Calculates total mass and center of mass for a leaf node (no further subdivision).
        """
        mass_sum = 0.0
        weighted_pos_sum = np.zeros(2)
        for i in range(self.start_index, self.end_index):
            p = self.particles[i]
            mass_sum += p.mass
            weighted_pos_sum += p.mass * p.pos
        self.mass = mass_sum
        self.center_of_mass = weighted_pos_sum / mass_sum if mass_sum > 0 else self.center

    def compute_internal_mass(self):
        """
        Calculates the mass and center of mass based on child nodes.
        """
        self.mass = 0.0
        self.center_of_mass = np.zeros(2)
        for node in self.sub_grids:
            self.mass += node.mass
            self.center_of_mass += node.mass * node.center_of_mass
        if self.mass > 0:
            self.center_of_mass /= self.mass

    def subdivide(self):
        """
        Subdivides the current region into 4 quadrants and assigns particles to the appropriate child node.
        """
        hs = self.half_size / 2
        for dx in [-hs, hs]:
            for dy in [-hs, hs]:
                sub_center = self.center + np.array([dx, dy])
                sub_particles = [
                        self.particles[i] for i in range(self.start_index, self.end_index)
                        if np.all(np.abs(self.particles[i].pos - sub_center) < hs)
                    ]
                if sub_particles:
                    self.sub_grids.append(
                        QuadtreeNode(sub_center, hs, 0, len(sub_particles), sub_particles)
                    )

    def compute_force_on(self, particle, theta=0.5, G=1.0, eps=1e-1):
        """
        Computes the gravitational force from this node on a target particle using Barnes-Hut approximation.
        """
        dx = self.center_of_mass - particle.pos
        dist = np.linalg.norm(dx) + eps  # Avoid division by zero
        if not self.sub_grids or self.half_size / dist < theta:
            # Use this node's total mass if it's sufficiently far
            force = G * self.mass * dx / (dist**3)
            return force
        else:
            # Otherwise, recurse into subnodes
            force = np.zeros(2)
            for node in self.sub_grids:
                force += node.compute_force_on(particle, theta, G, eps)
            return force


class GalaxyEngine:
    """
    The main simulation engine. Manages particles, updates their positions using gravitational forces,
    and builds the quadtree structure each frame.
    """
    def __init__(self, count=1000, bounds=100):
        self.bounds = bounds
        self.particles = self.generate_particles(count)
        self.sph = SPHModule(h=5.0)

    def generate_particles(self, count):
        """
        Initializes particles with random positions and masses within the simulation bounds.
        """
        pos = np.random.uniform(-self.bounds, self.bounds, size=(count, 2))
        mass = np.random.uniform(0.5, 1.5, size=count)
        return [Particle(p, m, i) for i, (p, m) in enumerate(zip(pos, mass))]

    def update(self, dt=0.1, theta=0.5, black_hole_mass=500):
        """
        Advance the simulation by one time step.

        Args:
            dt (float): Time step size.
            theta (float): Barnes-Hut approximation threshold.
            black_hole_mass (float): Mass of central black hole for gravitational pull.
        """

        #Step 1: Compute SPH-related densities and pressures (if used)
        if self.sph is not None:
            self.sph.compute_density_and_pressure(self.particles)

        #Step 2: Build quadtree for Barnes-Hut force calculation
        root = QuadtreeNode(center=np.array([0, 0]), 
                            half_size=self.bounds, 
                            start_index=0, 
                            end_index=len(self.particles), 
                            particles=self.particles)

        #Step 3: Use Barnes-Hut force approximation
        G = 1.0  # gravitational constant
        for p in self.particles:
            gravity_force = root.compute_force_on(p, theta=theta, G=G)
            acc = gravity_force / p.mass
            p.vel += acc * dt

        #Step 4: Apply SPH pressure forces (if available)
        for p in self.particles:
            acc_pressure = p.pressure_force / p.mass
            p.vel += acc_pressure * dt

        #Step 5: Move particles based on updated velocity
        for p in self.particles:
            p.pos += p.vel * dt    

    @property
    def positions(self):
        """
        Returns current particle positions as a numpy array.
        """
        return np.array([p.pos for p in self.particles])
    

def plot_starfield(positions, title="Starfield"):
    plt.figure(figsize=(6, 6))
    x = [p[0] for p in positions]
    y = [p[1] for p in positions]
    plt.scatter(x, y, s=0.5, color="white")
    plt.title(title)
    plt.gca().set_facecolor("black")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()
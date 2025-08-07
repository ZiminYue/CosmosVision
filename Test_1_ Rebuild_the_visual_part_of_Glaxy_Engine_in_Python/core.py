import numpy as np
from sph_module import SPHModule
from slingshot import particle_slingshot
import matplotlib.pyplot as plt

class Particle:
    """
    A simple particle object representing a star or mass point.
    Each particle has a position, mass, velocity, and net force.
    """
    def __init__(self, pos, mass, vel=None, index=None):
        # Force to expand into 3D vector
        if len(pos) == 2:
            pos = [pos[0], pos[1], 0.0]
        self.pos = np.array(pos, dtype=np.float64)  # 3D position
        self.mass = mass                             # Scalar mass
        self.index = index                           # Optional index ID
        self.force = np.zeros(3)                     # Net force acting on the particle
        self.vel = np.zeros(3)                       # Current velocity
        # SPH
        self.density = 0.0
        self.pressure = 0.0
        self.pressure_force = np.zeros(3)


class QuadtreeNode:
    """
    A node in the quadtree structure for spatial partitioning and Barnes-Hut force approximation.
    """
    max_leaf_particles = 10      # Maximum number of particles in a leaf node
    min_leaf_size = 1e-3         # Smallest allowable node size (prevents infinite subdivision)

    def __init__(self, center, half_size, start_index, end_index, particles):
        self.center = np.array(center, dtype=float)[:2]   # Force to 2D   # Center of the current quadrant
        self.half_size = half_size                        # Half the width/height of the node
        self.start_index = start_index                    # Start index in the particle list
        self.end_index = end_index                        # End index in the particle list
        self.particles = particles                        # List of particle objects in this region

        # Initialize mass and center of mass position
        # self.mass = 0.0
        # weighted_pos_sum = np.zeros(2)  # ← Explicitly 2D

        # for i in range(self.start_index, self.end_index):
        #     p = self.particles[i]
        #     weighted_pos_sum += p.mass * p.pos
        #     self.mass += p.mass

        # if self.mass > 0:
        #     self.center_of_mass = weighted_pos_sum / self.mass

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
        weighted_pos_sum = np.zeros(3)
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
                sub_center = self.center.copy()
                sub_center[0] += dx
                sub_center[1] += dy

                sub_particles = [
                    self.particles[i] for i in range(self.start_index, self.end_index)
                    if np.all(np.abs(self.particles[i].pos[:2] - sub_center[:2]) < hs)
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
    def __init__(self, num_particles=300, bounds=100):
        self.num_particles = num_particles
        self.bounds = bounds

        # Add missing constants
        self.G = 1.0                # Gravitational constant (tweakable)
        self.softening = 0.1        # Softening term to prevent divide-by-zero ←★ added this line
        self.central_mass = 1000.0  # Mass of the galaxy center (higher = faster rotation)

        # Call initialization method
        self.particles = self.generate_spiral_galaxy(arms=3)

    def generate_spiral_galaxy(self, arms=3):
        self.particles = []
        for i in range(self.num_particles):
            # Distance
            radius = np.random.normal(50, 15)

            # Spiral angle + noise
            arm = i % arms
            theta = arm * (2 * np.pi / arms) + radius * 0.3 + np.random.normal(0, 0.2)

            # 3D position (small perturbation in z direction)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = np.random.normal(0, 2)  # Simulate disk thickness

            # Compute rotational velocity (simplified)
            distance = np.sqrt(x**2 + y**2 + z**2)
            speed = np.sqrt(self.G * self.central_mass / (distance + 1e-2))
            vx = -speed * np.sin(theta)
            vy = speed * np.cos(theta)
            vz = 0

            self.particles.append(Particle(
                mass=np.random.uniform(1, 10),
                pos=np.array([x, y, z], dtype=np.float32),
                vel=np.array([vx, vy, vz], dtype=np.float32),
            ))

        return self.particles

    def update(self, dt=1.0):
        G = self.G
        softening = self.softening

        # Create projected_particles (2D copy) and mapping
        projected_particles = []
        particle_map = {}

        for p in self.particles:
            proj = Particle(
                mass=p.mass,
                pos=p.pos[:2].copy(),  # Only XY
                vel=p.vel[:2].copy()
            )
            projected_particles.append(proj)
            particle_map[id(proj)] = p

        # Create quadtree (note: use 2D positions)
        positions_2d = np.array([p.pos[:2] for p in self.particles])  # Extract x,y
        center = np.mean(positions_2d, axis=0)                        # Compute centroid
        half_size = np.max(np.abs(positions_2d - center)) * 1.5       # Max expansion

        root_center = np.array([center[0], center[1], 0.0])  # Expand to 3D
        root = QuadtreeNode(root_center, half_size, 0, len(self.particles), self.particles)

        # Initialize forces to zero
        for p in self.particles:
            p.force[:] = 0.0

        # For each projected particle, compute Barnes-Hut gravity and add to original particle
        for proj in projected_particles:
            force_2d = root.compute_force_on(proj, theta=0.5, G=G)
            original = particle_map[id(proj)]
            original.force[:2] = force_2d  # XY components
            original.force[2] = 0          # Z component set to 0

        # Update velocity and position using simple Euler integration
        for p in self.particles:
            acceleration = p.force / p.mass
            p.vel += acceleration * dt
            p.pos += p.vel * dt

    def get_positions(self):
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

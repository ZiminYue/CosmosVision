import numpy as np
import matplotlib.pyplot as plt

class Particle:
    """
    A simple particle object representing a star or mass point.
    Each particle has a position, mass, velocity, and net force.
    """
    def __init__(self, pos, mass, vel=None, index=None):
        # Force expansion to 3D vector
        if len(pos) == 2:
            pos = [pos[0], pos[1], 0.0]
        self.pos = np.array(pos, dtype=np.float64)  # 3D position
        self.mass = mass                             # Scalar mass
        self.index = index                           # Optional index ID
        self.force = np.zeros(3)                     # Net force acting on the particle
        
        # Velocity initialization
        if vel is not None:
            if len(vel) == 2:
                vel = [vel[0], vel[1], 0.0]
            self.vel = np.array(vel, dtype=np.float64)
        else:
            self.vel = np.zeros(3)                   # Current velocity
            
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
        self.center = np.array(center, dtype=float)[:2]   # Force to 2D # Center of the current quadrant
        self.half_size = half_size                    # Half the width/height of the node
        self.start_index = start_index                # Start index in the particle list
        self.end_index = end_index                    # End index in the particle list
        self.particles = particles                    # List of particle objects in this region

        self.center_of_mass = np.zeros(2)             # Center of mass for all particles in this node
        self.sub_grids = []                           # Children (subdivided) nodes
        self.mass = 0.0                               # Initialize quality

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
            weighted_pos_sum += p.mass * p.pos[:2]  # Take only XY components
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
        
        # First, collect particles for all quadrants
        quadrant_particles = [[] for _ in range(4)]
        
        for i in range(self.start_index, self.end_index):
            p = self.particles[i]
            pos_2d = p.pos[:2]
            
            # Determine which quadrant the particle belongs to
            dx = pos_2d[0] - self.center[0]
            dy = pos_2d[1] - self.center[1]
            
            if dx >= 0 and dy >= 0:  # Top-right
                quadrant_particles[0].append(p)
            elif dx < 0 and dy >= 0:  # Top-left
                quadrant_particles[1].append(p)
            elif dx < 0 and dy < 0:  # Bottom-left
                quadrant_particles[2].append(p)
            else:  # Bottom-right
                quadrant_particles[3].append(p)
        
        # Create child nodes for each quadrant that has particles
        quadrant_offsets = [(hs, hs), (-hs, hs), (-hs, -hs), (hs, -hs)]
        
        for i, particles_in_quad in enumerate(quadrant_particles):
            if particles_in_quad:
                dx, dy = quadrant_offsets[i]
                sub_center = self.center + np.array([dx, dy])
                
                self.sub_grids.append(
                    QuadtreeNode(sub_center, hs, 0, len(particles_in_quad), particles_in_quad)
                )

    def compute_force_on(self, particle, theta=0.5, G=1.0, eps=1e-1):
        """
        Computes the gravitational force from this node on a target particle using Barnes-Hut approximation.
        """
        dx = self.center_of_mass - particle.pos[:2]  # Compare only 2D positions
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
    def __init__(self, num_particles=600, bounds=100):
        self.num_particles = num_particles
        self.bounds = bounds

        # Adjust physical parameters for better rotation effects
        self.G = 2.0                # Increase gravitational constant to enhance cohesion
        self.softening = 0.5        # Moderate softening to avoid excessive perturbation
        self.central_mass = 1000.0  # Increase central mass to enhance rotation

        # Fix: Pass parameters when calling
        self.particles = self.generate_spiral_galaxy(arms=5)

    def generate_spiral_galaxy(self, arms=5):
        """Generate spiral galaxy structure"""
        particles = []
        for i in range(self.num_particles):
            # Distance - use more reasonable distribution, concentrated at medium radius
            radius = max(8, np.random.gamma(2, 15))  # More concentrated distribution
            
            # Spiral angle - enhance spiral structure
            arm = i % arms
            base_angle = arm * (2 * np.pi / arms)
            spiral_angle = radius * 0.08  # Enhance spiral winding degree
            noise = np.random.normal(0, 0.15)  # Moderate random perturbation
            theta = base_angle + spiral_angle + noise

            # 3D position (small perturbation in Z direction)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = np.random.normal(0, 1.5)  # Slightly reduce disk thickness

            # Key: Set reasonable rotation velocity
            distance_2d = np.sqrt(x**2 + y**2)
            
            # Use flat rotation curve (closer to real galaxies)
            if distance_2d < 10:
                # Inner region: rigid body rotation
                speed = distance_2d * 0.15  
            else:
                # Outer region: flat rotation curve
                speed = np.sqrt(self.G * self.central_mass / (distance_2d + 5)) * 1.2
            
            # Add radial velocity component (simulate spiral waves)
            radial_speed = 0.02 * np.sin(2 * spiral_angle)
            
            # Calculate velocity components
            vx = speed * np.sin(theta) + radial_speed * np.cos(theta)
            vy = -speed * np.cos(theta) + radial_speed * np.sin(theta)
            vz = np.random.normal(0, 0.05)  # ery small z-direction velocity

            particles.append(Particle(
                mass=np.random.uniform(0.8, 1.5),  # Reduce mass variation
                pos=[x, y, z],
                vel=[vx, vy, vz],
            ))
            
        return particles

    def update(self, dt=0.1):
        """Update particle positions and velocities"""
        G = self.G
        softening = self.softening

        # Build quadtree for force calculation
        if not self.particles:
            return
            
        positions_2d = np.array([p.pos[:2] for p in self.particles])
        center = np.mean(positions_2d, axis=0)
        half_size = max(np.max(np.abs(positions_2d - center)) * 1.2, 1.0)

        root = QuadtreeNode(center, half_size, 0, len(self.particles), self.particles)

        # Initialize force to zero
        for p in self.particles:
            p.force[:] = 0.0

        # Calculate gravitational forces
        for p in self.particles:
            # Barnes-Hut gravity (inter-particle) - reduce influence to avoid disruption
            force_2d = root.compute_force_on(p, theta=0.8, G=G, eps=softening)  
            p.force[:2] += force_2d * 0.1  # Further reduce inter-particle influence
            
            # Central black hole gravity - main source of centripetal force
            r_center = np.linalg.norm(p.pos[:2]) + softening
            central_force_mag = G * self.central_mass / (r_center**2)
            central_force_dir = -p.pos[:2] / r_center
            p.force[:2] += central_force_mag * central_force_dir
            
            # Z-direction restoring force (maintain disk structure) - fix the "bump" issue
            z_distance = abs(p.pos[2])
            if z_distance > 0.1:  # Only apply strong force when far from plane
                z_restoring = -5.0 * p.pos[2] - 1.0 * p.vel[2]  # Strong planar constraint
            else:
                z_restoring = -0.5 * p.pos[2] - 0.2 * p.vel[2]  # Gentle constraint near plane
            p.force[2] += z_restoring
            
            # Add gentle damping to stabilize system (simulate interstellar medium drag)
            p.force[:2] += -0.002 * p.vel[:2]  # Reduced damping

        # Update velocity and position
        for p in self.particles:
            acceleration = p.force / p.mass
            p.vel += acceleration * dt
            
            # Limit maximum velocity to prevent system instability
            speed_2d = np.linalg.norm(p.vel[:2])
            if speed_2d > 15:  # 2D velocity limit
                p.vel[:2] *= 15 / speed_2d
            
            # Separate limit for z-velocity to prevent "bump"
            if abs(p.vel[2]) > 2:  # Much stricter z-velocity limit
                p.vel[2] = np.sign(p.vel[2]) * 2
                
            p.pos += p.vel * dt
            
            # Boundary handling
            if np.linalg.norm(p.pos[:2]) > self.bounds * 1.5:
                p.pos[:2] *= 0.95

    def get_positions(self):
        """Get all particle positions"""
        return np.array([p.pos for p in self.particles])

def plot_starfield(positions, title="Starfield"):
    """Draw starfield"""
    plt.figure(figsize=(6, 6))
    x = [p[0] for p in positions]
    y = [p[1] for p in positions]
    plt.scatter(x, y, s=0.5, color="white")
    plt.title(title)
    plt.gca().set_facecolor("black")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sph_module import SPHModule

class Particle:
    def __init__(self, pos, mass, vel=None, index=None):
        if len(pos) == 2:
            pos = [pos[0], pos[1], 0.0]
        self.pos = np.array(pos, dtype=np.float64)
        self.mass = mass
        self.index = index
        self.force = np.zeros(3)
        if vel is not None:
            if len(vel) == 2:
                vel = [vel[0], vel[1], 0.0]
            self.vel = np.array(vel, dtype=np.float64)
        else:
            self.vel = np.zeros(3)
        # SPH
        self.density = 0.0
        self.pressure = 0.0
        self.pressure_force = np.zeros(3)

class QuadtreeNode:
    max_leaf_particles = 10
    min_leaf_size = 1e-3

    def __init__(self, center, half_size, start_index, end_index, particles):
        self.center = np.array(center, dtype=float)[:2]
        self.half_size = half_size
        self.start_index = start_index
        self.end_index = end_index
        self.particles = particles
        self.center_of_mass = np.zeros(2)
        self.sub_grids = []
        self.mass = 0.0

        if (end_index - start_index) <= QuadtreeNode.max_leaf_particles or half_size * 2 <= QuadtreeNode.min_leaf_size:
            self.compute_leaf_mass()
        else:
            self.subdivide()
            self.compute_internal_mass()

    def compute_leaf_mass(self):
        mass_sum = 0.0
        weighted_pos_sum = np.zeros(2)
        for i in range(self.start_index, self.end_index):
            p = self.particles[i]
            mass_sum += p.mass
            weighted_pos_sum += p.mass * p.pos[:2]
        self.mass = mass_sum
        self.center_of_mass = weighted_pos_sum / mass_sum if mass_sum > 0 else self.center

    def compute_internal_mass(self):
        self.mass = 0.0
        self.center_of_mass = np.zeros(2)
        for node in self.sub_grids:
            self.mass += node.mass
            self.center_of_mass += node.mass * node.center_of_mass
        if self.mass > 0:
            self.center_of_mass /= self.mass

    def subdivide(self):
        hs = self.half_size / 2
        quadrant_particles = [[] for _ in range(4)]
        for i in range(self.start_index, self.end_index):
            p = self.particles[i]
            dx, dy = p.pos[0] - self.center[0], p.pos[1] - self.center[1]
            if dx >= 0 and dy >= 0: quadrant_particles[0].append(p)
            elif dx < 0 and dy >= 0: quadrant_particles[1].append(p)
            elif dx < 0 and dy < 0: quadrant_particles[2].append(p)
            else: quadrant_particles[3].append(p)
        quadrant_offsets = [(hs, hs), (-hs, hs), (-hs, -hs), (hs, -hs)]
        for i, pts in enumerate(quadrant_particles):
            if pts:
                dx, dy = quadrant_offsets[i]
                sub_center = self.center + np.array([dx, dy])
                self.sub_grids.append(QuadtreeNode(sub_center, hs, 0, len(pts), pts))

    def compute_force_on(self, particle, theta=0.5, G=1.0, eps=1e-1):
        dx = self.center_of_mass - particle.pos[:2]
        dist = np.linalg.norm(dx) + eps
        if not self.sub_grids or self.half_size / dist < theta:
            return G * self.mass * dx / (dist**3)
        else:
            force = np.zeros(2)
            for node in self.sub_grids:
                force += node.compute_force_on(particle, theta, G, eps)
            return force

class GalaxyEngine:
    def __init__(self, num_particles=100, bounds=100):
        self.num_particles = num_particles
        self.bounds = bounds
        self.G = 2.0
        self.softening = 0.5
        self.central_mass = 1000.0
        self.particles = self.generate_spiral_galaxy(arms=5)
        self.sph = SPHModule(h=15.0, k=50.0, rest_density=0.8)
        self.use_sph = True

    def generate_spiral_galaxy(self, arms=5):
        particles = []
        for i in range(self.num_particles):
            radius = max(1, np.random.gamma(2, 15))
            arm = i % arms
            base_angle = arm * (2 * np.pi / arms)
            spiral_angle = radius * 0.08
            noise = np.random.normal(0, 0.15)
            theta = base_angle + spiral_angle + noise
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = np.random.normal(0, 1.5)
            distance_2d = np.sqrt(x**2 + y**2)
            if distance_2d < 10:
                speed = distance_2d * 0.15
            else:
                speed = np.sqrt(self.G * self.central_mass / (distance_2d + 5)) * 1.2
            radial_speed = 0.02 * np.sin(2 * spiral_angle)
            vx = speed * np.sin(theta) + radial_speed * np.cos(theta)
            vy = -speed * np.cos(theta) + radial_speed * np.sin(theta)
            vz = np.random.normal(0, 0.05)
            particles.append(Particle(mass=np.random.uniform(0.8, 1.5), pos=[x, y, z], vel=[vx, vy, vz]))

        # Light up the center
        num_core = 800
        for _ in range(num_core):
            r = np.random.exponential(scale=13) #Bigger number, wider range
            theta = np.random.uniform(0, 2*np.pi)
            z = np.random.normal(0, 0.5)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            speed = np.sqrt(self.G * self.central_mass / (r + 1))
            vx = -speed * np.sin(theta) * 0.5
            vy = speed * np.cos(theta) * 0.5
            vz = np.random.normal(0, 0.05)
            particles.append(Particle(mass=1.5, pos=[x, y, z]))
        # Give all particles an initial rotation speed
        omega = 10  
        for p in particles:
            x, y = p.pos[0], p.pos[1]
            p.vel[0] += -omega * y
            p.vel[1] += omega * x
        return particles

    def update(self, dt=0.1):
        G = self.G
        softening = self.softening
        if not self.particles:
            return

        positions_2d = np.array([p.pos[:2] for p in self.particles])
        center = np.mean(positions_2d, axis=0)
        half_size = max(np.max(np.abs(positions_2d - center)) * 1.2, 1.0)
        root = QuadtreeNode(center, half_size, 0, len(self.particles), self.particles)

        for p in self.particles:
            p.force[:] = 0.0
            # Barnes-Hut inter-particle
            force_2d = root.compute_force_on(p, theta=0.8, G=G, eps=softening)
            p.force[:2] += force_2d * 0.8
            # Central mass
            r_center = np.linalg.norm(p.pos[:2]) + softening
            central_force_mag = G * self.central_mass / ((r_center**2) + 1.0)
            central_force_dir = -p.pos[:2] / r_center
            p.force[:2] += central_force_mag * central_force_dir
            # Z restoring
            z_restoring = -0.01 * p.pos[2] - 0.005 * p.vel[2]
            p.force[2] += z_restoring
            # gentle damping
            p.force[:2] += -0.002 * p.vel[:2]

        # Update velocity & position
        for p in self.particles:
            acc = p.force / p.mass
            p.vel += acc * dt
            # limit 2D velocity
            speed2d = np.linalg.norm(p.vel[:2])
            if speed2d > 15: p.vel[:2] *= 15 / speed2d
            if abs(p.vel[2]) > 2: p.vel[2] = np.sign(p.vel[2]) * 2
            p.pos += p.vel * dt
            if np.linalg.norm(p.pos[:2]) > self.bounds * 1.5:
                p.pos[:2] *= 0.95

        # SPH update
        if self.use_sph:
            self.sph.compute_density_and_pressure(self.particles)
            self.sph.compute_pressure_forces(self.particles)
            for p in self.particles:
                p.force[:2] += p.pressure_force[:2]

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

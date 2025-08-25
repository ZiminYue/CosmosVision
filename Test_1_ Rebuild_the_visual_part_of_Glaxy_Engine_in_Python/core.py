import numpy as np
import cupy as cp
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

    def compute_force_on(self, particle, theta=0.8, G=1.0, eps=1e-1):
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
    def __init__(self, num_particles=100, bounds=100, velocity_factor=5.0):
        self.num_particles = num_particles
        self.bounds = bounds
        self.G = 1.0
        self.softening = 0.5
        self.central_mass = 2000.0 # USEFUL PARAMETER: around 1000-2000 bigger -> shape stabler
        self.velocity_factor = velocity_factor  # USEFUL PARAMETER: orbital stability adjustment
        self.particles = self.generate_spiral_galaxy(arms=7) # USEFUL PARAMETER: number of arms
        self.sph = SPHModule(h=15.0, k=50.0, rest_density=0.8)
        self.use_sph = True
        self.sph_interval = 10
        self._step = 0
        

        # === New: GPU switch & cache ===
        self.use_gpu = False                 # Can be switched externally
        self.gpu_n2_limit = 3000            # N² GPU safety limit
        self._gpu_ready = False
        self._d_pos = None                  # (N,3) float32
        self._d_vel = None                  # (N,3) float32
        self._d_mass = None                 # (N,)  float32

    # ---- GPU synchronization tools ----
    def _gpu_build_from_particles(self):
        # Use float32 to reduce memory usage by half
        pos = np.array([p.pos for p in self.particles], dtype=np.float32)
        vel = np.array([p.vel for p in self.particles], dtype=np.float32)
        mass = np.array([p.mass for p in self.particles], dtype=np.float32)
        self._d_pos = cp.asarray(pos)
        self._d_vel = cp.asarray(vel)
        self._d_mass = cp.asarray(mass)
        self._gpu_ready = True

    def _gpu_push_to_particles(self):
        # Write GPU position/velocity back to particle objects (for rendering & other CPU logic)
        pos = cp.asnumpy(self._d_pos)
        vel = cp.asnumpy(self._d_vel)
        for p, r, v in zip(self.particles, pos, vel):
            p.pos[:] = r
            p.vel[:] = v

    # ---- Generate initial galaxy (keep your original) ----
    def generate_spiral_galaxy(self, arms=5):
        particles = []
        for i in range(self.num_particles):
            radius = max(0.1, np.random.gamma(2, 5)) # USEFUL PARAMETER: bigger -> more hollow in center, bigger -> more concentrated, bigger -> greater scale
            arm = i % arms
            base_angle = arm * (2 * np.pi / arms)
            spiral_angle = radius * 0.1
            noise = np.random.normal(0, 0.15) # USEFUL PARAMETER: bigger -> "fatter" the arms
            theta = base_angle + spiral_angle + noise
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = np.random.normal(0, 1.5)
            distance_2d = np.sqrt(x**2 + y**2)
            if distance_2d < 10:
                speed = distance_2d * 0.15
            else:
                speed = np.sqrt(self.G * self.central_mass / (distance_2d + 0.5)) * self.velocity_factor 
            radial_speed = 0.2 * np.sin(2 * spiral_angle)
            vx = speed * np.sin(theta) + radial_speed * np.cos(theta)
            vy = -speed * np.cos(theta) + radial_speed * np.sin(theta)
            vz = np.random.normal(0, 0.05)
            particles.append(Particle(mass=np.random.uniform(0.8, 1.5), pos=[x, y, z], vel=[vx, vy, vz]))

        # Light up the center
        num_core = 500
        for _ in range(num_core):
            r = np.random.exponential(scale=13) # USEFUL PARAMETER: Bigger -> wider range of center stars
            theta = np.random.uniform(0, 2*np.pi)
            z = np.random.normal(0, 0.5)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            speed = np.sqrt(self.G * self.central_mass / (r + 1))
            vx = -speed * np.sin(theta) * 0.5
            vy = speed * np.cos(theta) * 0.5
            vz = np.random.normal(0, 0.05)
            particles.append(Particle(mass=1.5, pos=[x, y, z]))
        # Add uniform initial angular velocity
        omega = 15 # USEFUL PARAMETER: bigger -> faster spinning
        for p in particles:
            x, y = p.pos[0], p.pos[1]
            p.vel[0] += omega * y
            p.vel[1] += -omega * x
        return particles

    # ---- GPU update: N² vectorized (without SPH) ----
    def _update_gpu_n2(self, dt):
        N = len(self.particles)
        if (not self._gpu_ready) or (self._d_pos is None) or (self._d_pos.shape[0] != N):
            self._gpu_build_from_particles()

        G = self.G
        eps = self.softening
        M0 = self.central_mass

        # Only do interactions in XY plane, use soft constraint for Z
        P = self._d_pos[:, :2]            # (N,2) float32
        V = self._d_vel[:, :2]            # (N,2)
        M = self._d_mass                  # (N,)

        # Mutual displacement & distance (N,N,2) / (N,N)
        # Note: This is O(N^2) memory usage, recommend N<=~3000
        dR = P[None, :, :] - P[:, None, :]                    # (N,N,2)
        dist = cp.linalg.norm(dR, axis=2) + eps               # (N,N)

        # Mask self-interaction
        inv_dist3 = 1.0 / (dist * dist * dist)
        inv_dist3 = inv_dist3 * (1.0 - cp.eye(N, dtype=inv_dist3.dtype))

        # Mass product (N,N)
        mass_prod = M[:, None] * M[None, :]

        # Particle-particle gravitational force sum (N,2)
        F_pp = G * (mass_prod * inv_dist3)[:, :, None] * dR
        F_pp = F_pp.sum(axis=1)

        # Central black hole force (N,2)
        r0 = cp.linalg.norm(P, axis=1) + eps                  # (N,)
        dir0 = -P / r0[:, None]
        F_c = (G * M0 / (r0*r0 + 100.0))[:, None] * dir0

        # Damping
        F_damp_xy = -0.001 * V

        # Total force & acceleration
        F_xy = 0.8 * F_pp + F_c + F_damp_xy                   # (N,2)
        a_xy = F_xy / M[:, None]

        # Z direction: only soft constraint and damping
        z = self._d_pos[:, 2]
        vz = self._d_vel[:, 2]
        az = (-0.001 * z - 0.005 * vz) / M

        # Integration
        self._d_vel[:, 0:2] = V + a_xy * dt
        self._d_vel[:, 2]   = vz + az * dt
        self._d_pos += self._d_vel * dt

        # Speed limit (XY)
        speed2d = cp.linalg.norm(self._d_vel[:, :2], axis=1)
        mask = speed2d > 15.0
        self._d_vel[mask, 0] *= 15.0 / speed2d[mask]
        self._d_vel[mask, 1] *= 15.0 / speed2d[mask]
        # Speed limit (Z)
        self._d_vel[:, 2] = cp.clip(self._d_vel[:, 2], -10.0, 10.0)

        # Boundaries
        r_edge = cp.linalg.norm(self._d_pos[:, :2], axis=1)
        mask = r_edge > (self.bounds * 1.5)
        self._d_pos[mask, 0:2] *= 0.95

        # Write back to CPU particle objects (for rendering)
        self._gpu_push_to_particles()

    # ---- Original CPU update (keep your previous Barnes-Hut + optional SPH) ----
    def _update_cpu(self, dt):
        G = self.G
        softening = self.softening
        if not self.particles:
            return

        # Clear forces
        for p in self.particles:
            p.force[:] = 0.0

        # Barnes-Hut + central black hole + damping/thickness
        positions_2d = np.array([p.pos[:2] for p in self.particles])
        center = np.mean(positions_2d, axis=0)
        half_size = max(np.max(np.abs(positions_2d - center)) * 1.2, 1.0)
        root = QuadtreeNode(center, half_size, 0, len(self.particles), self.particles)

        for p in self.particles:
            force_2d = root.compute_force_on(p, theta=0.8, G=G, eps=softening)
            p.force[:2] += force_2d * 0.8

            r_center = np.linalg.norm(p.pos[:2]) + softening
            central_force_mag = G * self.central_mass / ((r_center**2) + 100)
            central_force_dir = -p.pos[:2] / r_center
            p.force[:2] += central_force_mag * central_force_dir

            p.force[2] += (-0.001 * p.pos[2] - 0.005 * p.vel[2])
            p.force[:2] += -0.00001 * p.vel[:2]

        # SPH (optional, expensive)
        if self.use_sph and (self._step % self.sph_interval == 0):
            backup_pos = [p.pos.copy() for p in self.particles]
            try:
                for p in self.particles:
                    p.pos = p.pos[:2].copy()
                self.sph.compute_density_and_pressure(self.particles)
                self.sph.compute_pressure_forces(self.particles)
                for p in self.particles:
                    pf_xy = np.array(p.pressure_force[:2], dtype=np.float64) * 0.02
                    p.force[:2] += pf_xy
            finally:
                for p, pos in zip(self.particles, backup_pos):
                    p.pos = pos

        # Integration
        for p in self.particles:
            acc = p.force / p.mass
            p.vel += acc * dt
            p.pos += p.vel * dt

            speed2d = np.linalg.norm(p.vel[:2])
            if speed2d > 15:
                p.vel[:2] *= 15 / speed2d
            if abs(p.vel[2]) > 10:
                p.vel[2] = np.sign(p.vel[2]) * 10

            if np.linalg.norm(p.pos[:2]) > self.bounds * 1.5:
                p.pos[:2] *= 0.95

    # ---- Unified external interface ----
    def update(self, dt=0.1):
        N = len(self.particles)
        if self.use_gpu and (N <= self.gpu_n2_limit):
            self._update_gpu_n2(dt)
        else:
            self._update_cpu(dt)
        self._step += 1

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
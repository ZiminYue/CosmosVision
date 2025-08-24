import numpy as np

class SPHModule:
    def __init__(self, h=10.0, k=100.0, rest_density=1.0):
        self.h = h
        self.k = k
        self.rest_density = rest_density

    def kernel_poly6(self, r, h):
        if r >= h:
            return 0
        return (315 / (64 * np.pi * h**9)) * (h**2 - r**2)**3

    def kernel_spiky_grad(self, r_vec, h):
        r = np.linalg.norm(r_vec)
        if r == 0 or r >= h:
            return np.zeros_like(r_vec)
        return (-45 / (np.pi * h**6)) * (h - r)**2 * (r_vec / r)

    def compute_density_and_pressure(self, particles):
        for pi in particles:
            density = 0.0
            for pj in particles:
                # Only use XY plane to calculate distance
                r_vec_xy = pi.pos[:2] - pj.pos[:2]
                r = np.linalg.norm(r_vec_xy)
                if r < self.h:
                    density += pj.mass * self.kernel_poly6(r, self.h)
            pi.density = density
            pi.pressure = self.k * (pi.density - self.rest_density)

    def compute_pressure_forces(self, particles):
        for pi in particles:
            pressure_force = np.zeros(3)
            for pj in particles:
                if pi is pj:
                    continue
                # XY plane vector
                r_vec_xy = pi.pos[:2] - pj.pos[:2]
                r = np.linalg.norm(r_vec_xy)
                if r < self.h and pj.density > 0:
                    pressure_term = (pi.pressure + pj.pressure) / (2 * pj.density)
                    # Return 2D force
                    f_xy = -pj.mass * pressure_term * self.kernel_spiky_grad(r_vec_xy, self.h)
                    # Map to first two dimensions of 3D
                    pressure_force[:2] += f_xy
            pi.pressure_force = pressure_force
import numpy as np

class Particle:
    def __init__(self, pos, mass, index=None):
        """
        Represents a point-mass particle in 2D space.

        Parameters:
        - pos: (x, y) coordinates as iterable
        - mass: scalar mass
        - index: optional identifier
        """
        self.pos = np.array(pos, dtype=np.float64)
        self.mass = mass
        self.index = index
        self.force = np.zeros(2)  # Net force accumulator (reset every step)


class QuadtreeNode:
    max_leaf_particles = 10   # Max particles in a node before it must subdivide
    min_leaf_size = 1e-3      # Minimum spatial size of a node before subdivision stops

    def __init__(self, center, half_size, start_index, end_index, particles, parent=None):
        """
        Represents a node in the quadtree.

        Parameters:
        - center: center (x, y) of the node
        - half_size: half-length of the square region represented by this node
        - start_index, end_index: range of particle indices managed by this node
        - particles: shared list of Particle instances
        - parent: optional parent node (for debugging or traversal)
        """
        self.center = np.array(center, dtype=float)
        self.half_size = half_size
        self.start_index = start_index
        self.end_index = end_index
        self.particles = particles
        self.parent = parent

        self.mass = 0.0
        self.center_of_mass = np.zeros(2)
        self.sub_grids = []

        # If small enough, treat as a leaf node and compute directly
        if (end_index - start_index) <= QuadtreeNode.max_leaf_particles or half_size * 2 <= QuadtreeNode.min_leaf_size:
            self.compute_leaf_mass()
        else:
            self.subdivide()
            self.compute_internal_mass()

    def compute_leaf_mass(self):
        """
        Compute the total mass and center of mass for a leaf node.
        """
        mass_sum = 0.0
        weighted_pos_sum = np.zeros(2)
        for i in range(self.start_index, self.end_index):
            p = self.particles[i]
            mass_sum += p.mass
            weighted_pos_sum += p.mass * p.pos
        self.mass = mass_sum
        self.center_of_mass = weighted_pos_sum / mass_sum if mass_sum > 0 else self.center

    def subdivide(self):
        """
        Subdivide this node into four quadrants and partition particles into each.
        """
        quarter = self.half_size * 0.5
        offsets = [
            [-quarter, -quarter],  # Bottom-left
            [ quarter, -quarter],  # Bottom-right
            [-quarter,  quarter],  # Top-left
            [ quarter,  quarter],  # Top-right
        ]

        boundaries = [self.start_index]

        def predicate_quad(index, quad_idx):
            """Check if particle at index belongs to quadrant `quad_idx`."""
            p = self.particles[index]
            cx, cy = self.center
            ox, oy = offsets[quad_idx]
            return (cx + ox - quarter <= p.pos[0] < cx + ox + quarter) and \
                   (cy + oy - quarter <= p.pos[1] < cy + oy + quarter)

        def partition(start, end, quad_idx):
            """
            Reorder particles so those in quadrant `quad_idx` are moved to the front.
            Returns the index separating matched from unmatched.
            """
            i = start
            for j in range(start, end):
                if predicate_quad(j, quad_idx):
                    if i != j:
                        self.particles[i], self.particles[j] = self.particles[j], self.particles[i]
                    i += 1
            return i

        # Divide particles into quadrants
        for q in range(4):
            boundaries.append(partition(boundaries[-1], self.end_index, q))

        # Create child nodes for non-empty partitions
        for q in range(4):
            start_q = boundaries[q]
            end_q = boundaries[q + 1]
            if end_q > start_q:
                new_center = self.center + offsets[q]
                child = QuadtreeNode(new_center, quarter, start_q, end_q, self.particles, parent=self)
                self.sub_grids.append(child)

    def compute_internal_mass(self):
        """
        For internal nodes, compute aggregate mass and center of mass
        from all children.
        """
        total_mass = 0.0
        weighted_sum = np.zeros(2)
        for child in self.sub_grids:
            total_mass += child.mass
            weighted_sum += child.mass * child.center_of_mass
        self.mass = total_mass
        self.center_of_mass = weighted_sum / total_mass if total_mass > 0 else self.center

    def compute_force_on(self, particle, theta):
        """
        Calculate gravitational force on a particle using this node
        with Barnes-Hut approximation.

        Parameters:
        - particle: the Particle on which force is being computed
        - theta: threshold value controlling approximation vs recursion

        Returns:
        - 2D numpy array representing the force vector
        """
        EPS = 1e-3  # Softening factor to avoid singularities
        dx = self.center_of_mass[0] - particle.pos[0]
        dy = self.center_of_mass[1] - particle.pos[1]
        dist = np.sqrt(dx*dx + dy*dy) + EPS

        # If only one particle and it's the same one, skip
        if self.mass == 0 or (
            self.start_index == self.end_index - 1 and self.particles[self.start_index] is particle
        ):
            return np.zeros(2)

        s = self.half_size * 2  # Full width of the node
        if len(self.sub_grids) == 0 or (s / dist) < theta:
            # Treat this node as a single distant mass
            force_magnitude = self.mass / (dist**2 + EPS**2)
            force = np.array([dx, dy]) * force_magnitude / dist
            return force
        else:
            # Recurse into children
            force = np.zeros(2)
            for child in self.sub_grids:
                force += child.compute_force_on(particle, theta)
            return force

    @staticmethod
    def build_bounding_box(particles):
        """
        Build the root quadtree node that contains all particles.

        Parameters:
        - particles: list of Particle instances

        Returns:
        - A root QuadtreeNode covering all particle positions
        """
        min_pos = np.min([p.pos for p in particles], axis=0)
        max_pos = np.max([p.pos for p in particles], axis=0)
        center = (min_pos + max_pos) / 2
        half_size = np.max(max_pos - min_pos) / 2
        return QuadtreeNode(center, half_size, 0, len(particles), particles)

    def debug_print(self, indent=0):
        """
        Recursively print structure of the quadtree (for debugging purposes).

        Parameters:
        - indent: indentation level (increases with depth)
        """
        print(" " * indent + f"Node center={self.center}, half_size={self.half_size:.3f}, mass={self.mass:.3f}, center_of_mass={self.center_of_mass}")
        for child in self.sub_grids:
            child.debug_print(indent + 2)

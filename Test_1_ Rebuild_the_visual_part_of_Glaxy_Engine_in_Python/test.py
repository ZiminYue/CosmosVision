import pygame
import numpy as np
from core import GalaxyEngine

# Initialize pygame
pygame.init()
width, height = 800, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("GalaxyEngine SPH Density Visualization")

font = pygame.font.SysFont(None, 24)

# Convert screen coords to simulation coords [-bounds, bounds]
def screen_to_sim(pos, bounds):
    x = (pos[0] / width) * 2 * bounds - bounds
    y = (pos[1] / height) * 2 * bounds - bounds
    return np.array([x, y])

# Convert simulation coords to screen coords
def sim_to_screen(pos, bounds):
    x = int((pos[0] + bounds) / (2 * bounds) * width)
    y = int((pos[1] + bounds) / (2 * bounds) * height)
    return (x, y)

# Config
bounds = 100
init_particle_count = 200
initial_h = 10.0
power = 5.0

# Create GalaxyEngine
def create_engine():
    engine = GalaxyEngine(count=init_particle_count, bounds=bounds)
    engine.sph.h = initial_h
    return engine

engine = create_engine()
clock = pygame.time.Clock()
dragging = False
slingshot_start = None
slingshot_end = None

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                dragging = True
                slingshot_start = screen_to_sim(event.pos, bounds)
                slingshot_end = slingshot_start

        elif event.type == pygame.MOUSEMOTION and dragging:
            slingshot_end = screen_to_sim(event.pos, bounds)

        elif event.type == pygame.MOUSEBUTTONUP and dragging:
            dragging = False
            if slingshot_start is not None and slingshot_end is not None:
                direction = slingshot_start - slingshot_end
                length = np.linalg.norm(direction)
                if length > 0:
                    norm = direction / length
                    impulse = norm * length * power
                    for p in engine.particles:
                        p.vel += impulse / p.mass

    # --- Key controls ---
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        power += 0.1
    if keys[pygame.K_DOWN]:
        power = max(0.1, power - 0.1)
    if keys[pygame.K_w]:
        engine.sph.h += 0.2
    if keys[pygame.K_s]:
        engine.sph.h = max(0.2, engine.sph.h - 0.2)
    if keys[pygame.K_r]:
        engine = create_engine()

    # Update
    engine.update(dt=0.1)

    # Draw
    screen.fill((0, 0, 0))
    max_density = max(p.density for p in engine.particles) or 1.0

    for p in engine.particles:
        x, y = sim_to_screen(p.pos, bounds)
        norm_density = p.density / max_density
        r = int(255 * norm_density)
        b = int(255 * (1 - norm_density))
        color = (r, 0, b)
        if 0 <= x < width and 0 <= y < height:
            pygame.draw.circle(screen, color, (x, y), 2)

    # Draw slingshot line
    if dragging and slingshot_start is not None and slingshot_end is not None:
        start_screen = sim_to_screen(slingshot_start, bounds)
        end_screen = sim_to_screen(slingshot_end, bounds)
        pygame.draw.line(screen, (255, 0, 0), end_screen, start_screen, 2)
        pygame.draw.circle(screen, (0, 0, 255), start_screen, 5)

    # Show UI texts
    text1 = font.render(f'Power (launch speed): {power:.1f}   (UP/DOWN)', True, (255, 255, 255))
    text2 = font.render(f'SPH h (density smoothing range): {engine.sph.h:.2f}   (W/S)', True, (255, 255, 255))
    text3 = font.render(f'Press R to reset particles', True, (200, 200, 200))
    screen.blit(text1, (10, 10))
    screen.blit(text2, (10, 30))
    screen.blit(text3, (10, 50))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

import pygame
import numpy as np
from core import GalaxyEngine, Particle  # Assuming GalaxyEngine is defined in core.py

# Initialize pygame
pygame.init()
width, height = 800, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Slingshot Demo")

font = pygame.font.SysFont(None, 24)

# Normalize screen coordinates to simulation coordinates
def screen_to_sim(pos, bounds):
    # Map screen coordinates to simulation coordinate system [-bounds, bounds]
    x = (pos[0] / width) * 2 * bounds - bounds
    y = (pos[1] / height) * 2 * bounds - bounds
    return np.array([x, y])

def sim_to_screen(pos, bounds):
    # Map simulation coordinates back to screen coordinates
    x = int((pos[0] + bounds) / (2 * bounds) * width)
    y = int((pos[1] + bounds) / (2 * bounds) * height)
    return (x, y)

# Initialize engine and parameters
bounds = 100
engine = GalaxyEngine(count=200, bounds=bounds)

clock = pygame.time.Clock()
running = True

dragging = False
slingshot_start = None
slingshot_end = None
power = 5.0  # Force multiplier, adjustable by keyboard

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button starts dragging
                dragging = True
                slingshot_start = screen_to_sim(event.pos, bounds)
                slingshot_end = slingshot_start

        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                slingshot_end = screen_to_sim(event.pos, bounds)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and dragging:
                dragging = False
                if slingshot_start is not None and slingshot_end is not None:
                    direction = slingshot_start - slingshot_end
                    length = np.linalg.norm(direction)
                    if length > 0:
                        norm = direction / length
                        impulse = norm * length * power
                        # Apply velocity impulse to all particles (simple addition)
                        for p in engine.particles:
                            p.vel += impulse / p.mass

    # Keyboard controls to adjust force power
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        power += 0.1
    if keys[pygame.K_DOWN]:
        power = max(0.1, power - 0.1)

    # Update particle physics
    engine.update(dt=0.1)

    # Fill background with black
    screen.fill((0, 0, 0))

    # Draw all particles
    for p in engine.particles:
        x, y = sim_to_screen(p.pos, bounds)
        pygame.draw.circle(screen, (255, 255, 255), (x, y), 2)

    # Draw slingshot line and starting circle while dragging
    if dragging and slingshot_start is not None and slingshot_end is not None:
        start_screen = sim_to_screen(slingshot_start, bounds)
        end_screen = sim_to_screen(slingshot_end, bounds)
        pygame.draw.line(screen, (255, 0, 0), end_screen, start_screen, 2)
        pygame.draw.circle(screen, (0, 0, 255), start_screen, 5)

    # Display power text
    power_text = font.render(f'Power: {power:.1f} (Use UP/DOWN keys)', True, (255, 255, 255))
    screen.blit(power_text, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

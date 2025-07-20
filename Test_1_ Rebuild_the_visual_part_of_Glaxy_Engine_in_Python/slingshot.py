import numpy as np
import pygame

class Slingshot:
    def __init__(self, norm=np.array([0.0, 0.0]), length=0.0):
        # Normalized direction vector
        self.norm = norm
        # Distance from initial to current mouse position
        self.length = length


def get_mouse_world_pos(camera):
    # Convert current mouse screen position to world coordinates
    mouse_x, mouse_y = pygame.mouse.get_pos()
    return np.array(camera.screen_to_world(mouse_x, mouse_y))


def particle_slingshot(myVar, camera):
    """
    Simulates slingshot drag behavior: hold and drag left-click to generate
    a normalized direction vector and a drag length.
    """

    if pygame.mouse.get_pressed()[2]:
        # Right click cancels dragging
        myVar.is_dragging = False

    mouse_world_pos = get_mouse_world_pos(camera)

    keys = pygame.key.get_pressed()

    # Start dragging when left click is pressed without Ctrl/Alt and one of the tools is active,
    # or a relevant shortcut key is pressed
    if (
        pygame.mouse.get_pressed()[0] and
        not (keys[pygame.K_LCTRL] or keys[pygame.K_LALT]) and
        (
            myVar.tool_spawn_heavy_particle or
            myVar.tool_spawn_big_galaxy or
            myVar.tool_spawn_small_galaxy or
            myVar.tool_spawn_star
        )
    ) or (
        myVar.shortcut_pressed("1") or
        myVar.shortcut_pressed("2") or
        myVar.shortcut_pressed("3") or
        myVar.shortcut_pressed("j")
    ):
        myVar.is_dragging = True
        myVar.slingshot_pos = mouse_world_pos

    if myVar.is_dragging:
        slingshot_dist = myVar.slingshot_pos - mouse_world_pos
        length = np.linalg.norm(slingshot_dist)

        if length != 0:
            norm = slingshot_dist / length

            # Visualization can be added here using matplotlib or pygame
            # draw_circle(myVar.slingshot_pos, radius=5, color=blue)
            # draw_line(mouse_world_pos, myVar.slingshot_pos, color=red)

            return Slingshot(norm, length)

    return Slingshot()

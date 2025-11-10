import cupy as cp
import numpy as np

def emission(position, intensity=1.0, color=cp.array([1.0,1.0,1.0])):
    """
    Create a light emitter.
    position: 3D position (array-like)
    intensity: scalar
    color: RGB (cupy array)
    """
    return {
        "position": cp.array(position, dtype=cp.float32),
        "intensity": float(intensity),
        "color": cp.array(color, dtype=cp.float32)
    }


def accumulate_lighting_vectorized(positions, normal, emitters, kd=0.7, brightness=1.5):
    """
    Lighting with softer falloff and minimum brightness.
    Returns raw CuPy array 'colors'.
    """
    N = positions.shape[0]
    colors = cp.zeros((N, 3), dtype=cp.float32)

    for e in emitters:
        dir_vec = positions - e['position']
        dist2 = cp.sum(dir_vec ** 2, axis=1) + 1e-8
        dist = cp.sqrt(dist2)

        # Hybrid falloff
        inv_sq = kd * e["intensity"] / dist2
        linear = kd * e["intensity"] / (1.0 + dist * 0.1)

        blend_threshold = 3.0
        blend_factor = cp.clip((dist - blend_threshold) / blend_threshold, 0.0, 1.0)
        intensity = (1.0 - blend_factor) * inv_sq + blend_factor * linear

        # Minimum brightness so far-out stars are visible
        min_brightness = kd * e["intensity"] * 0.2
        intensity = cp.maximum(intensity, min_brightness)

        dir_norm = dir_vec / cp.sqrt(dist2)[:, cp.newaxis]
        dot = cp.clip(cp.sum(dir_norm * normal[cp.newaxis, :], axis=1), 0, 1)

        colors += (dot[:, None] * intensity[:, None]) * e["color"]

    # Apply brightness multiplier and return directly
    colors = colors * brightness
    return colors

def apply_color_temperature(colors, color_temp, brightness=1.0):
    """
    Apply warm/cool color temperature effect.
    colors: (N,3) CuPy array of RGB values
    color_temp: 0.0 (cool) to 1.0 (warm), 0.5 neutral
    brightness: overall scaling
    """
    colors = colors * brightness  # scale by brightness
    colors = cp.clip(colors, 0.0, 1.0)

    if color_temp > 0.5:  # warm
        warmth_factor = (color_temp - 0.5) * 2
        colors[:, 0] += 0.4 * warmth_factor  # red
        colors[:, 2] -= 0.3 * warmth_factor  # blue
    else:  # cool
        coolness_factor = (0.5 - color_temp) * 2
        colors[:, 2] += 0.3 * coolness_factor  # blue
        colors[:, 0] -= 0.2 * coolness_factor  # red

    # Clamp again
    colors = cp.clip(colors, 0.0, 1.0)
    return colors

def generate_star_colors(positions, normal, emitters, color_temp=0.5, brightness=1.0, kd=0.7):
    """
    Simpler version that bypasses potential issues in apply_color_temperature
    """
    # Step 1: Get basic lighting from your improved accumulate_lighting_vectorized
    colors_gpu = accumulate_lighting_vectorized(positions, normal, emitters, kd=kd)
    
    # Step 2: Apply brightness scaling directly (skip the problematic apply_color_temperature)
    colors_gpu = colors_gpu * brightness * 3.0  # Boost overall brightness
    
    # Step 3: Apply color temperature manually
    if color_temp > 0.5:  # warm
        warmth = (color_temp - 0.5) * 2
        colors_gpu[:, 0] += 0.3 * warmth  # more red
        colors_gpu[:, 2] -= 0.4 * warmth  # less blue
    else:  # cool
        coolness = (0.5 - color_temp) * 2
        colors_gpu[:, 2] += 0.5 * coolness  # more blue
        colors_gpu[:, 0] -= 0.2 * coolness  # less red
    
    # Step 4: Ensure minimum visibility and clamp
    colors_gpu = cp.maximum(colors_gpu, 0.27)  # Minimum 30% brightness
    colors_gpu = cp.clip(colors_gpu, 0.0, 1.0)
    
    return cp.asnumpy(colors_gpu)

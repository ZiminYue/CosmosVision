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

def accumulate_lighting_vectorized(positions, normal, emitters, kd=0.7):
    """
    Balanced lighting: bright center, visible edges, good gradient
    """
    N = positions.shape[0]
    colors = cp.zeros((N, 3), dtype=cp.float32)

    for e in emitters:
        dir_vec = positions - e['position']
        dist2 = cp.sum(dir_vec ** 2, axis=1) + 1e-8
        dist = cp.sqrt(dist2)
        
        # BALANCED APPROACH: Combine inverse-square with linear falloff
        # This keeps the bright center but prevents outer stars from being too dark
        inverse_square = kd * e["intensity"] / dist2  # Original bright center
        linear_falloff = kd * e["intensity"] / (1.0 + dist * 0.15)  # Gentler edges
        
        # Blend the two: use inverse-square for close stars, linear for far stars
        blend_factor = cp.clip(dist / 10.0, 0.0, 1.0)  # Transition at distance ~10
        intensity = (1.0 - blend_factor) * inverse_square + blend_factor * linear_falloff
        
        # Add small minimum brightness to prevent complete darkness
        min_intensity = kd * e["intensity"] * 0.08  # 8% minimum brightness
        intensity = cp.maximum(intensity, min_intensity)
        
        # Standard Lambert shading
        dir_norm = dir_vec / cp.sqrt(dist2)[:, cp.newaxis]
        dot = cp.clip(cp.sum(dir_norm * normal[cp.newaxis, :], axis=1), 0, 1)
        colors += (dot[:, cp.newaxis] * intensity[:, cp.newaxis]) * e["color"]

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
    Wrapper function: compute lighting and apply temperature & brightness.
    Returns NumPy array ready for scatter face_color
    """
    colors_gpu = accumulate_lighting_vectorized(positions, normal, emitters, kd=kd)
    colors_gpu = apply_color_temperature(colors_gpu, color_temp, brightness)
    return cp.asnumpy(colors_gpu)  # convert back to NumPy for vispy/scatter

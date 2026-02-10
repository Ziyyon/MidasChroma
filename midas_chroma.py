import os
import re
import json
import time
import random
import io
from pathlib import Path
from typing import Dict, Any, List, Tuple

import rawpy
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageStat, ImageOps, ImageFilter

VERSION = "Midas Chroma v11.0 (AI + RAW/JPEG Optimization)"


class Config:
    model = "gemini-3-flash-preview"
    max_images = 1000
    out_quality = 95
    sleep_between_calls = 1.2
    supported_exts = ('.jpg', '.jpeg', '.png', '.cr2', '.nef', '.arw', '.dng', '.rw2')
    raw_exts = ('.cr2', '.nef', '.arw', '.dng', '.rw2')
    offline_mode = False

CFG = Config()
CLIENT = None

PRESETS = {
    1: {"name": "Portrait Pro", "desc": "Warm tones, subtle contrast"},
    2: {"name": "Landscape Vivid", "desc": "High saturation, strong contrast"},
    3: {"name": "Urban Street", "desc": "Cool tones, gritty contrast"},
    4: {"name": "Light & Airy", "desc": "Bright, soft, Instagram style"},
    5: {"name": "Moody Noir", "desc": "Black & white, high contrast"},
    6: {"name": "Vintage Film", "desc": "Warm, faded, retro"},
    7: {"name": "Clean", "desc": "Clean, natural colors"},
    8: {"name": "Night Life", "desc": "Contrast boost for low light"},
    9: {"name": "Golden Hour", "desc": "Warm orange/gold tones"},
    10: {"name": "Cinematic", "desc": "Teal/orange Hollywood look"},
    11: {"name": "Modern Matte", "desc": "Lifted blacks, desaturated"},
    12: {"name": "Clean Corporate", "desc": "Neutral, sharp, professional"},
    13: {"name": "Smart", "desc": "Auto morning/night detection"},
    14: {"name": "Guibinga Silhouette", "desc": "Deep blue/purple, artistic"},
    15: {"name": "Afro-Pop Vivid", "desc": "Extremely vibrant colors"},
    16: {"name": "Neon Future", "desc": "Cyberpunk blue tint"},
    17: {"name": "Luxury Wedding", "desc": "Warm, S-curve, editorial"},
}


# =========================
# Image Conversion
# =========================

def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img):
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


# =========================
# Noise Detection & Removal
# =========================

def estimate_noise_level(img):
    """Estimate noise using Laplacian variance + shadow analysis."""
    cv_img = pil_to_cv2(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    noise_estimate = min(variance / 100, 50)
    
    dark_mask = gray < 80
    if np.any(dark_mask):
        dark_std = np.std(gray[dark_mask])
        shadow_noise = dark_std / 3
        noise_estimate = max(noise_estimate, shadow_noise)
    
    return noise_estimate


def get_denoise_strength(img, is_raw=True):
    """Get denoise strength. RAW typically needs more than JPEG."""
    noise_level = estimate_noise_level(img)
    
    # JPEGs have less visible noise (camera already denoised)
    if not is_raw:
        noise_level *= 0.7
    
    if noise_level < 5:
        return 0, "clean"
    elif noise_level < 12:
        return 1, "light"
    elif noise_level < 25:
        return 2, "medium"
    else:
        return 3, "heavy"


def denoise_chroma(img, strength=2):
    """Color noise reduction."""
    if strength == 0:
        return img
    
    cv_img = pil_to_cv2(img)
    ycrcb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    blur_size = {1: 3, 2: 5, 3: 7}.get(strength, 5)
    cr_d = cv2.GaussianBlur(cr, (blur_size, blur_size), 0)
    cb_d = cv2.GaussianBlur(cb, (blur_size, blur_size), 0)
    
    result = cv2.cvtColor(cv2.merge([y, cr_d, cb_d]), cv2.COLOR_YCrCb2BGR)
    return cv2_to_pil(result)


def denoise_bilateral(img, strength=2):
    """Bilateral filter - edge-preserving, good for skin."""
    if strength == 0:
        return img
    
    cv_img = pil_to_cv2(img)
    params = {1: (5, 20, 20), 2: (7, 40, 40), 3: (9, 75, 75)}
    d, sigmaColor, sigmaSpace = params.get(strength, params[2])
    
    denoised = cv2.bilateralFilter(cv_img, d, sigmaColor, sigmaSpace)
    return cv2_to_pil(denoised)


def denoise_adaptive(img, strength=2):
    """Adaptive denoising - more in shadows."""
    if strength == 0:
        return img
    
    cv_img = pil_to_cv2(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    shadow_mask = 1.0 - (gray.astype(np.float32) / 255.0)
    shadow_mask = np.clip(shadow_mask * 1.5, 0, 1)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
    
    params = {1: (4, 4), 2: (7, 7), 3: (12, 12)}
    h, hColor = params.get(strength, params[2])
    
    denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, h, hColor, 7, 21)
    
    shadow_mask_3ch = np.stack([shadow_mask] * 3, axis=-1)
    result = (denoised.astype(np.float32) * shadow_mask_3ch + 
              cv_img.astype(np.float32) * (1 - shadow_mask_3ch))
    
    return cv2_to_pil(np.clip(result, 0, 255).astype(np.uint8))


def apply_smart_denoise(img, is_raw=True):
    """Intelligent multi-stage denoising."""
    strength, noise_desc = get_denoise_strength(img, is_raw)
    
    if strength == 0:
        return img, {"strength": 0, "desc": noise_desc}
    
    result = img
    result = denoise_chroma(result, strength)
    result = denoise_adaptive(result, strength)
    
    if strength >= 2:
        result = denoise_bilateral(result, max(1, strength - 1))
    
    return result, {"strength": strength, "desc": noise_desc}


def simple_denoise_pil(img):
    """Fallback PIL-based denoise."""
    smooth = img.filter(ImageFilter.MedianFilter(size=3))
    return Image.blend(img, smooth, 0.5)


# =========================
# Core Helpers
# =========================

def get_average_brightness(img):
    return ImageStat.Stat(img.convert('L')).mean[0]


def get_saturation_level(img):
    """Estimate current saturation level."""
    r, g, b = img.split()
    r_avg = ImageStat.Stat(r).mean[0]
    g_avg = ImageStat.Stat(g).mean[0]
    b_avg = ImageStat.Stat(b).mean[0]
    
    rgb_max = max(r_avg, g_avg, b_avg)
    rgb_min = min(r_avg, g_avg, b_avg)
    
    if rgb_max == 0:
        return 0
    return (rgb_max - rgb_min) / rgb_max * 100


def apply_channel_mixer(img, r_adj, g_adj, b_adj):
    r, g, b = img.split()
    r = r.point(lambda i: min(255, int(i * r_adj)))
    g = g.point(lambda i: min(255, int(i * g_adj)))
    b = b.point(lambda i: min(255, int(i * b_adj)))
    return Image.merge('RGB', (r, g, b))


def apply_s_curve(img):
    def curve(x):
        x = x / 255.0
        return int((2 * x * x if x < 0.5 else 1 - 2 * (1 - x) * (1 - x)) * 255)
    return img.point([curve(i) for i in range(256)] * 3)


def apply_melanin_reflector(img, strength=1.0):
    """Lift shadows with warm skin-friendly tones."""
    grayscale = img.convert("L")
    mask = grayscale.filter(ImageFilter.GaussianBlur(radius=50))
    shadow_mask = ImageOps.invert(mask)

    fill = img.copy()
    brightness_boost = 1.0 + (0.3 * strength)
    fill = ImageEnhance.Brightness(fill).enhance(brightness_boost)
    
    r_mult = 1.0 + (0.06 * strength)
    g_mult = 1.0 + (0.03 * strength)
    
    r, g, b = fill.split()
    r = r.point(lambda i: min(255, int(i * r_mult)))
    g = g.point(lambda i: min(255, int(i * g_mult)))
    fill = Image.merge('RGB', (r, g, b))

    return Image.composite(fill, img, shadow_mask)


def lift_shadows_and_gamma(img, is_raw=True):
    """Lift shadows. RAW needs more aggressive lift than JPEG."""
    brightness = get_average_brightness(img)
    
    if is_raw:
        if brightness < 90:
            gamma = 0.65
            img = img.point(lambda p: int(255 * ((p / 255) ** gamma)))
            img = ImageEnhance.Color(img).enhance(1.1)
        elif brightness < 120:
            gamma = 0.8
            img = img.point(lambda p: int(255 * ((p / 255) ** gamma)))
    else:
        # JPEG: More conservative
        if brightness < 70:
            gamma = 0.75
            img = img.point(lambda p: int(255 * ((p / 255) ** gamma)))
        elif brightness < 100:
            gamma = 0.88
            img = img.point(lambda p: int(255 * ((p / 255) ** gamma)))
    
    return img


# =========================
# Base Development Pipelines
# =========================

def develop_raw(img):
    """Development pipeline for RAW files."""
    noise_info = {"strength": 0, "desc": "clean"}
    
    try:
        img, noise_info = apply_smart_denoise(img, is_raw=True)
    except Exception:
        img = simple_denoise_pil(img)
        noise_info = {"strength": -1, "desc": "fallback"}
    
    img = lift_shadows_and_gamma(img, is_raw=True)
    img = apply_melanin_reflector(img, strength=1.0)
    
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Color(img).enhance(1.3)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    
    return img, noise_info


def develop_jpeg(img):
    """Development pipeline for JPEG files - more conservative."""
    noise_info = {"strength": 0, "desc": "clean"}
    current_sat = get_saturation_level(img)
    
    try:
        img, noise_info = apply_smart_denoise(img, is_raw=False)
    except Exception:
        img = simple_denoise_pil(img)
        noise_info = {"strength": -1, "desc": "fallback"}
    
    img = lift_shadows_and_gamma(img, is_raw=False)
    img = apply_melanin_reflector(img, strength=0.6)
    
    img = ImageEnhance.Contrast(img).enhance(1.08)
    
    # Adaptive saturation
    if current_sat < 25:
        img = ImageEnhance.Color(img).enhance(1.15)
    elif current_sat < 40:
        img = ImageEnhance.Color(img).enhance(1.08)
    else:
        img = ImageEnhance.Color(img).enhance(1.02)
    
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    
    noise_info["saturation"] = current_sat
    return img, noise_info


def load_image(file_path: str) -> Tuple[Image.Image, Dict[str, Any]]:
    """Load image with appropriate processing."""
    ext = os.path.splitext(file_path)[1].lower()
    is_raw = ext in CFG.raw_exts
    
    meta = {
        "path": file_path,
        "is_raw": is_raw,
        "type": "RAW" if is_raw else "JPEG"
    }
    
    if is_raw:
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=False,
                    bright=1.4,
                    noise_thr=15
                )
                img = Image.fromarray(rgb)
                img, noise_info = develop_raw(img)
                meta["noise"] = noise_info
                meta["brightness"] = get_average_brightness(img)
                return img, meta
        except Exception as e:
            return None, {"error": str(e)}
    else:
        img = Image.open(file_path).convert("RGB")
        meta["original_saturation"] = get_saturation_level(img)
        img, noise_info = develop_jpeg(img)
        meta["noise"] = noise_info
        meta["brightness"] = get_average_brightness(img)
        return img, meta


# =========================
# Local Analysis
# =========================

def analyze_image_locally(img: Image.Image, is_raw: bool) -> Dict[str, Any]:
    """Analyze image without AI."""
    brightness = get_average_brightness(img)
    saturation = get_saturation_level(img)
    
    r, g, b = img.split()
    r_avg = ImageStat.Stat(r).mean[0]
    b_avg = ImageStat.Stat(b).mean[0]
    warmth = (r_avg - b_avg) / 255
    
    contrast = ImageStat.Stat(img.convert('L')).stddev[0]
    
    return {
        "brightness": brightness,
        "saturation": saturation,
        "warmth": warmth,
        "contrast": contrast,
        "is_raw": is_raw,
    }


def select_preset_locally(img: Image.Image, is_raw: bool) -> Dict[str, Any]:
    """Select preset based on local analysis."""
    a = analyze_image_locally(img, is_raw)
    brightness, saturation = a["brightness"], a["saturation"]
    warmth, contrast = a["warmth"], a["contrast"]
    
    if brightness < 60:
        return {"preset": 8, "reasoning": f"Dark ({brightness:.0f})"}
    if brightness < 90 and warmth > 0.05:
        return {"preset": 9, "reasoning": "Dark warm"}
    if saturation > 35 and brightness > 120:
        return {"preset": 7, "reasoning": "Colorful bright"}
    if saturation > 35:
        return {"preset": 15, "reasoning": "Colorful"}
    if warmth < -0.05 and brightness > 150:
        return {"preset": 4, "reasoning": "Cool bright"}
    if warmth < -0.05:
        return {"preset": 3, "reasoning": "Cool tones"}
    if warmth > 0.08:
        return {"preset": 9, "reasoning": "Warm tones"}
    if contrast < 40:
        return {"preset": 17, "reasoning": "Flat image"}
    if contrast > 70:
        return {"preset": 7, "reasoning": "Good contrast"}
    if 100 < brightness < 180:
        return {"preset": 7, "reasoning": "Balanced"}
    if brightness > 180:
        return {"preset": 1, "reasoning": "Bright"}
    return {"preset": 7, "reasoning": "Default"}


def select_shoot_preset_locally(images: List[Tuple[Image.Image, bool]]) -> Dict[str, Any]:
    """Analyze shoot from multiple images."""
    if not images:
        return {"shoot_type": "event", "recommended_preset": 7, "reasoning": "Default"}
    
    analyses = [analyze_image_locally(img, is_raw) for img, is_raw in images]
    avg = lambda k: sum(a[k] for a in analyses) / len(analyses)
    
    avg_brightness = avg("brightness")
    avg_saturation = avg("saturation")
    avg_warmth = avg("warmth")
    avg_contrast = avg("contrast")
    raw_count = sum(1 for a in analyses if a["is_raw"])
    
    base = {
        "avg_brightness": avg_brightness,
        "raw_count": raw_count,
        "jpeg_count": len(analyses) - raw_count
    }
    
    if avg_saturation > 30 and avg_brightness > 80:
        return {**base, "shoot_type": "colorful_event", "recommended_preset": 7, "reasoning": "Colorful event"}
    if avg_brightness < 80:
        return {**base, "shoot_type": "low_light", "recommended_preset": 8, "reasoning": "Low light"}
    if avg_warmth > 0.06:
        return {**base, "shoot_type": "warm", "recommended_preset": 9, "reasoning": "Warm tones"}
    if avg_contrast < 45:
        return {**base, "shoot_type": "soft", "recommended_preset": 17, "reasoning": "Soft light"}
    return {**base, "shoot_type": "general", "recommended_preset": 7, "reasoning": "Balanced"}


# =========================
# Gemini AI
# =========================

def get_client():
    global CLIENT
    if CLIENT is None:
        try:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("⚠ No GEMINI_API_KEY. Using local mode.")
                CFG.offline_mode = True
                return None
            CLIENT = genai.Client(api_key=api_key)
        except ImportError:
            print("⚠ google-genai not installed. Using local mode.")
            CFG.offline_mode = True
            return None
    return CLIENT


def build_preset_prompt():
    return "\n".join([f"  {n}. {p['name']} - {p['desc']}" for n, p in PRESETS.items()])


def gemini_analyze_shoot(images: List[Tuple[Image.Image, bool]]) -> Dict[str, Any]:
    """Analyze shoot with AI."""
    local_result = select_shoot_preset_locally(images)
    if CFG.offline_mode:
        return local_result
    
    from google.genai import types
    client = get_client()
    if not client:
        return local_result
    
    parts = []
    for img, is_raw in images[:3]:
        buf = io.BytesIO()
        preview = img.copy()
        preview.thumbnail((384, 384), Image.LANCZOS)
        preview.save(buf, format="JPEG", quality=65)
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
        parts.append(f" [{'RAW' if is_raw else 'JPEG'}] ")
    
    parts.append(f"""Analyze these images and recommend the best editing preset.

{build_preset_prompt()}

Return JSON:
{{"shoot_type": "wedding/event/portrait/etc", "recommended_preset": 1-17, "reasoning": "brief explanation"}}""")

    schema = {
        "type": "object",
        "properties": {
            "shoot_type": {"type": "string"},
            "recommended_preset": {"type": "integer"},
            "reasoning": {"type": "string"}
        },
        "required": ["shoot_type", "recommended_preset", "reasoning"]
    }
    
    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema)
    
    for attempt in range(3):
        try:
            resp = client.models.generate_content(model=CFG.model, contents=parts, config=config)
            result = json.loads(resp.text)
            preset = result.get("recommended_preset", 7)
            result["recommended_preset"] = preset if 1 <= preset <= 17 else 7
            return {**local_result, **result}
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                match = re.search(r"(\d+)s", str(e))
                wait = min(int(match.group(1)) if match else 30, 60)
                print(f"\n  ⏳ Waiting {wait}s...", end="")
                time.sleep(wait)
                continue
            print(f"\n  ⚠ {e}")
            break
    return local_result


def gemini_select_preset(img: Image.Image, is_raw: bool, shoot_context: Dict[str, Any]) -> Dict[str, Any]:
    """Select preset with AI."""
    local_result = select_preset_locally(img, is_raw)
    if CFG.offline_mode:
        return local_result
    
    from google.genai import types
    client = get_client()
    if not client:
        return local_result
    
    buf = io.BytesIO()
    preview = img.copy()
    preview.thumbnail((512, 512), Image.LANCZOS)
    preview.save(buf, format="JPEG", quality=70)
    img_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
    
    default = shoot_context.get("recommended_preset", 7)
    file_type = "RAW" if is_raw else "JPEG"
    
    prompt = f"""Select preset for this {file_type} image.
Default: {default} ({PRESETS[default]['name']})

{build_preset_prompt()}

Return JSON: {{"preset": 1-17, "reasoning": "brief explanation"}}"""

    schema = {
        "type": "object",
        "properties": {"preset": {"type": "integer"}, "reasoning": {"type": "string"}},
        "required": ["preset", "reasoning"]
    }
    
    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema)
    
    for attempt in range(3):
        try:
            resp = client.models.generate_content(model=CFG.model, contents=[img_part, prompt], config=config)
            result = json.loads(resp.text)
            preset = result.get("preset", default)
            result["preset"] = preset if 1 <= preset <= 17 else default
            return result
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                match = re.search(r"(\d+)s", str(e))
                wait = min(int(match.group(1)) if match else 30, 60)
                print(f"\n  ⏳ Waiting {wait}s...", end="")
                time.sleep(wait)
                continue
            print(f"\n  ⚠ {e}")
            break
    return local_result


# =========================
# 17 Preset Styles (RAW/JPEG Aware)
# =========================

def style_1(img, is_raw=True):
    """Portrait Pro"""
    img = apply_channel_mixer(img, 1.05, 1.0, 0.95)
    contrast = 1.05 if is_raw else 1.02
    color = 1.1 if is_raw else 1.05
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(color)
    return img


def style_2(img, is_raw=True):
    """Landscape Vivid"""
    color = 1.3 if is_raw else 1.15
    contrast = 1.2 if is_raw else 1.1
    img = ImageEnhance.Color(img).enhance(color)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def style_3(img, is_raw=True):
    """Urban Street"""
    img = apply_channel_mixer(img, 0.95, 1.0, 1.05)
    contrast = 1.25 if is_raw else 1.12
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def style_4(img, is_raw=True):
    """Light & Airy"""
    img = apply_channel_mixer(img, 1.02, 0.98, 1.0)
    img = ImageEnhance.Contrast(img).enhance(0.95)
    brightness = 1.05 if is_raw else 1.02
    img = ImageEnhance.Brightness(img).enhance(brightness)
    return img


def style_5(img, is_raw=True):
    """Moody Noir"""
    img = ImageOps.grayscale(img).convert("RGB")
    contrast = 1.4 if is_raw else 1.25
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def style_6(img, is_raw=True):
    """Vintage Film"""
    r_adj = 1.1 if is_raw else 1.05
    img = apply_channel_mixer(img, r_adj, 1.05, 0.9)
    img = ImageEnhance.Contrast(img).enhance(0.9)
    return img


def style_7(img, is_raw=True):
    """Clean"""
    color = 1.05 if is_raw else 1.02
    img = ImageEnhance.Color(img).enhance(color)
    return img


def style_8(img, is_raw=True):
    """Night Life"""
    contrast = 1.15 if is_raw else 1.08
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def style_9(img, is_raw=True):
    """Golden Hour"""
    r_adj = 1.15 if is_raw else 1.08
    g_adj = 1.05 if is_raw else 1.02
    img = apply_channel_mixer(img, r_adj, g_adj, 0.9)
    brightness = 1.05 if is_raw else 1.02
    img = ImageEnhance.Brightness(img).enhance(brightness)
    return img


def style_10(img, is_raw=True):
    """Cinematic Teal/Orange"""
    r_adj = 1.1 if is_raw else 1.05
    img = apply_channel_mixer(img, r_adj, 0.95, 0.9)
    contrast = 1.15 if is_raw else 1.08
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def style_11(img, is_raw=True):
    """Modern Matte"""
    img = ImageEnhance.Contrast(img).enhance(0.9)
    brightness = 1.1 if is_raw else 1.05
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Color(img).enhance(0.9)
    return img


def style_12(img, is_raw=True):
    """Clean Corporate"""
    img = ImageEnhance.Color(img).enhance(0.95)
    sharpness = 1.2 if is_raw else 1.1
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = apply_channel_mixer(img, 1.0, 1.0, 1.02)
    return img


def style_13(img, is_raw=True):
    """Christmas Smart"""
    avg = get_average_brightness(img)
    if avg > 90:
        img = apply_channel_mixer(img, 1.05, 0.98, 0.95)
        color = 1.0 if is_raw else 0.98
        img = ImageEnhance.Color(img).enhance(color)
        brightness = 1.05 if is_raw else 1.02
        img = ImageEnhance.Brightness(img).enhance(brightness)
    else:
        img = apply_channel_mixer(img, 1.05, 0.98, 0.95)
        color = 0.85 if is_raw else 0.9
        img = ImageEnhance.Color(img).enhance(color)
        contrast = 1.2 if is_raw else 1.1
        img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def style_14(img, is_raw=True):
    """Guibinga Silhouette"""
    contrast = 1.4 if is_raw else 1.25
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(0.85)
    img = apply_channel_mixer(img, 1.0, 0.9, 1.25)
    color = 1.4 if is_raw else 1.2
    img = ImageEnhance.Color(img).enhance(color)
    return img


def style_15(img, is_raw=True):
    """Afro-Pop Vivid"""
    color = 1.3 if is_raw else 1.0
    img = ImageEnhance.Color(img).enhance(color)
    gamma = 0.8 if is_raw else 0.88
    img = img.point(lambda p: int(255 * ((p / 255) ** gamma)))
    contrast = 1.15 if is_raw else 1.08
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def style_16(img, is_raw=True):
    """Neon Future"""
    brightness = 0.7 if is_raw else 0.8
    img = ImageEnhance.Brightness(img).enhance(brightness)
    r, g, b = img.split()
    r_mult = 0.8 if is_raw else 0.85
    g_mult = 0.85 if is_raw else 0.88
    b_mult = 1.5 if is_raw else 1.3
    r = r.point(lambda i: int(i * r_mult))
    g = g.point(lambda i: int(i * g_mult))
    b = b.point(lambda i: min(255, int(i * b_mult)))
    img = Image.merge('RGB', (r, g, b))
    contrast = 1.5 if is_raw else 1.3
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def style_17(img, is_raw=True):
    """Luxury Wedding"""
    r_adj = 1.08 if is_raw else 1.04
    img = apply_channel_mixer(img, r_adj, 0.95, 0.92)
    color = 0.9 if is_raw else 0.95
    img = ImageEnhance.Color(img).enhance(color)
    img = apply_s_curve(img)
    brightness = 1.05 if is_raw else 1.02
    img = ImageEnhance.Brightness(img).enhance(brightness)
    sharpness = 1.3 if is_raw else 1.15
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img


STYLE_FUNCS = {
    1: style_1, 2: style_2, 3: style_3, 4: style_4, 5: style_5, 6: style_6,
    7: style_7, 8: style_8, 9: style_9, 10: style_10, 11: style_11, 12: style_12,
    13: style_13, 14: style_14, 15: style_15, 16: style_16, 17: style_17
}


def apply_preset(img, num, is_raw=True):
    """Apply preset with RAW/JPEG awareness."""
    return STYLE_FUNCS.get(num, style_7)(img, is_raw)


# =========================
# Main
# =========================

def main():
    print(f"\n{'='*65}")
    print(f"  {VERSION}")
    print(f"  AI Preset Selection + RAW/JPEG Optimization + Smart Denoise")
    print(f"{'='*65}\n")

    import sys
    if "--offline" in sys.argv:
        CFG.offline_mode = True
    
    get_client()
    mode = "🤖 AI MODE" if not CFG.offline_mode else "📊 LOCAL MODE"
    print(f"{mode}\n")

    folder = input("Enter folder name: ").strip()
    input_folder = Path.cwd() / folder
    output_folder = Path.cwd() / f"{folder}_midas"

    if not input_folder.exists():
        raise SystemExit(f"❌ Not found: {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in input_folder.iterdir() if p.suffix.lower() in CFG.supported_exts])
    if not files:
        raise SystemExit("❌ No images found.")

    files = files[:CFG.max_images]
    print(f"📁 {len(files)} images\n")

    # Shoot analysis
    print("🔍 Analyzing shoot...")
    samples = []
    for p in random.sample(files, min(3, len(files))):
        img, meta = load_image(str(p))
        if img:
            samples.append((img, meta.get("is_raw", False)))
    
    shoot = gemini_analyze_shoot(samples)
    default_preset = shoot.get("recommended_preset", 7)
    
    print(f"   Type: {shoot.get('shoot_type', 'event')}")
    print(f"   Preset: {default_preset} ({PRESETS[default_preset]['name']})")
    print(f"   Reason: {shoot.get('reasoning', 'N/A')}\n")

    manifest = {"version": VERSION, "shoot": shoot, "images": []}
    raw_count, jpeg_count = 0, 0

    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {p.name}", end="")

        try:
            img, meta = load_image(str(p))
            if not img:
                print(" ❌")
                continue

            is_raw = meta.get("is_raw", False)
            file_type = "RAW" if is_raw else "JPEG"
            
            if is_raw:
                raw_count += 1
            else:
                jpeg_count += 1
            
            noise = meta.get("noise", {})
            noise_desc = noise.get("desc", "?")
            
            print(f" [{file_type}]", end="")
            print(f" [Noise: {noise_desc}]", end="")
            
            if not is_raw and "saturation" in noise:
                print(f" [Sat: {noise['saturation']:.0f}%]", end="")

            selection = gemini_select_preset(img, is_raw, shoot)
            preset = selection["preset"]
            
            print(f" → P{preset}", end="")

            img = apply_preset(img, preset, is_raw)

            out_path = output_folder / f"{p.stem}_midas.jpg"
            img.save(str(out_path), quality=CFG.out_quality, subsampling=0)

            manifest["images"].append({
                "file": p.name,
                "type": file_type,
                "meta": meta,
                "preset": preset,
                "preset_name": PRESETS[preset]["name"],
                "reasoning": selection["reasoning"],
                "output": out_path.name
            })

            print(" ✓")

            if not CFG.offline_mode and i < len(files):
                time.sleep(CFG.sleep_between_calls)

        except Exception as e:
            print(f" ❌ {e}")

    with open(output_folder / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*65}")
    print(f"✅ Done: {output_folder}")
    print(f"   RAW files: {raw_count} | JPEG files: {jpeg_count}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
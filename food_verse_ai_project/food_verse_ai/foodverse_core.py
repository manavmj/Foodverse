"""Core services and deterministic demo logic for Food Verse AI.

No database is used. API credentials are passed in at runtime and are never written to disk.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import tempfile
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageOps

MODEL_DEFAULTS = {
    "Google Gemini": "gemini-2.5-flash",
    "Groq": "meta-llama/llama-4-scout-17b-16e-instruct",
}


def provider_env_key(provider: str) -> str:
    return {"Google Gemini": "GEMINI_API_KEY", "Groq": "GROQ_API_KEY"}.get(provider, "")


def get_env_or_secret(provider: str) -> str:
    """Read a key from environment or Streamlit secrets without failing locally."""
    env_name = provider_env_key(provider)
    if not env_name:
        return ""
    value = os.getenv(env_name, "")
    if value:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(env_name, "") or "")
    except Exception:
        return ""


def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON from a model response, including responses wrapped in fences."""
    if not text or not text.strip():
        raise ValueError("The model returned an empty response.")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The model response was not valid JSON.")
        data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from the model.")
    return data


def call_model(
    provider: str,
    api_key: str,
    model: str,
    prompt: str,
    image_bytes: Optional[bytes] = None,
    mime_type: str = "image/jpeg",
) -> str:
    """Call Gemini or Groq. Imports are lazy so demo mode has minimal requirements."""
    if not api_key:
        raise ValueError("No API key provided.")
    if provider == "Google Gemini":
        from google import genai

        client = genai.Client(api_key=api_key)
        if image_bytes is not None:
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                response = client.models.generate_content(
                    model=model or MODEL_DEFAULTS[provider],
                    contents=[prompt, image],
                )
        else:
            response = client.models.generate_content(
                model=model or MODEL_DEFAULTS[provider],
                contents=prompt,
            )
        result = getattr(response, "text", None)
        if not result:
            raise RuntimeError("Gemini returned no text content.")
        return str(result)

    if provider == "Groq":
        from groq import Groq

        client = Groq(api_key=api_key)
        if image_bytes is not None:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            data_url = f"data:{mime_type};base64,{encoded}"
            content: Any = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        else:
            content = prompt
        response = client.chat.completions.create(
            model=model or MODEL_DEFAULTS[provider],
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            max_tokens=2200,
        )
        result = response.choices[0].message.content
        if not result:
            raise RuntimeError("Groq returned no text content.")
        return str(result)

    raise ValueError("Choose Google Gemini or Groq to enable live analysis.")


def call_model_json(
    provider: str,
    api_key: str,
    model: str,
    prompt: str,
    image_bytes: Optional[bytes] = None,
    mime_type: str = "image/jpeg",
) -> Dict[str, Any]:
    text = call_model(provider, api_key, model, prompt, image_bytes, mime_type)
    return parse_json_response(text)


def image_to_png_bytes(image: Image.Image, max_width: int = 1400) -> bytes:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.width > max_width:
        new_h = max(1, round(image.height * (max_width / image.width)))
        image = image.resize((max_width, new_h))
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()


def video_contact_sheet(video_bytes: bytes, max_frames: int = 4) -> bytes:
    """Sample a small set of frames from a video and return a contact sheet as PNG."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video support requires opencv-python-headless.") from exc

    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(video_bytes)
        path = temp.name
    frames: List[Image.Image] = []
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("Could not open this video. Try an MP4 or upload a still image.")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            sample_indices = [0]
        else:
            sample_indices = sorted(set(int(i * max(0, total - 1) / max(1, max_frames - 1)) for i in range(max_frames)))
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil.thumbnail((600, 400))
            frames.append(pil.copy())
    finally:
        if cap is not None:
            cap.release()
        try:
            os.unlink(path)
        except OSError:
            pass
    if not frames:
        raise ValueError("No readable frames were found in this video.")

    cell_w, cell_h = 600, 430
    sheet = Image.new("RGB", (cell_w * min(2, len(frames)), cell_h * ((len(frames) + 1) // 2)), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    for i, frame in enumerate(frames):
        frame.thumbnail((cell_w - 16, cell_h - 42))
        x = (i % 2) * cell_w + (cell_w - frame.width) // 2
        y = (i // 2) * cell_h + 30
        sheet.paste(frame, (x, y))
        draw.text(((i % 2) * cell_w + 12, (i // 2) * cell_h + 10), f"Sample frame {i + 1}", fill="#243047")
    return image_to_png_bytes(sheet, max_width=1200)


FOOD_DEMO_SAMPLES: Dict[str, Dict[str, Any]] = {
    "Balanced grain bowl": {
        "dish_name": "Balanced grain bowl (demo sample)", "cuisine": "Contemporary / fusion",
        "portion_estimate": "1 medium bowl; roughly 350–450 g (illustrative)",
        "ingredients": [{"name": "whole grain", "estimated_grams": 140}, {"name": "mixed vegetables", "estimated_grams": 140}, {"name": "beans or legumes", "estimated_grams": 90}],
        "nutrition": {"calories_kcal": 520, "protein_g": 19, "carbohydrates_g": 76, "fat_g": 14, "fiber_g": 13},
        "possible_allergens": ["Depends on dressing, sauces, and toppings; verify labels"], "health_score": 84,
        "health_score_notes": "Illustrative score favors fiber-rich plants and a balanced macro mix; it is not a clinical rating.",
        "recommendations": ["Use a generous variety of vegetables.", "Choose a lower-sodium dressing and measure added oil."],
        "confidence": "Demo only", "assumptions": ["Sample data is not derived from the uploaded photo."]
    },
    "Vegetable pasta": {
        "dish_name": "Vegetable pasta (demo sample)", "cuisine": "Italian-inspired",
        "portion_estimate": "1 medium plate; roughly 300–400 g (illustrative)",
        "ingredients": [{"name": "cooked pasta", "estimated_grams": 200}, {"name": "mixed vegetables and tomato sauce", "estimated_grams": 150}, {"name": "olive oil or cheese", "estimated_grams": 15}],
        "nutrition": {"calories_kcal": 560, "protein_g": 18, "carbohydrates_g": 83, "fat_g": 16, "fiber_g": 9},
        "possible_allergens": ["Wheat/gluten", "Milk if cheese is used"], "health_score": 73,
        "health_score_notes": "Illustrative score; preparation, portion size, and sauce can change the result substantially.",
        "recommendations": ["Add beans or lentils for more protein.", "Add vegetables and check sodium in jarred sauces."],
        "confidence": "Demo only", "assumptions": ["Sample data is not derived from the uploaded photo."]
    },
    "Dal and rice": {
        "dish_name": "Dal and rice (demo sample)", "cuisine": "Indian",
        "portion_estimate": "1 meal plate; roughly 350–450 g (illustrative)",
        "ingredients": [{"name": "cooked lentil dal", "estimated_grams": 220}, {"name": "cooked rice", "estimated_grams": 160}, {"name": "tempering oil", "estimated_grams": 8}],
        "nutrition": {"calories_kcal": 490, "protein_g": 19, "carbohydrates_g": 78, "fat_g": 10, "fiber_g": 12},
        "possible_allergens": ["May contain dairy, nuts, or other allergens depending on recipe"], "health_score": 82,
        "health_score_notes": "Illustrative score recognizes legumes and fiber; oil, salt, and rice portion affect the total.",
        "recommendations": ["Include a vegetable side or salad.", "Adjust rice portion and add protein to fit personal needs."],
        "confidence": "Demo only", "assumptions": ["Sample data is not derived from the uploaded photo."]
    },
    "Paneer veggie wrap": {
        "dish_name": "Paneer veggie wrap (demo sample)", "cuisine": "Indian fusion",
        "portion_estimate": "1 large wrap; roughly 300–400 g (illustrative)",
        "ingredients": [{"name": "wheat wrap", "estimated_grams": 75}, {"name": "paneer", "estimated_grams": 90}, {"name": "vegetables", "estimated_grams": 110}],
        "nutrition": {"calories_kcal": 590, "protein_g": 27, "carbohydrates_g": 55, "fat_g": 27, "fiber_g": 8},
        "possible_allergens": ["Milk", "Wheat/gluten"], "health_score": 70,
        "health_score_notes": "Illustrative score; paneer, sauces, cooking oil, and wrap size are major variables.",
        "recommendations": ["Add extra vegetables and a yogurt-based sauce if suitable.", "Choose a whole-grain wrap and moderate added oil."],
        "confidence": "Demo only", "assumptions": ["Sample data is not derived from the uploaded photo."]
    },
    "Chickpea salad": {
        "dish_name": "Chickpea salad (demo sample)", "cuisine": "Mediterranean-inspired",
        "portion_estimate": "1 large bowl; roughly 300–400 g (illustrative)",
        "ingredients": [{"name": "chickpeas", "estimated_grams": 170}, {"name": "salad vegetables", "estimated_grams": 170}, {"name": "dressing", "estimated_grams": 15}],
        "nutrition": {"calories_kcal": 440, "protein_g": 17, "carbohydrates_g": 60, "fat_g": 14, "fiber_g": 15},
        "possible_allergens": ["Depends on dressing and add-ins; verify labels"], "health_score": 88,
        "health_score_notes": "Illustrative score emphasizes legumes, vegetables, and fiber; dressing and portion sizes vary.",
        "recommendations": ["Rinse canned chickpeas to reduce sodium.", "Pair with a suitable protein source if higher protein is needed."],
        "confidence": "Demo only", "assumptions": ["Sample data is not derived from the uploaded photo."]
    },
}


def demo_food_analysis(sample_name: str) -> Dict[str, Any]:
    result = FOOD_DEMO_SAMPLES.get(sample_name, FOOD_DEMO_SAMPLES["Balanced grain bowl"])
    # Return a deep-enough copy so UI modifications don't mutate the catalog.
    return json.loads(json.dumps(result))


def demo_freshness(scenario: str, item_type: str) -> Dict[str, Any]:
    if "mold" in scenario.lower() or "slimy" in scenario.lower():
        stage, action = "Spoiled", "Discard immediately"
        indicators = ["Demo scenario includes visible mold or sliminess", "Visual appearance alone cannot identify all pathogens"]
        tips = ["Discard the affected item; do not taste it.", "For soft/high-moisture foods, discard rather than cutting around mold.", "Clean hands, utensils, and any surfaces that contacted the item."]
    elif "wilt" in scenario.lower() or "wrinkl" in scenario.lower():
        stage, action = "Moderately Fresh", "Consume within 24h"
        indicators = ["Demo scenario includes mild wilting or wrinkling", "No claim is made about hidden microbial hazards"]
        tips = ["Refrigerate promptly if the item is normally refrigerated.", "Use soon only if it has been stored correctly and has no off odor, slime, or mold.", "Discard if storage history is uncertain or other spoilage signs appear."]
    else:
        stage, action = "Fresh", "Safe to consume"
        indicators = ["Demo scenario assumes intact appearance", "An intact appearance does not prove microbiological safety"]
        tips = ["Follow package dates and recommended refrigeration.", "Keep raw meat and seafood separate from ready-to-eat foods.", "Wash produce under running water; do not wash raw poultry."]
    return {
        "item_name": f"{item_type} (demo scenario)", "freshness_stage": stage, "visual_indicators": indicators,
        "action": action, "storage_tips": tips, "confidence": "Demo only",
        "caution": "Illustrative scenario only; it does not analyze the uploaded image or test for pathogens."
    }


def demo_kitchen_analysis(scenario: str) -> Dict[str, Any]:
    if "strong" in scenario.lower():
        score = 92
        checks = [
            {"indicator": "Hair restraint / headwear", "status": "Pass", "evidence": "Demo scenario assumes effective hair restraint is visible.", "severity": "Low", "recommendation": "Maintain the policy and verify coverage during each shift."},
            {"indicator": "Glove use / hand hygiene", "status": "Pass", "evidence": "Demo scenario assumes appropriate hand hygiene and glove practices.", "severity": "Low", "recommendation": "Change gloves between tasks; gloves do not replace handwashing."},
            {"indicator": "Surface sanitization", "status": "Pass", "evidence": "Demo scenario assumes cleaned and sanitized food-contact surfaces.", "severity": "Low", "recommendation": "Follow sanitizer concentration and contact-time instructions."},
            {"indicator": "Raw / ready-to-eat separation", "status": "Pass", "evidence": "Demo scenario assumes separate zones and utensils.", "severity": "Low", "recommendation": "Keep raw animal products below and apart from ready-to-eat foods."},
        ]
    elif "multiple" in scenario.lower():
        score = 38
        checks = [
            {"indicator": "Hair restraint / headwear", "status": "Concern", "evidence": "Demo scenario assumes hair is uncovered.", "severity": "Medium", "recommendation": "Provide and enforce suitable hair restraints."},
            {"indicator": "Glove use / hand hygiene", "status": "Concern", "evidence": "Demo scenario assumes inconsistent hand hygiene or glove changes.", "severity": "High", "recommendation": "Stop and correct unsafe handling; wash hands and change gloves between tasks."},
            {"indicator": "Surface sanitization", "status": "Concern", "evidence": "Demo scenario assumes visibly soiled food-contact surfaces.", "severity": "High", "recommendation": "Clean, rinse if required, and sanitize using product label contact time."},
            {"indicator": "Raw / ready-to-eat separation", "status": "Concern", "evidence": "Demo scenario assumes a cross-contamination hazard.", "severity": "High", "recommendation": "Separate raw animal products, ready-to-eat foods, utensils, and prep boards."},
        ]
    else:
        score = 64
        checks = [
            {"indicator": "Hair restraint / headwear", "status": "Concern", "evidence": "Demo scenario assumes headwear is unclear or inconsistently used.", "severity": "Medium", "recommendation": "Ensure hair is fully restrained in food preparation areas."},
            {"indicator": "Glove use / hand hygiene", "status": "Pass", "evidence": "Demo scenario assumes some controls are visible, but behavior cannot be confirmed from a still image.", "severity": "Low", "recommendation": "Use task-based glove changes and frequent handwashing."},
            {"indicator": "Surface sanitization", "status": "Concern", "evidence": "Demo scenario assumes sanitization status is not verifiable.", "severity": "Medium", "recommendation": "Use a documented cleaning schedule and verify sanitizer contact time."},
            {"indicator": "Raw / ready-to-eat separation", "status": "Concern", "evidence": "Demo scenario assumes raw ingredients may be near produce.", "severity": "High", "recommendation": "Physically separate raw meat/poultry from produce and ready-to-eat items."},
        ]
    return {
        "compliance_score": score, "checks": checks,
        "summary": "Demonstration checklist only; observations are not inferred from the uploaded media.",
        "confidence": "Demo only"
    }


def normalize_food_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Defensively normalize the model payload before rendering it."""
    result = dict(data)
    result.setdefault("dish_name", "Unidentified dish")
    result.setdefault("cuisine", "Not confidently determined")
    result.setdefault("portion_estimate", "Not confidently estimated")
    result.setdefault("ingredients", [])
    result.setdefault("possible_allergens", [])
    result.setdefault("recommendations", [])
    result.setdefault("assumptions", [])
    result.setdefault("confidence", "Unspecified")
    result.setdefault("health_score_notes", "Score is an estimate, not a validated clinical measure.")
    nutrition = result.get("nutrition") if isinstance(result.get("nutrition"), dict) else {}
    normalized_nutrition = {}
    for key, default in [("calories_kcal", 0), ("protein_g", 0), ("carbohydrates_g", 0), ("fat_g", 0), ("fiber_g", 0)]:
        try:
            normalized_nutrition[key] = max(0, float(nutrition.get(key, default) or default))
        except (TypeError, ValueError):
            normalized_nutrition[key] = float(default)
    result["nutrition"] = normalized_nutrition
    try:
        result["health_score"] = max(1, min(100, int(float(result.get("health_score", 50)))))
    except (ValueError, TypeError):
        result["health_score"] = 50
    if not isinstance(result["ingredients"], list):
        result["ingredients"] = [str(result["ingredients"])]
    for field in ["possible_allergens", "recommendations", "assumptions"]:
        if not isinstance(result[field], list):
            result[field] = [str(result[field])] if result[field] else []
    return result


def normalize_freshness_data(data: Dict[str, Any], item_type: str) -> Dict[str, Any]:
    result = dict(data)
    stage = str(result.get("freshness_stage", "Moderately Fresh"))
    valid_stages = {"fresh": "Fresh", "moderately fresh": "Moderately Fresh", "spoiled": "Spoiled"}
    stage = valid_stages.get(stage.lower(), "Moderately Fresh")
    result["freshness_stage"] = stage
    result.setdefault("item_name", item_type)
    result.setdefault("visual_indicators", [])
    result.setdefault("storage_tips", [])
    result.setdefault("confidence", "Unspecified")
    result.setdefault("caution", "Image-based appearance cannot establish microbiological safety.")
    if stage == "Spoiled":
        result["action"] = "Discard immediately"
    elif stage == "Moderately Fresh":
        result["action"] = "Consume within 24h"
    else:
        result["action"] = "Safe to consume"
    for key in ["visual_indicators", "storage_tips"]:
        if not isinstance(result[key], list):
            result[key] = [str(result[key])] if result[key] else []
    return result


def normalize_kitchen_data(data: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(data)
    try:
        result["compliance_score"] = max(0, min(100, int(float(result.get("compliance_score", 0)))))
    except (ValueError, TypeError):
        result["compliance_score"] = 0
    result.setdefault("checks", [])
    result.setdefault("summary", "Visual screening only; not an official compliance audit.")
    result.setdefault("confidence", "Unspecified")
    if not isinstance(result["checks"], list):
        result["checks"] = []
    for row in result["checks"]:
        if isinstance(row, dict):
            row.setdefault("indicator", "Unspecified indicator")
            row.setdefault("status", "Not visible")
            row.setdefault("evidence", "No evidence returned.")
            row.setdefault("severity", "Medium")
            row.setdefault("recommendation", "Review this item manually.")
    return result


def make_demo_recipe(pantry_text: str, diet: str, servings: int) -> Dict[str, Any]:
    pantry = [item.strip() for item in re.split(r"[,\n;]+", pantry_text) if item.strip()]
    picked = pantry[:8] or ["seasonal vegetables", "a protein of choice", "whole grains or a suitable alternative", "herbs and spices"]
    diet_lower = diet.lower()
    if "keto" in diet_lower:
        title, method = "Herby low-carb skillet", "Sauté low-carb vegetables and a suitable protein in a little oil; season with herbs and finish with lemon."
        base = ["low-carb vegetables", "eggs, tofu, fish, or paneer", "olive oil", "herbs", "lemon"]
    elif "vegan" in diet_lower:
        title, method = "Plant-powered pantry bowl", "Cook a legume or tofu with vegetables and spices; serve over a suitable grain if desired."
        base = ["beans, lentils, or tofu", "mixed vegetables", "garlic", "spices", "lemon"]
    elif "diabetic" in diet_lower:
        title, method = "Fiber-forward balanced bowl", "Build a plate around non-starchy vegetables, a protein source, and a measured high-fiber carbohydrate portion."
        base = ["non-starchy vegetables", "beans, lentils, tofu, eggs, or lean protein", "a measured whole-grain portion", "herbs", "lemon"]
    else:
        title, method = "Flexible one-pan pantry meal", "Cook the protein and vegetables thoroughly in a pan, season to taste, and serve with a suitable grain or starch."
        base = ["protein of choice", "mixed vegetables", "whole grain or potato", "olive oil", "herbs and spices"]
    ingredients = list(dict.fromkeys(picked + base))
    return {
        "title": title + " (demo recipe)",
        "description": "A flexible concept based on the pantry notes; check labels and adjust quantities to your needs.",
        "servings": servings, "prep_time_minutes": 10, "cook_time_minutes": 20,
        "difficulty": "Easy", "ingredients": [{"item": name, "amount": "Adjust to servings and appetite"} for name in ingredients],
        "steps": [
            "Wash hands and clean the work surface; prepare ingredients with separate utensils for raw animal products.",
            "Wash and chop vegetables. Drain and rinse canned beans where applicable.",
            method,
            "Cook animal proteins to an appropriate safe internal temperature; avoid tasting food before it is fully cooked.",
            "Taste, adjust seasoning, and serve promptly. Refrigerate leftovers within two hours (sooner in very hot conditions).",
        ],
        "substitutions": ["Use a similar protein that suits your diet.", "Swap vegetables based on season and availability.", "Use a gluten-free grain or omit grains if required."],
        "nutrition_estimate": "Not calculated in demo mode; nutrient content depends on exact ingredients and quantities.",
        "possible_allergens": ["Check the labels of packaged ingredients, sauces, and spice blends."],
        "safety_notes": ["Demo recipe only; verify dietary fit and allergen labels before cooking."],
        "mode": "Demo only",
    }


# Compact template banks for local 7-day planning. Recipe calories are not asserted as measured values;
# the displayed calorie numbers are per-meal allocations from the user's estimated target.
MEAL_BANK: Dict[str, Dict[str, List[Tuple[str, List[str]]]]] = {
    "Balanced": {
        "Breakfast": [
            ("Oats, berries and chia", ["rolled oats", "milk or fortified soy milk", "berries", "chia seeds"]),
            ("Egg and spinach toast", ["eggs", "whole-grain bread", "spinach", "tomatoes"]),
            ("Yogurt, fruit and nuts", ["plain yogurt", "seasonal fruit", "nuts", "seeds"]),
            ("Vegetable poha with peanuts", ["poha", "onion", "peas", "peanuts", "lemon"]),
            ("Besan vegetable chilla", ["gram flour", "onion", "tomato", "spinach", "yogurt"]),
            ("Banana-oat smoothie", ["banana", "oats", "milk or soy milk", "nut butter"]),
            ("Paneer and vegetable wrap", ["paneer", "whole-wheat wrap", "bell pepper", "onion"]),
        ],
        "Lunch": [
            ("Chickpea grain bowl", ["chickpeas", "brown rice or quinoa", "cucumber", "tomato", "lemon"]),
            ("Dal, rice and greens", ["lentils", "rice", "spinach", "tomato", "spices"]),
            ("Chicken and vegetable plate", ["chicken", "brown rice", "broccoli", "carrot"]),
            ("Paneer and mixed vegetables", ["paneer", "mixed vegetables", "whole-wheat roti", "yogurt"]),
            ("Bean and corn salad", ["beans", "corn", "lettuce", "tomato", "lime"]),
            ("Tofu veggie stir-fry", ["tofu", "broccoli", "bell pepper", "brown rice", "soy sauce"]),
            ("Lentil soup and toast", ["lentils", "carrots", "celery", "whole-grain bread"]),
        ],
        "Dinner": [
            ("Tofu and vegetable stir-fry", ["tofu", "mixed vegetables", "brown rice", "ginger"]),
            ("Baked fish and greens", ["fish", "green beans", "potato", "lemon"]),
            ("Rajma with measured rice", ["kidney beans", "rice", "tomato", "onion", "spices"]),
            ("Chicken and roasted vegetables", ["chicken", "sweet potato", "broccoli", "olive oil"]),
            ("Chickpea and spinach curry", ["chickpeas", "spinach", "tomato", "brown rice"]),
            ("Vegetable omelette and salad", ["eggs", "mushrooms", "spinach", "mixed salad"]),
            ("Paneer tikka and vegetables", ["paneer", "bell pepper", "onion", "yogurt", "spices"]),
        ],
        "Snack": [
            ("Apple and nuts", ["apple", "almonds"]), ("Hummus and carrots", ["chickpeas", "tahini", "carrots"]),
            ("Yogurt with seeds", ["plain yogurt", "pumpkin seeds"]), ("Roasted chana", ["roasted chickpeas"]),
            ("Fruit and peanut butter", ["seasonal fruit", "peanut butter"]), ("Cucumber and cheese", ["cucumber", "cheese"]),
            ("Boiled egg and tomato", ["eggs", "tomatoes"]),
        ],
    },
    "Vegan": {
        "Breakfast": [
            ("Overnight oats with soy milk", ["rolled oats", "soy milk", "berries", "chia seeds"]),
            ("Tofu scramble", ["tofu", "spinach", "tomato", "whole-grain toast"]),
            ("Peanut-banana oats", ["oats", "banana", "peanut butter", "fortified plant milk"]),
            ("Besan chilla", ["gram flour", "onion", "spinach", "tomato"]),
            ("Fruit and soy yogurt bowl", ["fortified soy yogurt", "seasonal fruit", "pumpkin seeds"]),
            ("Savory vegetable poha", ["poha", "peas", "onion", "peanuts", "lemon"]),
            ("Chia pudding", ["chia seeds", "fortified plant milk", "berries"]),
        ],
        "Lunch": [
            ("Chickpea salad bowl", ["chickpeas", "cucumber", "tomato", "greens", "lemon"]),
            ("Lentil dal and rice", ["lentils", "rice", "tomato", "spinach"]),
            ("Tofu brown-rice bowl", ["tofu", "brown rice", "broccoli", "soy sauce"]),
            ("Bean and corn wrap", ["black beans", "corn", "whole-wheat wrap", "lettuce"]),
            ("Quinoa edamame bowl", ["quinoa", "edamame", "cucumber", "carrot"]),
            ("Peanut vegetable noodles", ["whole-grain noodles", "peanut butter", "cabbage", "carrot"]),
            ("Chickpea spinach curry", ["chickpeas", "spinach", "tomato", "rice"]),
        ],
        "Dinner": [
            ("Tofu veggie stir-fry", ["tofu", "broccoli", "bell pepper", "brown rice"]),
            ("Three-bean chili", ["kidney beans", "black beans", "tomato", "bell pepper"]),
            ("Lentil and vegetable soup", ["lentils", "carrots", "celery", "whole-grain bread"]),
            ("Tempeh and sweet potato", ["tempeh", "sweet potato", "green beans"]),
            ("Chana masala", ["chickpeas", "tomato", "onion", "rice"]),
            ("Stuffed bell peppers", ["bell peppers", "quinoa", "beans", "tomato"]),
            ("Mushroom tofu lettuce cups", ["tofu", "mushrooms", "lettuce", "ginger"]),
        ],
        "Snack": [
            ("Apple and peanut butter", ["apple", "peanut butter"]), ("Hummus and carrots", ["chickpeas", "tahini", "carrots"]),
            ("Roasted chickpeas", ["chickpeas", "paprika"]), ("Soy yogurt and berries", ["soy yogurt", "berries"]),
            ("Edamame", ["edamame"]), ("Nuts and an orange", ["mixed nuts", "orange"]),
            ("Cucumber with hummus", ["cucumber", "hummus"]),
        ],
    },
    "Keto": {
        "Breakfast": [
            ("Spinach and egg scramble", ["eggs", "spinach", "olive oil"]),
            ("Paneer and pepper skillet", ["paneer", "bell pepper", "spices"]),
            ("Plain Greek yogurt with walnuts", ["plain Greek yogurt", "walnuts", "chia seeds"]),
            ("Tofu scramble with mushrooms", ["tofu", "mushrooms", "spinach"]),
            ("Avocado and boiled eggs", ["avocado", "eggs", "lemon"]),
            ("Cottage cheese and cucumber", ["cottage cheese", "cucumber", "herbs"]),
            ("Chia seed pudding, unsweetened", ["chia seeds", "unsweetened almond milk", "cinnamon"]),
        ],
        "Lunch": [
            ("Chicken avocado salad", ["chicken", "avocado", "leafy greens", "olive oil"]),
            ("Paneer and broccoli bowl", ["paneer", "broccoli", "olive oil"]),
            ("Tuna cucumber lettuce cups", ["tuna", "cucumber", "lettuce", "Greek yogurt"]),
            ("Egg salad greens bowl", ["eggs", "leafy greens", "mayonnaise", "celery"]),
            ("Tofu cabbage stir-fry", ["tofu", "cabbage", "mushrooms", "sesame oil"]),
            ("Grilled fish and asparagus", ["fish", "asparagus", "lemon", "olive oil"]),
            ("Halloumi and roasted zucchini", ["halloumi", "zucchini", "herbs"]),
        ],
        "Dinner": [
            ("Salmon with greens", ["salmon", "spinach", "green beans", "olive oil"]),
            ("Cauliflower paneer curry", ["paneer", "cauliflower", "cream or coconut cream", "spices"]),
            ("Chicken lettuce wraps", ["chicken", "lettuce", "mushrooms", "ginger"]),
            ("Tofu and bok choy skillet", ["tofu", "bok choy", "sesame oil"]),
            ("Zucchini noodles and pesto", ["zucchini", "pesto", "parmesan or alternative"]),
            ("Eggplant and minced-protein skillet", ["eggplant", "minced protein of choice", "tomato", "herbs"]),
            ("Mushroom omelette and salad", ["eggs", "mushrooms", "leafy greens"]),
        ],
        "Snack": [
            ("Walnuts and cheese", ["walnuts", "cheese"]), ("Celery and nut butter", ["celery", "nut butter"]),
            ("Boiled egg", ["eggs"]), ("Olives and cucumber", ["olives", "cucumber"]),
            ("Plain yogurt and chia", ["plain yogurt", "chia seeds"]), ("Avocado with lime", ["avocado", "lime"]),
            ("Roasted pumpkin seeds", ["pumpkin seeds"]),
        ],
    },
    "Diabetic-Friendly": {
        "Breakfast": [
            ("Oats with chia and berries", ["rolled oats", "chia seeds", "berries", "plain yogurt"]),
            ("Egg and vegetable omelette", ["eggs", "spinach", "tomato", "mushrooms"]),
            ("Besan chilla with yogurt", ["gram flour", "vegetables", "plain yogurt"]),
            ("Unsweetened yogurt and nuts", ["plain yogurt", "nuts", "seeds"]),
            ("Tofu scramble and vegetables", ["tofu", "spinach", "tomato"]),
            ("Moong dal chilla", ["moong dal", "ginger", "spinach", "yogurt"]),
            ("Cottage cheese and berries", ["cottage cheese", "berries", "seeds"]),
        ],
        "Lunch": [
            ("Lentil and vegetable bowl", ["lentils", "non-starchy vegetables", "measured brown rice"]),
            ("Chickpea cucumber salad", ["chickpeas", "cucumber", "tomato", "greens"]),
            ("Chicken and greens plate", ["chicken", "leafy greens", "broccoli", "measured whole grain"]),
            ("Tofu and vegetable bowl", ["tofu", "mixed vegetables", "measured quinoa"]),
            ("Rajma with salad and measured rice", ["kidney beans", "salad", "measured rice"]),
            ("Paneer and broccoli", ["paneer", "broccoli", "tomato", "whole-wheat roti"]),
            ("Bean soup and side salad", ["beans", "vegetable soup base", "leafy greens"]),
        ],
        "Dinner": [
            ("Fish and non-starchy vegetables", ["fish", "broccoli", "green beans", "lemon"]),
            ("Tofu stir-fry with cauliflower rice", ["tofu", "mixed vegetables", "cauliflower"]),
            ("Dal, vegetables and measured roti", ["lentils", "vegetables", "whole-wheat roti"]),
            ("Chicken and roasted vegetables", ["chicken", "cauliflower", "zucchini"]),
            ("Chickpea spinach stew", ["chickpeas", "spinach", "tomato"]),
            ("Egg curry with vegetables", ["eggs", "tomato", "spinach", "measured whole grain"]),
            ("Paneer tikka salad", ["paneer", "cucumber", "tomato", "leafy greens"]),
        ],
        "Snack": [
            ("Small apple with nuts", ["apple", "almonds"]), ("Plain yogurt", ["plain yogurt"]),
            ("Cucumber and hummus", ["cucumber", "hummus"]), ("Boiled egg", ["eggs"]),
            ("Roasted chana, measured portion", ["roasted chickpeas"]), ("Tomato and paneer bites", ["tomato", "paneer"]),
            ("Walnuts and seeds", ["walnuts", "pumpkin seeds"]),
        ],
    },
    "Vegetarian": {
        "Breakfast": [], "Lunch": [], "Dinner": [], "Snack": []
    },
}
# Vegetarian is made from the balanced bank with animal flesh removed through these swaps.
MEAL_BANK["Vegetarian"] = {
    "Breakfast": MEAL_BANK["Balanced"]["Breakfast"],
    "Lunch": [x for x in MEAL_BANK["Balanced"]["Lunch"] if not any(w in x[0].lower() for w in ["chicken"])],
    "Dinner": [x for x in MEAL_BANK["Balanced"]["Dinner"] if not any(w in x[0].lower() for w in ["chicken", "fish"])],
    "Snack": MEAL_BANK["Balanced"]["Snack"],
}


def calculate_targets(age: int, gender: str, weight_kg: float, height_cm: float, activity: str, goal: str) -> Dict[str, float]:
    """Estimate adult BMR/TDEE with Mifflin–St Jeor; not a clinical prescription."""
    if age < 18:
        raise ValueError("This planner is designed for adults aged 18 and older.")
    if weight_kg <= 0 or height_cm <= 0:
        raise ValueError("Weight and height must be positive.")
    sex_term = 5 if gender == "Male" else (-161 if gender == "Female" else -78)
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + sex_term
    activity_factors = {
        "Sedentary": 1.2,
        "Lightly active": 1.375,
        "Moderately active": 1.55,
        "Very active": 1.725,
        "Extra active": 1.9,
    }
    tdee = bmr * activity_factors.get(activity, 1.2)
    goal_multipliers = {"Fat Loss": 0.85, "Muscle Gain": 1.10, "Maintain Weight": 1.0}
    target = tdee * goal_multipliers.get(goal, 1.0)
    return {"bmr_kcal": round(bmr), "tdee_kcal": round(tdee), "target_kcal": round(target), "activity_factor": activity_factors.get(activity, 1.2)}


def build_weekly_plan(diet: str, target_kcal: int) -> Tuple[List[Dict[str, Any]], List[Tuple[str, int]]]:
    bank = MEAL_BANK.get(diet, MEAL_BANK["Balanced"])
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    allocation = [("Breakfast", 0.25), ("Lunch", 0.30), ("Dinner", 0.30), ("Snack", 0.15)]
    plan: List[Dict[str, Any]] = []
    groceries: Counter[str] = Counter()
    for i, day in enumerate(days):
        meals = []
        for meal_type, ratio in allocation:
            options = bank[meal_type]
            title, ingredients = options[i % len(options)]
            kcal = round(target_kcal * ratio)
            meals.append({"type": meal_type, "title": title, "target_kcal": kcal, "ingredients": ingredients})
            for item in ingredients:
                groceries[item] += 1
        # Force the displayed allocations to add up to target exactly despite rounding.
        delta = target_kcal - sum(m["target_kcal"] for m in meals)
        meals[-1]["target_kcal"] += delta
        plan.append({"day": day, "meals": meals, "daily_target_kcal": target_kcal})
    grocery_list = sorted(groceries.items(), key=lambda pair: (-pair[1], pair[0]))
    return plan, grocery_list

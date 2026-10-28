from __future__ import annotations

import json
import math
from io import BytesIO
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from foodverse_core import (
    MODEL_DEFAULTS,
    build_weekly_plan,
    calculate_targets,
    call_model,
    call_model_json,
    demo_food_analysis,
    demo_freshness,
    demo_kitchen_analysis,
    get_env_or_secret,
    make_demo_recipe,
    normalize_food_data,
    normalize_freshness_data,
    normalize_kitchen_data,
    video_contact_sheet,
)

st.set_page_config(
    page_title="Food Verse AI | Food Intelligence",
    page_icon="🥑",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: radial-gradient(circle at 8% 0%, rgba(87, 225, 173, .10), transparent 26%), #f6f8fc; }
      h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.025em; }
      .hero { padding: 1.35rem 1.5rem; border-radius: 22px; background: linear-gradient(120deg, #123a35 0%, #146b59 55%, #46b99b 100%); color: white; margin: .2rem 0 1.25rem; box-shadow: 0 12px 34px rgba(18,58,53,.16); }
      .hero h1 { color: white; margin: .15rem 0 .35rem; font-size: 2rem; }
      .hero p { color: #e5fff5; margin: 0; font-size: 1rem; }
      .eyebrow { text-transform: uppercase; letter-spacing: .13em; font-size: .72rem; font-weight: 700; color: #b8f5df; }
      .pill { display: inline-block; padding: .28rem .65rem; border-radius: 999px; background: #e2f8ee; color: #155d49; font-size: .78rem; font-weight: 700; }
      .muted { color: #657084; font-size: .9rem; }
      .small-label { color: #68778d; text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; font-weight: 700; }
      .score-wrap { padding: 1rem 1.1rem; background: white; border: 1px solid #e2e9f0; border-radius: 18px; }
      .score-num { font-family: 'Space Grotesk', sans-serif; font-size: 2.35rem; font-weight: 700; color: #126b57; line-height: 1.05; }
      .status-good { color: #126b57; font-weight: 700; }
      .status-warn { color: #a35c00; font-weight: 700; }
      .status-bad { color: #b42318; font-weight: 700; }
      div[data-testid="stMetric"] { border: 1px solid #e2e9f0; background: white; border-radius: 16px; padding: .8rem 1rem; }
      div[data-testid="stExpander"] { border-radius: 14px; border-color: #e2e9f0; }
      .stButton > button, .stDownloadButton > button { border-radius: 12px; font-weight: 700; border: 0; }
      .stButton > button[kind="primary"] { background: #137b63; }
      section[data-testid="stSidebar"] { background: #102f31; }
      section[data-testid="stSidebar"] * { color: #f0fbf7; }
      section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] textarea { color: #152536; }
      section[data-testid="stSidebar"] [data-testid="stRadio"] label { background: rgba(255,255,255,.045); padding: .55rem .7rem; border-radius: 10px; margin-bottom: .2rem; }
      hr { border-color: #e1e8f0; }
      @media (max-width: 640px) { .hero { padding: 1rem; } .hero h1 { font-size: 1.5rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

NAV_ITEMS = [
    "Overview",
    "Food recognition & nutrition",
    "Freshness & spoilage",
    "Kitchen hygiene audit",
    "Smart recipe assistant",
    "Personalized meal planner",
]


def get_runtime_config() -> Dict[str, Any]:
    st.sidebar.markdown("## 🥑 Food Verse AI")
    st.sidebar.caption("Multimodal food intelligence · Capstone 03")
    page = st.sidebar.radio("Navigate", NAV_ITEMS, label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### AI configuration")
    provider = st.sidebar.selectbox("Analysis provider", ["Demo only", "Google Gemini", "Groq"], index=0)
    api_key = ""
    model = ""
    if provider != "Demo only":
        api_key = st.sidebar.text_input(
            "API key", value=get_env_or_secret(provider), type="password", key=f"api_key_{provider}",
            help="Read from an environment variable / Streamlit secret when available. Keys are not written to project files.",
        )
        model = st.sidebar.text_input("Model ID", value=MODEL_DEFAULTS[provider], key=f"model_{provider}")
        consent = st.sidebar.checkbox(
            "I consent to sending the selected image/video or prompt to this AI provider for analysis.",
            value=False,
            help="Uploads may contain personal or workplace information. Do not submit sensitive images or information you are not authorized to share.",
        )
    else:
        consent = False
        st.sidebar.info("Demo mode runs locally with illustrative sample outputs. Uploaded media is not analyzed by a vision model.")
    use_live = provider != "Demo only" and bool(api_key.strip()) and consent
    if provider != "Demo only" and not api_key:
        st.sidebar.caption("Add a key to enable live AI analysis.")
    elif provider != "Demo only" and api_key and not consent:
        st.sidebar.caption("Check the upload consent box to enable live analysis; otherwise the app uses demo samples.")
    st.sidebar.markdown("---")
    st.sidebar.caption("Privacy: API keys stay in session input / configured secrets. No database is used.")
    return {"page": page, "provider": provider, "api_key": api_key.strip(), "model": model, "use_live": use_live}


def hero(title: str, subtitle: str, tag: str = "FOOD INTELLIGENCE") -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{tag}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def list_items(values: Any, empty_text: str = "None reported") -> None:
    if isinstance(values, str):
        values = [values]
    if not values:
        st.write(empty_text)
        return
    for value in values:
        if isinstance(value, dict):
            st.write("• " + str(value.get("name", value)))
        else:
            st.write("• " + str(value))


def run_live_json(cfg: Dict[str, Any], prompt: str, image_bytes: bytes | None = None, mime_type: str = "image/jpeg") -> tuple[Dict[str, Any] | None, str | None]:
    if not cfg["use_live"]:
        return None, None
    try:
        result = call_model_json(cfg["provider"], cfg["api_key"], cfg["model"], prompt, image_bytes, mime_type)
        return result, None
    except Exception as exc:
        # Avoid printing exception strings that could contain request details.
        return None, f"Live analysis failed ({type(exc).__name__}). A labelled demo fallback is shown instead. Check provider, model ID, quota, and network access."


def run_live_text(cfg: Dict[str, Any], prompt: str) -> tuple[str | None, str | None]:
    if not cfg["use_live"]:
        return None, None
    try:
        return call_model(cfg["provider"], cfg["api_key"], cfg["model"], prompt), None
    except Exception as exc:
        return None, f"Live generation failed ({type(exc).__name__}). A labelled demo fallback is shown instead. Check provider, model ID, quota, and network access."


def export_json_button(data: Dict[str, Any], filename: str, label: str = "Download JSON report") -> None:
    st.download_button(label, data=json.dumps(data, indent=2, ensure_ascii=False), file_name=filename, mime="application/json")


def badge_for_score(score: int, label: str = "Health score") -> None:
    if score >= 80:
        cls, desc = "status-good", "Strong"
    elif score >= 60:
        cls, desc = "status-warn", "Moderate"
    else:
        cls, desc = "status-bad", "Needs attention"
    st.markdown(
        f'<div class="score-wrap"><div class="small-label">{label}</div><div class="score-num">{score}<span style="font-size:1rem;color:#718096"> / 100</span></div><div class="{cls}">{desc}</div></div>',
        unsafe_allow_html=True,
    )


def render_overview(cfg: Dict[str, Any]) -> None:
    hero("Food Verse AI", "See food more clearly. Make better kitchen decisions. Plan with purpose.", "SKILL WALLET · CAPSTONE PROJECT 3")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🍽️ Understand")
        st.write("Recognize dishes and turn a meal photo into an estimated nutrition audit, ingredient list, and allergen prompts.")
    with col2:
        st.markdown("### 🥬 Protect")
        st.write("Screen visible freshness signals and kitchen hygiene risks—without confusing visual screening with lab testing or certification.")
    with col3:
        st.markdown("### 🧑‍🍳 Personalize")
        st.write("Create pantry-based recipe ideas and a weekly meal framework from adult energy estimates and dietary preferences.")
    st.markdown("---")
    st.markdown("### Your workspace")
    tiles = st.columns(5)
    features = [
        ("01", "Nutrition audit", "Dish, portion, macros, health score"),
        ("02", "Freshness check", "Appearance cues and storage"),
        ("03", "Kitchen audit", "Visible hygiene controls"),
        ("04", "Recipe coach", "Use what is in your pantry"),
        ("05", "Meal planner", "BMR, TDEE and grocery list"),
    ]
    for col, (num, title, desc) in zip(tiles, features):
        with col:
            st.markdown(f"<div class='score-wrap'><div class='small-label'>PILLAR {num}</div><h4>{title}</h4><div class='muted'>{desc}</div></div>", unsafe_allow_html=True)
    st.markdown("### Operating mode")
    if cfg["use_live"]:
        st.success(f"Live mode enabled · {cfg['provider']} · model `{cfg['model']}`")
    else:
        st.info("Demo mode is active. Sample results are illustrative and are not inferred from media uploads. Configure a provider and consent in the sidebar to use live AI.")
    st.caption("Food Verse AI is a decision-support demo. It does not replace a dietitian, food-safety professional, or accredited inspection.")


def render_food_recognition(cfg: Dict[str, Any]) -> None:
    hero("Food recognition & nutrition audit", "Estimate dish identity, serving size, macros, and possible allergen exposure.", "EPIC 2 · FOOD RECOGNITION")
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        uploaded = st.file_uploader("Upload a meal photo", type=["jpg", "jpeg", "png", "webp"], key="food_image")
        if uploaded:
            try:
                st.image(uploaded, caption="Uploaded meal image", use_container_width=True)
            except Exception:
                st.warning("The file could not be previewed. Try a JPG or PNG image.")
        sample = st.selectbox("Demo sample (used only when live analysis is off or fails)", list(["Balanced grain bowl", "Vegetable pasta", "Dal and rice", "Paneer veggie wrap", "Chickpea salad"]))
        st.caption("Portion and nutrients are estimates. For packaged food, use the nutrition label when available.")
        analyze = st.button("Analyze meal", type="primary", use_container_width=True)
    with right:
        st.markdown("#### What you'll get")
        st.write("Dish and cuisine guess, ingredients, estimated portion, Calories, protein, carbohydrates, fats, fiber, possible allergens, and a composite score.")
        st.warning("Image-based allergen detection can miss hidden ingredients and cross-contact. Never use it to confirm a food is allergy-safe.")
    if analyze:
        raw = None
        error = None
        if uploaded and cfg["use_live"]:
            mime = uploaded.type or "image/jpeg"
            prompt = """You are a careful food-science assistant. Inspect the meal photo and return ONLY a valid JSON object with this schema: {\"dish_name\": string, \"cuisine\": string, \"portion_estimate\": string, \"ingredients\": [{\"name\": string, \"estimated_grams\": number}], \"nutrition\": {\"calories_kcal\": number, \"protein_g\": number, \"carbohydrates_g\": number, \"fat_g\": number, \"fiber_g\": number}, \"possible_allergens\": [string], \"health_score\": integer 1-100, \"health_score_notes\": string, \"recommendations\": [string], \"confidence\": \"low\"|\"medium\"|\"high\", \"assumptions\": [string]}. Estimate only what can reasonably be inferred; state uncertainty and portion assumptions. Do not claim exact nutrients from pixels. Include common hidden allergens only as possibilities, not confirmed ingredients. The health score is a non-clinical heuristic based on estimated balance, variety, fiber, and likely processing/sodium where visible."""
            raw, error = run_live_json(cfg, prompt, uploaded.getvalue(), mime)
        if error:
            st.warning(error)
        is_demo = raw is None
        if raw is None:
            data = demo_food_analysis(sample)
            if uploaded:
                data["assumptions"] = list(data.get("assumptions", [])) + ["An image was uploaded, but demo mode does not interpret uploaded pixels."]
        else:
            data = raw
        data = normalize_food_data(data)
        with st.container(border=True):
            col_a, col_b = st.columns([1.4, 1])
            with col_a:
                st.markdown(f"## {data['dish_name']}")
                st.caption(f"Cuisine: {data['cuisine']} · Portion: {data['portion_estimate']}")
                st.markdown("**Ingredient estimates**")
                if data["ingredients"]:
                    ingredient_rows = []
                    for item in data["ingredients"]:
                        if isinstance(item, dict):
                            ingredient_rows.append({"Ingredient": item.get("name", "Unknown"), "Estimated grams": item.get("estimated_grams", "—")})
                        else:
                            ingredient_rows.append({"Ingredient": str(item), "Estimated grams": "—"})
                    st.dataframe(pd.DataFrame(ingredient_rows), hide_index=True, use_container_width=True)
                else:
                    st.write("No ingredients were returned confidently.")
            with col_b:
                badge_for_score(data["health_score"])
                st.caption(data.get("health_score_notes", ""))
                st.markdown(f"**Analysis mode:** {'DEMO SAMPLE' if is_demo else 'LIVE AI ESTIMATE'}")
                st.caption(f"Confidence reported: {data.get('confidence', 'Unspecified')}")
        st.markdown("### Estimated nutrition")
        nutrition = data["nutrition"]
        metrics = st.columns(5)
        nutrient_items = [("Calories", f"{nutrition['calories_kcal']:.0f}", "kcal"), ("Protein", f"{nutrition['protein_g']:.1f}", "g"), ("Carbohydrates", f"{nutrition['carbohydrates_g']:.1f}", "g"), ("Fat", f"{nutrition['fat_g']:.1f}", "g"), ("Fiber", f"{nutrition['fiber_g']:.1f}", "g")]
        for col, (name, val, unit) in zip(metrics, nutrient_items):
            with col:
                st.metric(name, val, unit)
        chart_left, chart_right = st.columns(2)
        macros_g = {"Protein": nutrition["protein_g"], "Carbohydrates": nutrition["carbohydrates_g"], "Fat": nutrition["fat_g"]}
        with chart_left:
            st.markdown("#### Macronutrients by grams")
            macro_df = pd.DataFrame({"Macronutrient": list(macros_g.keys()), "Grams": list(macros_g.values())})
            fig = px.bar(macro_df, x="Macronutrient", y="Grams", color="Macronutrient", color_discrete_sequence=["#167d64", "#e5a144", "#6f77d8"])
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10), height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with chart_right:
            st.markdown("#### Approximate macro energy share")
            energy_values = {"Protein": nutrition["protein_g"] * 4, "Carbohydrates": nutrition["carbohydrates_g"] * 4, "Fat": nutrition["fat_g"] * 9}
            energy_df = pd.DataFrame({"Macronutrient": list(energy_values.keys()), "Estimated kcal": list(energy_values.values())})
            fig = px.pie(energy_df, names="Macronutrient", values="Estimated kcal", hole=.58, color="Macronutrient", color_discrete_map={"Protein": "#167d64", "Carbohydrates": "#e5a144", "Fat": "#6f77d8"})
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=300, paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-.1))
            st.plotly_chart(fig, use_container_width=True)
        lower1, lower2 = st.columns(2)
        with lower1:
            st.markdown("#### Possible allergen alerts")
            list_items(data.get("possible_allergens", []), "No possible allergens returned — not proof of absence.")
        with lower2:
            st.markdown("#### Practical improvements")
            list_items(data.get("recommendations", []), "No recommendations returned.")
        if data.get("assumptions"):
            with st.expander("Assumptions and limitations"):
                list_items(data["assumptions"])
        export_json_button(data, "foodverse-nutrition-audit.json")


def render_freshness(cfg: Dict[str, Any]) -> None:
    hero("Freshness & spoilage diagnostic", "Screen visible quality cues and get conservative storage guidance.", "EPIC 3 · FRESHNESS")
    c1, c2 = st.columns([1, 1.1], gap="large")
    with c1:
        uploaded = st.file_uploader("Upload a perishable food photo", type=["jpg", "jpeg", "png", "webp"], key="fresh_image")
        if uploaded:
            st.image(uploaded, caption="Uploaded perishable item", use_container_width=True)
        item_type = st.selectbox("Food category", ["Fruit", "Vegetable", "Raw meat / poultry", "Fish / seafood", "Dairy", "Cooked leftovers", "Other perishable food"])
        scenario = st.selectbox("Demo scenario (used only without successful live AI)", ["Bright, firm produce", "Slightly wilted or wrinkled item", "Visible mold / sliminess"])
        go = st.button("Run freshness screen", type="primary", use_container_width=True)
    with c2:
        st.markdown("#### Visual indicators checked")
        st.write("Surface texture, discoloration, wrinkling, visible mold, and other apparent changes. Results depend on lighting, image quality, and the item shown.")
        st.error("Food safety limitation: a photo cannot detect most pathogens, toxins, unsafe time-temperature history, or all spoilage. Do not taste suspect food. Follow labels and local food-safety guidance.")
    if go:
        raw, error = None, None
        if uploaded and cfg["use_live"]:
            prompt = f"""You are a conservative food-quality screening assistant. Inspect the image of this perishable category: {item_type}. Return ONLY JSON: {{\"item_name\": string, \"freshness_stage\": \"Fresh\"|\"Moderately Fresh\"|\"Spoiled\", \"visual_indicators\": [string], \"storage_tips\": [string], \"confidence\": \"low\"|\"medium\"|\"high\", \"caution\": string}}. Describe only visible cues. Never claim a photo can prove the item safe or detect pathogens. If visible mold on soft/high-moisture food, slime, or strong discoloration indicates spoilage, classify conservatively. Do not recommend tasting or smelling closely. Include advice to follow use-by dates and time-temperature guidance. The application will map stages to actions: Fresh = Safe to consume (visual-only label), Moderately Fresh = Consume within 24h, Spoiled = Discard immediately."""
            raw, error = run_live_json(cfg, prompt, uploaded.getvalue(), uploaded.type or "image/jpeg")
        if error:
            st.warning(error)
        data = raw if raw is not None else demo_freshness(scenario, item_type)
        if raw is None and uploaded:
            data["caution"] += " Uploaded pixels were not interpreted in demo mode."
        data = normalize_freshness_data(data, item_type)
        is_demo = raw is None
        stage = data["freshness_stage"]
        if stage == "Fresh":
            st.success(f"Visual category: {stage} · Action label: {data['action']}")
        elif stage == "Moderately Fresh":
            st.warning(f"Visual category: {stage} · Action label: {data['action']}")
        else:
            st.error(f"Visual category: {stage} · Action label: {data['action']}")
        st.caption(f"Mode: {'DEMO SCENARIO' if is_demo else 'LIVE AI VISUAL SCREEN'} · Confidence: {data.get('confidence', 'Unspecified')}")
        st.markdown("### Findings")
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**Visible indicators / scenario notes**")
            list_items(data.get("visual_indicators", []))
        with c_right:
            st.markdown("**Storage and handling advice**")
            list_items(data.get("storage_tips", []))
        st.info(data.get("caution", "Image-based appearance cannot establish microbiological safety."))
        if item_type in ["Raw meat / poultry", "Fish / seafood", "Dairy", "Cooked leftovers"]:
            st.warning("For high-risk perishables, a fresh appearance is not a safety guarantee. If time-temperature history is unknown, packaging is swollen/leaking, or use-by guidance is exceeded, do not rely on this visual score.")
        export_json_button(data, "foodverse-freshness-screen.json")


def render_kitchen(cfg: Dict[str, Any]) -> None:
    hero("Kitchen hygiene & compliance audit", "Review visible hygiene controls and prioritize corrective actions.", "EPIC 4 · KITCHEN SAFETY")
    col1, col2 = st.columns([1, 1.1], gap="large")
    with col1:
        media = st.file_uploader("Upload a kitchen photo or short video", type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi"], key="kitchen_media")
        is_video = bool(media and media.name.lower().endswith((".mp4", ".mov", ".avi")))
        image_bytes = None
        mime = "image/png"
        if media:
            if is_video:
                st.video(media)
                st.caption("Live mode samples up to four frames into a contact sheet for visual review.")
            else:
                st.image(media, caption="Uploaded kitchen image", use_container_width=True)
        scenario = st.selectbox("Demo scenario (used only without successful live AI)", ["Strong visible controls", "Mixed controls / possible raw-produce proximity", "Multiple visible hygiene concerns"])
        go = st.button("Audit kitchen", type="primary", use_container_width=True)
    with col2:
        st.markdown("#### Review checklist")
        st.write("Hair restraint / headwear, glove and hand-hygiene cues, visible sanitization state, and separation of raw animal products from ready-to-eat foods.")
        st.warning("This is a visual screening—not HACCP verification, legal certification, or a substitute for observing staff practices and temperature records. Still images cannot prove sanitization or handwashing.")
    if go:
        raw, error = None, None
        if media and cfg["use_live"]:
            try:
                if is_video:
                    image_bytes = video_contact_sheet(media.getvalue())
                else:
                    image_bytes = media.getvalue()
                    mime = media.type or "image/jpeg"
                prompt = """Act as a careful commercial-kitchen visual screening assistant. Inspect the image/contact sheet. Return ONLY JSON: {\"compliance_score\": integer 0-100, \"checks\": [{\"indicator\": string, \"status\": \"Pass\"|\"Concern\"|\"Not visible\", \"evidence\": string, \"severity\": \"Low\"|\"Medium\"|\"High\", \"recommendation\": string}], \"summary\": string, \"confidence\": \"low\"|\"medium\"|\"high\"}. Include separate checks for hair restraint/headwear, glove use/hand hygiene, surface sanitization, and raw/ready-to-eat separation. Use Not visible when evidence is absent; do not infer behavior from a still. Flag raw poultry or meat near ready-to-eat produce as a potential cross-contamination hazard. The score is a heuristic visual screen, not an official inspection; do not claim pathogens can be detected."""
                raw, error = run_live_json(cfg, prompt, image_bytes, mime)
            except Exception as exc:
                error = f"Media processing failed ({type(exc).__name__}). Try a shorter MP4 or a still image. Demo fallback will be shown."
        if error:
            st.warning(error)
        data = raw if raw is not None else demo_kitchen_analysis(scenario)
        if raw is None and media:
            data["summary"] += " Uploaded media was not interpreted in demo mode."
        data = normalize_kitchen_data(data)
        is_demo = raw is None
        c_score, c_summary = st.columns([1, 2])
        with c_score:
            st.markdown(f'<div class="score-wrap"><div class="small-label">Visual compliance score</div><div class="score-num">{data["compliance_score"]}<span style="font-size:1rem;color:#718096">%</span></div><div class="muted">{ "Demo checklist" if is_demo else "AI visual heuristic" }</div></div>', unsafe_allow_html=True)
        with c_summary:
            st.markdown("#### Assessment summary")
            st.write(data.get("summary", ""))
            st.caption(f"Confidence reported: {data.get('confidence', 'Unspecified')}")
        st.markdown("### Itemized checks")
        checks = data.get("checks", [])
        if checks:
            df = pd.DataFrame([{ "Indicator": x.get("indicator"), "Status": x.get("status"), "Severity": x.get("severity"), "Evidence": x.get("evidence"), "Recommended action": x.get("recommendation") } for x in checks if isinstance(x, dict)])
            st.dataframe(df, hide_index=True, use_container_width=True, column_config={"Evidence": st.column_config.TextColumn(width="large"), "Recommended action": st.column_config.TextColumn(width="large")})
            concerns = [x for x in checks if isinstance(x, dict) and str(x.get("status", "")).lower() == "concern"]
            if concerns:
                st.markdown("#### Priority corrective actions")
                for item in concerns:
                    st.write(f"**{item.get('severity', 'Medium')} · {item.get('indicator', 'Review')}** — {item.get('recommendation', 'Review manually.')}")
            else:
                st.success("No explicit concerns were returned. Items marked Not visible still require manual verification.")
        else:
            st.info("No itemized checks were returned. Capture a clearer image and try again.")
        export_json_button(data, "foodverse-kitchen-audit.json")


def render_recipe(cfg: Dict[str, Any]) -> None:
    hero("Smart kitchen recipe assistant", "Turn pantry ingredients into flexible, practical meals.", "EPIC 4 · RECIPE COACH")
    with st.form("recipe_form"):
        pantry = st.text_area("Pantry inventory", placeholder="e.g. chickpeas, tomatoes, spinach, rice, onion, yogurt, lemon", height=110, help="Separate ingredients with commas or new lines.")
        c1, c2, c3 = st.columns(3)
        with c1:
            diet = st.selectbox("Diet preference", ["Balanced", "Vegan", "Keto", "Diabetic-Friendly", "Vegetarian"])
        with c2:
            servings = st.number_input("Servings", min_value=1, max_value=12, value=2, step=1)
        with c3:
            max_time = st.selectbox("Preferred total time", ["Any", "15 minutes", "30 minutes", "45 minutes"])
        avoid = st.text_input("Allergies / ingredients to avoid (optional)", placeholder="e.g. peanuts, shellfish")
        submitted = st.form_submit_button("Create recipe", type="primary", use_container_width=True)
    if submitted:
        if not pantry.strip():
            st.error("Add at least one pantry ingredient to personalize the recipe. A generic demo recipe will still be shown below.")
        prompt = f"""You are a practical nutrition-aware home chef. Create ONE original recipe as JSON only with schema: {{\"title\": string, \"description\": string, \"servings\": integer, \"prep_time_minutes\": integer, \"cook_time_minutes\": integer, \"difficulty\": \"Easy\"|\"Medium\"|\"Advanced\", \"ingredients\": [{{\"item\": string, \"amount\": string}}], \"steps\": [string], \"substitutions\": [string], \"nutrition_estimate\": string, \"possible_allergens\": [string], \"safety_notes\": [string]}}. Pantry: {pantry or 'common pantry staples'}. Diet preference: {diet}. Servings: {servings}. Time preference: {max_time}. Avoid/allergies supplied by user: {avoid or 'none stated'}. Prioritize pantry use and budget-friendly ingredients. Respect the declared diet. If a restriction cannot safely be guaranteed, say so. Include food-safety steps, no invented medical claims, realistic ingredient amounts and step-by-step cooking instructions. Do not state exact nutrient values unless clearly estimated."""
        text, error = run_live_text(cfg, prompt)
        if error:
            st.warning(error)
        data = None
        if text:
            try:
                from foodverse_core import parse_json_response
                data = parse_json_response(text)
                data["mode"] = "Live AI"
            except Exception:
                st.warning("The live model response could not be parsed as recipe JSON; showing a demo recipe instead.")
        if data is None:
            data = make_demo_recipe(pantry, diet, int(servings))
            if cfg["use_live"]:
                data["mode"] = "Demo fallback"
        if avoid.strip():
            data["user_allergen_note"] = avoid.strip()
        st.markdown("---")
        header_l, header_r = st.columns([1.6, 1])
        with header_l:
            st.markdown(f"## {data.get('title', 'Pantry recipe')}")
            st.write(data.get("description", ""))
        with header_r:
            st.metric("Total cook time", f"{int(data.get('prep_time_minutes', 0) or 0) + int(data.get('cook_time_minutes', 0) or 0)} min")
            st.metric("Difficulty", data.get("difficulty", "Easy"))
            st.caption(f"Mode: {data.get('mode', 'Live AI' if cfg['use_live'] else 'Demo only')}")
        c_ing, c_steps = st.columns([1, 1.4])
        with c_ing:
            st.markdown("### Ingredients")
            ingredients = data.get("ingredients", [])
            if ingredients:
                for item in ingredients:
                    if isinstance(item, dict):
                        st.write(f"☐ {item.get('amount', '')} {item.get('item', '')}".strip())
                    else:
                        st.write(f"☐ {item}")
            else:
                st.write("Ingredients were not returned; review the recipe before use.")
        with c_steps:
            st.markdown("### Method")
            steps = data.get("steps", [])
            for i, step in enumerate(steps, 1):
                st.markdown(f"**{i}.** {step}")
        c_sub, c_safety = st.columns(2)
        with c_sub:
            st.markdown("### Ingredient substitutions")
            list_items(data.get("substitutions", []))
        with c_safety:
            st.markdown("### Allergens & safety")
            if data.get("user_allergen_note"):
                st.warning(f"User-supplied avoid list: {data['user_allergen_note']}. Verify every ingredient and package label; recipes cannot guarantee absence of cross-contact.")
            list_items(data.get("possible_allergens", []), "No allergens were returned; verify labels yourself.")
            list_items(data.get("safety_notes", []), "Follow standard food-safety practices.")
        st.info(f"Nutrition: {data.get('nutrition_estimate', 'Not independently verified.')} Verify dietary restrictions and all ingredient labels before cooking.")
        export_json_button(data, "foodverse-recipe.json")


def render_planner() -> None:
    hero("Personalized meal planner", "Estimate adult energy needs and build a practical 7-day menu framework.", "EPIC 5 · TDEE PLANNER")
    st.info("Uses the Mifflin–St Jeor equation and a standard activity multiplier. Results are estimates, not medical advice. The meal allocations are calorie targets, not laboratory-verified recipe nutrition.")
    with st.form("planner_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=30, step=1)
            gender = st.selectbox("Sex term for BMR equation", ["Female", "Male", "Prefer not to say / midpoint"])
            weight = st.number_input("Weight (kg)", min_value=25.0, max_value=350.0, value=70.0, step=0.5)
        with c2:
            height = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=170.0, step=0.5)
            activity = st.selectbox("Activity level", ["Sedentary", "Lightly active", "Moderately active", "Very active", "Extra active"], index=2)
            goal = st.selectbox("Fitness goal", ["Fat Loss", "Muscle Gain", "Maintain Weight"])
        with c3:
            diet = st.selectbox("Dietary pattern", ["Balanced", "Vegan", "Keto", "Diabetic-Friendly", "Vegetarian"])
            constraints = st.text_input("Other dietary constraints (optional)", placeholder="e.g. lactose-free, no peanuts")
            cooking = st.selectbox("Cooking style", ["Quick and simple", "Batch cooking", "Variety-focused"])
        submitted = st.form_submit_button("Calculate targets & build 7-day plan", type="primary", use_container_width=True)
    if submitted:
        try:
            sex_term = "Female" if gender == "Female" else ("Male" if gender == "Male" else "Prefer not to say")
            targets = calculate_targets(int(age), sex_term, float(weight), float(height), activity, goal)
            plan, grocery = build_weekly_plan(diet, int(targets["target_kcal"]))
        except Exception as exc:
            st.error(f"Could not calculate a plan: {exc}")
            return
        st.markdown("---")
        st.markdown("### Daily energy estimate")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("BMR", f"{targets['bmr_kcal']:,} kcal/day", help="Estimated energy expenditure at rest.")
        with c2:
            st.metric("TDEE", f"{targets['tdee_kcal']:,} kcal/day", help="Estimated daily energy expenditure including activity.")
        with c3:
            st.metric("Goal target", f"{targets['target_kcal']:,} kcal/day", help="Fat loss uses a modest 15% reduction; muscle gain uses a 10% increase; maintenance uses estimated TDEE.")
        if targets["target_kcal"] < 1200:
            st.warning("This estimate is unusually low for many adults. Do not treat it as a prescription; seek individualized advice from a registered dietitian or clinician before following a calorie-restricted plan.")
        if goal == "Fat Loss":
            st.caption("Goal adjustment: approximately 15% below estimated TDEE. Weight-loss needs vary, and this calculation does not account for every health factor.")
        elif goal == "Muscle Gain":
            st.caption("Goal adjustment: approximately 10% above estimated TDEE. Pair with progressive resistance training and adequate recovery.")
        else:
            st.caption("Goal adjustment: estimated TDEE for weight maintenance.")
        st.markdown("### 7-day schedule")
        for day in plan:
            with st.expander(f"{day['day']} · target {day['daily_target_kcal']:,} kcal", expanded=(day["day"] == "Monday")):
                df = pd.DataFrame([{"Meal": m["type"], "Menu idea": m["title"], "Calorie allocation target": f"{m['target_kcal']:,} kcal"} for m in day["meals"]])
                st.dataframe(df, hide_index=True, use_container_width=True)
                st.caption("Ingredients: " + "; ".join(f"{m['type']}: {', '.join(m['ingredients'])}" for m in day["meals"]))
        st.markdown("### Grocery list")
        grocery_text = "FOOD VERSE AI — WEEKLY GROCERY CHECKLIST\nDiet pattern: " + diet + "\nOther constraints: " + (constraints or "None stated") + "\nCooking style: " + cooking + "\n\n" + "\n".join(f"[ ] {item} (appears in {count} menu item(s))" for item, count in grocery)
        gc1, gc2 = st.columns([1.4, 1])
        with gc1:
            st.text_area("Checklist (copy or edit)", value=grocery_text, height=270)
        with gc2:
            st.markdown("#### Planning notes")
            st.write("• Portion sizes should be adjusted to the calorie target and hunger/satiety cues.")
            st.write("• If you have diabetes, kidney disease, are pregnant, take glucose-lowering medication, or have a history of disordered eating, seek personalized clinical guidance.")
            st.write("• Keto is restrictive and may not be appropriate for everyone; do not use it as a substitute for medical care.")
            st.write("• Review all labels for allergens and other dietary constraints.")
            st.download_button("Download grocery checklist", grocery_text, file_name="foodverse-grocery-checklist.txt", mime="text/plain", use_container_width=True)
        export_json_button({"targets": targets, "diet": diet, "goal": goal, "activity": activity, "plan": plan, "grocery_list": grocery}, "foodverse-7-day-meal-plan.json")


cfg = get_runtime_config()
if cfg["page"] == "Overview":
    render_overview(cfg)
elif cfg["page"] == "Food recognition & nutrition":
    render_food_recognition(cfg)
elif cfg["page"] == "Freshness & spoilage":
    render_freshness(cfg)
elif cfg["page"] == "Kitchen hygiene audit":
    render_kitchen(cfg)
elif cfg["page"] == "Smart recipe assistant":
    render_recipe(cfg)
elif cfg["page"] == "Personalized meal planner":
    render_planner()

st.markdown("---")
st.caption("Food Verse AI · Skill Wallet Capstone Project 3 · Streamlit · No external database")

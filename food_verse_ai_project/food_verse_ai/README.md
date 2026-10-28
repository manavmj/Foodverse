# Food Verse AI — Multimodal Food Intelligence Agent

**Skill Wallet Capstone Project 3** · Generative AI · Nutrition, freshness, kitchen safety, recipes, and meal planning

Food Verse AI is a single-app Streamlit project with five functional pillars: food recognition and nutrition audit, freshness screening, kitchen hygiene review, pantry-based recipe generation, and a personalized 7-day meal framework. It runs without a database and has a local demonstration mode for demos and coursework.

## Features

- **Optional live multimodal AI:** Gemini or Groq for food/kitchen images and recipe generation. Model IDs can be changed in the sidebar.
- **Demo fallback:** runs without API credentials; sample food, freshness, and kitchen results are explicitly labeled as illustrative and are not inferred from uploaded images.
- **Nutrition audit:** dish/cuisine estimate, ingredients and portion assumptions, macros, possible allergens, heuristic health score, interactive Plotly charts, JSON export.
- **Freshness screen:** visible cues, freshness stage, handling notes, and conservative storage reminders.
- **Kitchen screen:** photo or video upload; video mode samples up to four frames and submits a contact sheet when live AI is enabled; itemized hygiene observations and corrective actions.
- **Recipe assistant:** pantry and diet preference input, recipe steps, substitutions, safety notes, and JSON export.
- **Meal planner:** adult BMR/TDEE estimate using Mifflin–St Jeor, modest goal adjustment, a rotating 7-day meal framework, and downloadable grocery checklist/JSON.
- **Deployable without a database:** local calculations and template menus are included; no external data store is required.

## Run locally

Python 3.9+ is the stated project baseline. Use Python 3.10 or newer where possible for current dependency compatibility.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). No API key is needed for demo mode.

## Enable live AI

1. Choose **Google Gemini** or **Groq** in the sidebar.
2. Enter a key in the password field or supply it as a Streamlit secret / environment variable.
3. Confirm the consent checkbox explaining that uploads/prompts are sent to the selected provider.
4. Keep the default model ID or replace it with a model available to your provider account.

Environment variable names:

- Gemini: `GEMINI_API_KEY`
- Groq: `GROQ_API_KEY`

For local Streamlit secrets, create `.streamlit/secrets.toml` (this file is ignored by Git):

```toml
GEMINI_API_KEY = "your-key-here"
# Or:
GROQ_API_KEY = "your-key-here"
```

Do not commit API keys or secrets. API key fields are password-masked; the app never writes keys to project files. Use provider keys with suitable usage limits and follow your organization's privacy policies.

## Deploy to Streamlit Community Cloud

1. Create a Git repository containing `app.py`, `foodverse_core.py`, `requirements.txt`, `.streamlit/config.toml`, and this README.
2. Push the repository to GitHub.
3. In Streamlit Community Cloud, create an app pointing to `app.py`.
4. Add `GEMINI_API_KEY` or `GROQ_API_KEY` under the app's Secrets settings if you want live AI enabled. The sidebar still requires the user to choose the provider and grant per-session upload consent.
5. Deploy. Demo mode works when secrets are omitted.

## Important limitations and safety notes

- **Nutrition values are estimates.** A photo cannot reveal exact ingredient weights, preparation oil, recipe details, or nutrition-label data. Use package labels and measured quantities for more reliable results.
- **Allergens:** visual AI cannot guarantee absence of allergens or cross-contact. Always verify packaging and recipe ingredients; never use this app to decide whether a food is safe for a person with a serious allergy.
- **Freshness:** appearance cannot detect most pathogens, toxins, unsafe time/temperature history, or all spoilage. The app is not a microbiological test and must not override use-by dates or official guidance. Do not taste suspect food.
- **Kitchen hygiene:** the score is a heuristic visual screen, not a regulatory inspection, HACCP verification, or certification. Photos cannot prove handwashing, sanitizer concentration/contact time, food temperatures, or absence of pathogens.
- **Meal planning:** BMR/TDEE are estimates for adults. Calorie allocations are menu-planning targets rather than analyzed recipe nutrition. The planner is not a medical diet prescription. Seek individualized clinical advice for chronic conditions, pregnancy, medication interactions, or eating-disorder concerns.
- **Live model availability:** model IDs and quotas can change. If the selected provider/model fails, the app shows an error category and a labelled demo fallback; choose a supported model in the sidebar.
- **Privacy:** when live analysis is enabled, selected images/video frames and prompts are transmitted to the chosen provider. Demo mode does not transmit uploaded media.

## Project structure

```text
food_verse_ai/
├── app.py                  # Streamlit UI and page orchestration
├── foodverse_core.py       # AI adapter, validation, demo fixtures, planning logic
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

## Data contracts

Core model responses are requested as JSON and normalized before UI rendering. See the prompts in `app.py` for the food, freshness, kitchen, and recipe schemas. Reports can be downloaded as JSON from each relevant page.

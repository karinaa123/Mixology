import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import anthropic
import json
import time
import requests
import os
from dotenv import load_dotenv

# ==========================================
# 1. Configuration & Custom Styling
# ==========================================
st.set_page_config(page_title="AI Mixologist", page_icon="🍷", layout="centered")

# Load environment variables (Critical for local lab computer)
load_dotenv()


def get_api_key():
    """Robustly get API key without crashing on missing secrets file"""

    # 1. Check Session State (User manually typed it)
    if st.session_state.get("api_key"):
        return st.session_state.api_key

    # 2. Check Local Environment (.env file) -> PRIORITY FOR LOCAL LAB COMPUTER
    env_key = os.getenv("ANTHROPIC_API_KEY")
    if env_key:
        return env_key

    # 3. Check Streamlit Secrets (Cloud) - Wrapped in TRY/EXCEPT to prevent crash
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except (FileNotFoundError, Exception):
        # This catches the StreamlitSecretNotFoundError so the app doesn't crash locally
        pass

    return None


# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;600&display=swap');

    /* Force Text Colors */
    h1, h2, h3, h4, h5, h6 { color: #E0E0E0 !important; }
    p, .stMarkdown, li, span, label, div { color: #FFFFFF !important; }

    h1 {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem !important;
        text-align: center;
        margin-bottom: 30px;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0E1117; color: #FFFFFF; }

    /* Buttons */
    .stButton>button {
        border-radius: 25px;
        background-color: #1C1C1C;
        color: white !important;
        border: 1px solid #333;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        border-color: #FF4B4B;
        color: #FF4B4B !important;
        transform: scale(1.02);
    }

    .stChatMessage { background-color: #1E1E1E; border-radius: 12px; }
    .stMultiSelect span { color: #FFFFFF !important; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. Session State Management
# ==========================================
if "step" not in st.session_state: st.session_state.step = "mood"
if "messages" not in st.session_state: st.session_state.messages = []
if "inventory_dict" not in st.session_state: st.session_state.inventory_dict = {}
if "flat_inventory" not in st.session_state: st.session_state.flat_inventory = []
if "preferences" not in st.session_state: st.session_state.preferences = {}
if "detected_images" not in st.session_state: st.session_state.detected_images = []


# ==========================================
# 3. Load Resources
# ==========================================
@st.cache_resource
def load_model():
    return YOLO('best.pt')


try:
    model = load_model()
except:
    st.error("Error: 'best.pt' not found. Please place the model file in the same directory.")
    model = None


# ==========================================
# 4. Logic Functions (Claude & Processing)
# ==========================================

def categorize_bottles(brands, api_key):
    """Uses Claude to categorize detected brands."""
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""
    I have these detected bottle labels: {brands}.
    Categorize them strictly into this JSON structure:
    {{
        "Vodka": [],
        "Gin": [],
        "Rum": [],
        "Whiskey": [],
        "Tequila": [],
        "Brandy": [],
        "Liqueurs/Others": []
    }}
    Place the specific brand names (e.g., 'Tito 500ml') into the correct lists.
    Return ONLY valid JSON.
    """
    try:
        msg = client.messages.create(
            model="claude-3-haiku-20240307",  # Works on ALL accounts
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(msg.content[0].text)
    except Exception as e:
        st.error(f"Error categorizing: {e}")
        return {"Uncategorized": brands}


def generate_recipe(inventory, mixers, prefs, api_key):
    """Generates recipe using Haiku model."""
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""
    Act as a Resourceful Home Bartender.

    User Profile:
    - **Current Inventory**: {inventory}
    - **Mixers**: {mixers}
    - **Mood**: {prefs.get('general_flavor')}
    - **Style**: {prefs.get('specific_style')}
    - **Strength**: {prefs.get('strength')}

    Goal: Create the best cocktail using ONLY what the user has.

    STRICT RULES:
    1. **Prioritize Inventory**: Do NOT suggest recipes requiring buying new liquor.
    2. **Kitchen Hacks**: If an ingredient is missing, provide a substitution.

    Output Format (Markdown):
    # [Cocktail Name]
    *[One sentence description]*

    ### Ingredients
    * List quantities.
    * Mark owned items with ✅
    * Mark missing items with ⚠️

    ### 🔄 Substitutions
    * [Missing Item] -> [Household Substitute]

    ### Instructions
    1. Step 1...
    2. Step 2...

    ### Taste Profile
    Describe the taste.
    """

    try:
        with client.messages.stream(
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                model="claude-3-haiku-20240307",  # Works on ALL accounts
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"⚠️ **Error:** {str(e)}"


# ==========================================
# 5. Sidebar (API Key)
# ==========================================
with st.sidebar:
    st.header("Settings")

    # This will now safely check .env first, then Secrets (safely), then Manual
    active_key = get_api_key()

    if active_key:
        st.success("✅ API Key Ready")
        if "api_key" not in st.session_state:
            st.session_state.api_key = active_key
        if st.button("Change Key"):
            st.session_state.api_key = None
            st.rerun()
    else:
        st.warning("⚠️ API Key Missing")
        st.caption("Add to .env file or enter below:")
        api_key_input = st.text_input("Enter Claude API Key", type="password")
        if api_key_input:
            st.session_state.api_key = api_key_input
            st.rerun()

    st.divider()
    if st.button("Reset App"):
        st.session_state.clear()
        st.rerun()

# ==========================================
# 6. Main Application Flow
# ==========================================

st.markdown("<h1>AI Mixologist</h1>", unsafe_allow_html=True)


def display_chat():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


display_chat()

# --- STEP 1: GENERAL FLAVOR ---
if st.session_state.step == "mood":
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "Welcome. What's the vibe today? 🍷"})
        st.rerun()

    st.subheader("Select a Vibe:")
    col1, col2, col3 = st.columns(3)
    flavors = ["Refreshing 🌿", "Sweet & Fruity 🍓", "Sour & Tart 🍋", "Strong & Boozy 🥃", "Bitter & Complex 🍊",
               "Creamy 🥛"]

    for i, flavor in enumerate(flavors):
        col = [col1, col2, col3][i % 3]
        if col.button(flavor, use_container_width=True):
            st.session_state.preferences['general_flavor'] = flavor
            st.session_state.messages.append({"role": "user", "content": flavor})
            st.session_state.messages.append(
                {"role": "assistant", "content": "Got it. Let's see what bottles you have. Upload a photo! 📸"})
            st.session_state.step = "upload"
            st.rerun()

# --- STEP 2: UPLOAD & DETECT ---
elif st.session_state.step == "upload":
    st.subheader("Show me your bar 📸")
    uploaded_files = st.file_uploader("Upload photos", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

    if uploaded_files and st.button("Analyze Bottles"):
        api_key = get_api_key()
        if not api_key:
            st.error("❌ Please check API Key settings in the sidebar.")
        else:
            with st.spinner("Scanning your bar..."):
                all_detected = set()
                st.session_state.detected_images = []
                cols = st.columns(3)

                for idx, uploaded_file in enumerate(uploaded_files):
                    image = Image.open(uploaded_file)
                    image = ImageOps.exif_transpose(image)
                    # Lowered confidence
                    results = model(image, conf=0.15)

                    if results:
                        res_plotted = results[0].plot()
                        st.session_state.detected_images.append(res_plotted)
                        with cols[idx % 3]:
                            st.image(res_plotted, caption=f"Pic #{idx + 1}", channels="BGR", use_container_width=True)
                        for r in results:
                            for box in r.boxes:
                                all_detected.add(model.names[int(box.cls[0])])

                detected_list = list(all_detected)

                if not detected_list:
                    st.warning("⚠️ No bottles detected.")
                    if st.button("Manual Entry"):
                        st.session_state.flat_inventory = []
                        st.session_state.step = "verify"
                        st.rerun()
                else:
                    categorized = categorize_bottles(detected_list, api_key)
                    st.session_state.inventory_dict = categorized
                    flat_list = []
                    for cat, items in categorized.items():
                        flat_list.extend(items)
                    st.session_state.flat_inventory = flat_list
                    st.success("Analysis Complete!")
                    time.sleep(1)
                    st.session_state.step = "verify"
                    st.rerun()

# --- STEP 3: VERIFY ---
elif st.session_state.step == "verify":
    st.write("---")
    if st.session_state.detected_images:
        with st.expander("📸 View Scanned Images", expanded=True):
            img_cols = st.columns(3)
            for i, img in enumerate(st.session_state.detected_images):
                with img_cols[i % 3]:
                    st.image(img, channels="BGR", use_container_width=True)

    st.info("Verify your inventory:")
    cols = st.columns(3)
    idx = 0
    for category, items in st.session_state.inventory_dict.items():
        if items:
            with cols[idx % 3]:
                st.markdown(f"**{category}**")
                for item in items:
                    st.caption(f"- {item}")
            idx += 1

    st.write("**Add missing items:**")
    final_inventory = st.multiselect(
        "Inventory",
        options=["Vodka", "Gin", "Rum", "Whiskey", "Tequila", "Brandy", "Vermouth", "Cointreau", "Baileys",
                 "Kahlua"] + st.session_state.flat_inventory,
        default=st.session_state.flat_inventory
    )

    if st.button("Confirm Inventory ✅"):
        st.session_state.flat_inventory = final_inventory
        st.session_state.messages.append({"role": "assistant", "content": f"Bar locked in."})
        st.session_state.messages.append({"role": "assistant", "content": "What mixers/kitchen items do you have?"})
        st.session_state.step = "mixers"
        st.rerun()

# --- STEP 4: MIXERS ---
elif st.session_state.step == "mixers":
    st.write("---")
    st.markdown("Select **everything** you have:")
    mixer_options = [
        "Coke", "Soda Water", "Tonic", "Ginger Beer", "Sprite",
        "Lemon (Fresh)", "Lime (Fresh)", "Orange Juice", "Cranberry Juice",
        "Sugar", "Honey", "Maple Syrup", "Bitters", "Mint", "Ice", "Milk/Cream", "Coffee"
    ]
    selected_mixers = st.multiselect("Pantry & Mixers", options=mixer_options)
    if st.button("Next Step"):
        st.session_state.preferences['mixers'] = selected_mixers
        st.session_state.step = "refine"
        st.rerun()

# --- STEP 5: REFINE ---
elif st.session_state.step == "refine":
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        strength = st.radio("Strength", ["Low", "Medium", "High"], label_visibility="collapsed")
    with col2:
        style = st.selectbox("Style", ["Classic", "Complex", "Refreshing", "Sweet", "Creamy"])

    if st.button("Mix My Drink! 🍸"):
        st.session_state.preferences['strength'] = strength
        st.session_state.preferences['specific_style'] = style
        st.session_state.step = "mixing"
        st.rerun()

# --- STEP 6: MIXING ---
elif st.session_state.step == "mixing":
    api_key = get_api_key()
    if not api_key:
        st.error("❌ API Key missing.")
        st.stop()

    output_placeholder = st.empty()
    try:
        recipe_stream = generate_recipe(
            st.session_state.flat_inventory,
            st.session_state.preferences['mixers'],
            st.session_state.preferences,
            api_key
        )
        full_response = output_placeholder.write_stream(recipe_stream)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.step = "done"
    except Exception as e:
        st.error(f"❌ Error: {e}")

elif st.session_state.step == "done":
    if st.button("Mix Another 🔄"):
        key = st.session_state.get("api_key")
        st.session_state.clear()
        if key: st.session_state.api_key = key
        st.session_state.step = "mood"
        st.rerun()
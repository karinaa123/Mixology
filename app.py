import streamlit as st
from ultralytics import YOLO
from PIL import Image
import anthropic
import json
import time
import requests

# ==========================================
# 1. Configuration & Custom Styling (Fonts)
# ==========================================
st.set_page_config(page_title="AI Mixologist", page_icon="🍷", layout="centered")


# Load Lottie Animation (Helper function)
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Custom CSS for Fonts and UI
st.markdown("""
    <style>
    /* Import Google Fonts: Playfair Display (Title) and Inter (Body) */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;600&display=swap');

    /* Apply Fonts */
    h1 {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem !important;
        color: #E0E0E0;
        text-align: center;
        margin-bottom: 30px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* App Background */
    .stApp { background-color: #0E1117; color: #FFFFFF; }

    /* Button Styling */
    .stButton>button {
        border-radius: 25px;
        background-color: #1C1C1C;
        color: white;
        border: 1px solid #333;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        border-color: #FF4B4B;
        color: #FF4B4B;
        transform: scale(1.02);
    }

    /* Chat Bubble Styling */
    .stChatMessage { background-color: #1E1E1E; border-radius: 12px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. Session State Management
# ==========================================
if "step" not in st.session_state:
    st.session_state.step = "mood"  # Step 1: Mood/General Flavor
if "messages" not in st.session_state:
    st.session_state.messages = []
if "inventory_dict" not in st.session_state:
    st.session_state.inventory_dict = {}  # Dictionary: {'Vodka': ['Tito'], 'Gin': []}
if "flat_inventory" not in st.session_state:
    st.session_state.flat_inventory = []  # Flat list for the multiselect widget
if "preferences" not in st.session_state:
    st.session_state.preferences = {}


# ==========================================
# 3. Load Resources
# ==========================================
@st.cache_resource
def load_model():
    return YOLO('best.pt')


try:
    model = load_model()
except:
    st.error("Error: 'best.pt' not found. Please add your model file.")
    model = None


# ==========================================
# 4. Logic Functions (Claude & Processing)
# ==========================================

def categorize_bottles(brands, api_key):
    """
    Uses Claude to categorize detected brands into the 6 Base Liquors.
    """
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
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(msg.content[0].text)
    except:
        return {"Uncategorized": brands}


def generate_recipe(inventory, mixers, prefs, api_key):
    """
    Generates recipe based on specific constraints.
    """
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""
    Act as a Master Mixologist.

    User Profile:
    - **Inventory**: {inventory}
    - **Mixers**: {mixers}
    - **General Mood**: {prefs.get('general_flavor')}
    - **Specific Style**: {prefs.get('specific_style')}
    - **Alcohol Strength**: {prefs.get('strength')}

    Task: Create the PERFECT cocktail recipe.

    Output Format (Markdown):
    # [Cocktail Name]
    *[One sentence description explaining why it fits the mood]*

    ### Ingredients
    * List with quantities (ml/oz)
    * Mark user's owned items with ✅
    * Mark missing items with ⚠️

    ### Instructions
    1. Step 1...
    2. Step 2...

    ### Taste Profile
    Describe the taste (e.g., "Sour, Boozy, Floral").
    """

    with client.messages.stream(
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            model="claude-3-5-sonnet-20241022",
    ) as stream:
        for text in stream.text_stream:
            yield text


# ==========================================
# 5. Sidebar (API Key)
# ==========================================
with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input("Claude API Key", type="password")
    if api_key_input:
        st.session_state.api_key = api_key_input
    st.divider()
    if st.button("Reset App"):
        st.session_state.clear()
        st.rerun()

# ==========================================
# 6. Main Application Flow
# ==========================================

# --- APP HEADER ---
st.markdown("<h1>AI Mixologist</h1>", unsafe_allow_html=True)


# Helper to display chat
def display_chat():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


display_chat()

# --- STEP 1: GENERAL FLAVOR (The "Vibe") ---
if st.session_state.step == "mood":
    if not st.session_state.messages:
        st.session_state.messages.append(
            {"role": "assistant", "content": "Welcome. What would you like to drink today? 🍷"})
        st.rerun()

    st.subheader("Select a Vibe:")
    col1, col2, col3 = st.columns(3)

    flavors = ["Refreshing 🌿", "Sweet & Fruity 🍓", "Sour & Tart 🍋", "Strong & Boozy 🥃", "Bitter & Complex 🍊",
               "Creamy 🥛"]

    # Create a grid of buttons
    for i, flavor in enumerate(flavors):
        col = [col1, col2, col3][i % 3]
        if col.button(flavor, use_container_width=True):
            st.session_state.preferences['general_flavor'] = flavor
            st.session_state.messages.append({"role": "user", "content": flavor})
            st.session_state.messages.append({"role": "assistant",
                                              "content": "Excellent choice. Now, what do you have in your bar? Upload a photo and I'll see. 📸"})
            st.session_state.step = "upload"
            st.rerun()

# --- STEP 2: UPLOAD & DETECT ---
elif st.session_state.step == "upload":
    # Accept multiple files
    uploaded_files = st.file_uploader("Upload photos of your bottles", type=['jpg', 'png', 'jpeg'],
                                      accept_multiple_files=True)

    if uploaded_files and st.button("Analyze Bottles"):
        if not st.session_state.get("api_key"):
            st.error("Please enter your API Key in the sidebar.")
        else:
            with st.spinner("Scanning your bar..."):
                all_detected = set()

                # Process each image
                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file)
                    results = model(image)
                    for r in results:
                        for box in r.boxes:
                            all_detected.add(model.names[int(box.cls[0])])

                detected_list = list(all_detected)

                # Claude Categorization
                categorized = categorize_bottles(detected_list, st.session_state.api_key)
                st.session_state.inventory_dict = categorized

                # Flatten for the next step's multiselect
                flat_list = []
                for cat, items in categorized.items():
                    flat_list.extend(items)
                st.session_state.flat_inventory = flat_list

                st.session_state.step = "verify"
                st.rerun()

# --- STEP 3: VERIFY INVENTORY ---
elif st.session_state.step == "verify":
    st.write("---")

    # Display categorized output nicely
    st.info("Here is what I found. Please verify:")

    cols = st.columns(3)
    idx = 0
    # Display the categorized dictionary visually
    for category, items in st.session_state.inventory_dict.items():
        if items:  # Only show if not empty
            with cols[idx % 3]:
                st.markdown(f"**{category}**")
                for item in items:
                    st.caption(f"- {item}")
            idx += 1

    st.write("")
    st.write("**Correction:** If I made a mistake, please edit the list below:")

    # Multiselect for correction
    final_inventory = st.multiselect(
        "Confirm Inventory",
        options=["Vodka", "Gin", "Rum", "Whiskey", "Tequila", "Brandy", "Vermouth",
                 "Cointreau"] + st.session_state.flat_inventory,
        default=st.session_state.flat_inventory
    )

    if st.button("Confirm Inventory ✅"):
        st.session_state.flat_inventory = final_inventory
        st.session_state.messages.append(
            {"role": "assistant", "content": f"I see you have: **{', '.join(final_inventory)}**."})
        st.session_state.messages.append({"role": "assistant", "content": "What mixers or soft drinks do you have?"})
        st.session_state.step = "mixers"
        st.rerun()

# --- STEP 4: MIXERS ---
elif st.session_state.step == "mixers":
    st.write("---")
    mixer_options = [
        "Coke", "Soda Water", "Tonic", "Ginger Beer", "Sprite",
        "Lemon Juice", "Lime Juice", "Orange Juice", "Cranberry Juice",
        "Simple Syrup", "Honey", "Bitters", "Mint", "Ice"
    ]

    selected_mixers = st.multiselect("Select Mixers", options=mixer_options)

    if st.button("Next Step"):
        st.session_state.preferences['mixers'] = selected_mixers
        st.session_state.step = "refine"
        st.rerun()

# --- STEP 5: REFINE PREFERENCES ---
elif st.session_state.step == "refine":
    st.write("---")
    st.subheader("Fine-tune your drink")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Alcohol Strength**")
        strength = st.radio("Strength", ["Low (Light & Easy)", "Medium (Balanced)", "High (Strong)"],
                            label_visibility="collapsed")

    with col2:
        st.markdown("**Specific Style**")
        style = st.selectbox("Style", ["Classic", "Complex", "Long Drink", "Short & Sipping", "Soft & Creamy", "Spicy"])

    if st.button("Mix My Drink! 🍸"):
        st.session_state.preferences['strength'] = strength
        st.session_state.preferences['specific_style'] = style
        st.session_state.step = "mixing"
        st.rerun()

# --- STEP 6: MIXING ANIMATION & RESULT ---
elif st.session_state.step == "mixing":

    # Placeholder for the animation
    animation_placeholder = st.empty()

    # Display a mixing GIF or Animation
    with animation_placeholder.container():
        # Using a reliable GIF from Giphy (Cocktail shaker)
        st.markdown(
            """
            <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                <img src="https://media.giphy.com/media/l0HlOaQcLJ2hHpYcw/giphy.gif" width="300" style="border-radius: 15px;">
            </div>
            <h3 style="text-align: center; color: #aaa;">Mixing your drink...</h3>
            """,
            unsafe_allow_html=True
        )

    # Generate Recipe
    output_placeholder = st.empty()
    recipe_stream = generate_recipe(
        st.session_state.flat_inventory,
        st.session_state.preferences['mixers'],
        st.session_state.preferences,
        st.session_state.api_key
    )

    # Clear animation
    animation_placeholder.empty()

    # Stream result
    full_response = output_placeholder.write_stream(recipe_stream)

    # Save result to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.step = "done"

elif st.session_state.step == "done":
    if st.button("Make Another Drink 🔄"):
        # Keep API Key, reset logic
        key = st.session_state.api_key
        st.session_state.clear()
        st.session_state.api_key = key
        st.session_state.step = "mood"
        st.rerun()
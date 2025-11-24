import streamlit as st
from ultralytics import YOLO
from PIL import Image
import openai
import json
import os

# ==========================================
# 1. App Configuration & Styling
# ==========================================
st.set_page_config(page_title="Mixology - AI_bartander", page_icon="🍸")

# Custom CSS for a sleek, modern look
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stButton>button { border-radius: 20px; border: 1px solid #4b4b4b; width: 100%; }
    .stChatMessage { background-color: #262730; border-radius: 10px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. Session State Management
#    (This keeps track of where the user is in the conversation)
# ==========================================
if "step" not in st.session_state:
    st.session_state.step = "api_key"
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! Please enter your OpenAI API Key in the sidebar to start."}]
if "detected_brands" not in st.session_state:
    st.session_state.detected_brands = []  # e.g., "Tito's"
if "generic_types" not in st.session_state:
    st.session_state.generic_types = []  # e.g., "Vodka"
if "user_mixers" not in st.session_state:
    st.session_state.user_mixers = []


# ==========================================
# 3. Load YOLO Model
# ==========================================
@st.cache_resource
def load_model():
    # Make sure 'best.pt' is in the same folder
    return YOLO('best.pt')


try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'best.pt' is in the folder.")
    model = None


# ==========================================
# 4. OpenAI Helper Functions
# ==========================================
def convert_brands_to_generic(brands, api_key):
    """
    Uses GPT to convert specific labels (Tito 500ml) into generic types (Vodka).
    """
    client = openai.OpenAI(api_key=api_key)
    prompt = f"""
    I have a list of detected bottle labels: {brands}.
    Please categorize them into generic liquor types (e.g., 'Tito 500ml' -> 'Vodka', 'Bacardi' -> 'Rum').
    Return ONLY a JSON list of strings. Example: ["Vodka", "Rum"]
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except:
        return brands  # Fallback to original names if GPT fails


def generate_recipe(liquors, mixers, flavor, api_key):
    """
    Uses GPT to generate a recipe based on inventory and flavor.
    """
    client = openai.OpenAI(api_key=api_key)
    prompt = f"""
    Act as a professional mixologist.
    User Inventory: {liquors}.
    User Mixers: {mixers}.
    Desired Flavor: {flavor}.

    Recommend ONE cocktail. Format it nicely using Markdown.
    Include:
    1. Name 🍸
    2. Description
    3. Ingredients (Mark missing ones with ⚠️)
    4. Steps
    """
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    return stream


# ==========================================
# 5. Sidebar (API Key Input)
# ==========================================
with st.sidebar:
    st.title("Settings")
    api_key_input = st.text_input("OpenAI API Key", type="password")
    if api_key_input:
        st.session_state.api_key = api_key_input
        if st.session_state.step == "api_key":
            st.session_state.step = "upload"
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi! Snap a photo of your bottles and upload it. 📸"}]
            st.rerun()

# ==========================================
# 6. Main App Interface
# ==========================================
st.title("🍸 AI Mixologist")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- STEP 1: UPLOAD & DETECT ---
if st.session_state.step == "upload" and model:
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Bottles", use_column_width=True)

        if st.button("Analyze Bottles"):
            with st.spinner("Scanning bottles..."):
                # Run YOLO
                results = model(image)
                detected = set()
                for r in results:
                    for box in r.boxes:
                        detected.add(model.names[int(box.cls[0])])

                st.session_state.detected_brands = list(detected)

                if not st.session_state.detected_brands:
                    st.error("No bottles detected. Please try again.")
                else:
                    # Convert to generic types using GPT
                    generics = convert_brands_to_generic(st.session_state.detected_brands, st.session_state.api_key)
                    st.session_state.generic_types = generics

                    # Update Chat
                    st.session_state.messages.append({"role": "user", "content": "Image uploaded."})
                    response_text = f"I detected: **{', '.join(st.session_state.detected_brands)}**.\n\nThis looks like: **{', '.join(generics)}**. Is this correct?"
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                    st.session_state.step = "confirm"
                    st.rerun()

# --- STEP 2: CONFIRM / CORRECT LIQUORS ---
elif st.session_state.step == "confirm":
    st.write("---")
    st.write("Edit your liquor list if needed:")

    # Allow user to edit the list
    confirmed_liquors = st.multiselect(
        "Liquor Inventory",
        options=["Vodka", "Gin", "Rum", "Whiskey", "Tequila", "Brandy"] + st.session_state.generic_types,
        default=st.session_state.generic_types
    )

    if st.button("Confirm Liquors"):
        st.session_state.generic_types = confirmed_liquors
        st.session_state.messages.append({"role": "user", "content": f"Confirmed: {', '.join(confirmed_liquors)}"})
        st.session_state.messages.append({"role": "assistant", "content": "Got it. Now, what mixers do you have?"})
        st.session_state.step = "mixers"
        st.rerun()

# --- STEP 3: SELECT MIXERS (Tags/Buttons) ---
elif st.session_state.step == "mixers":
    st.write("---")
    st.write("Select your mixers:")

    mixer_options = ["Coke", "Sprite", "Soda Water", "Tonic", "Lemon Juice", "Lime Juice", "Orange Juice", "Syrup",
                     "Mint", "Ice"]

    selected_mixers = st.multiselect("Mixers", options=mixer_options)

    if st.button("Confirm Mixers"):
        st.session_state.user_mixers = selected_mixers
        st.session_state.messages.append({"role": "user", "content": f"Mixers: {', '.join(selected_mixers)}"})
        st.session_state.messages.append(
            {"role": "assistant", "content": "Understood. Finally, what flavor profile do you prefer?"})
        st.session_state.step = "flavor"
        st.rerun()

# --- STEP 4: CHOOSE FLAVOR & GET RESULT ---
elif st.session_state.step == "flavor":
    st.write("---")
    flavors = ["Refreshing 🌿", "Sweet 🍬", "Strong 💪", "Sour 🍋", "Fruity 🍓"]

    cols = st.columns(len(flavors))
    for i, flavor in enumerate(flavors):
        if cols[i].button(flavor):
            st.session_state.messages.append({"role": "user", "content": f"I want something {flavor}."})
            st.session_state.step = "processing"  # Prevent double clicks
            st.rerun()

# --- STEP 5: DISPLAY RECOMMENDATION ---
elif st.session_state.step == "processing":
    with st.chat_message("assistant"):
        st.write("Thinking of a recipe...")
        # Get last user flavor preference from chat history
        last_flavor = st.session_state.messages[-1]["content"]

        stream = generate_recipe(
            st.session_state.generic_types,
            st.session_state.user_mixers,
            last_flavor,
            st.session_state.api_key
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.step = "done"

elif st.session_state.step == "done":
    if st.button("Start Over"):
        api_key = st.session_state.api_key
        st.session_state.clear()
        st.session_state.api_key = api_key
        st.session_state.step = "upload"
        st.session_state.messages = [{"role": "assistant", "content": "Ready for round two? Upload a photo. 📸"}]
        st.rerun()
import csv
import io
import json
import re
import time
from collections import defaultdict

import streamlit as st
from google import genai
from google.genai import types
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# ======================================================
# CONFIGURATION
# ======================================================

APP_TITLE = "Business Card Extractor"
MODEL_NAME = "gemini-2.5-flash"
CSV_NAME = "business_cards_result.csv"

DEFAULT_DELAY = 1.5
MAX_DELAY = 5.0

FRONT_SUFFIX = "_front"
BACK_SUFFIX = "_back"

SUPPORTED_MIME_TYPES = [
    "image/jpeg",
    "image/png",
    "image/webp",
]

COLONNES = [
    "card_id", "front_image", "back_image",
    "first_name", "last_name", "full_name",
    "job_title", "company", "department",
    "email", "phone_main", "phone_mobile", "phone_direct", "fax",
    "website", "street_address", "city", "state_region",
    "postal_code", "country", "linkedin",
    "raw_text", "erreur"
]

PROMPT = """
You are reading an English-language business card.
The card may have a front side and a back side.
Use both images as one single business card.

Extract the information as strict valid JSON:

{
  "first_name": "",
  "last_name": "",
  "full_name": "",
  "job_title": "",
  "company": "",
  "department": "",
  "email": "",
  "phone_main": "",
  "phone_mobile": "",
  "phone_direct": "",
  "fax": "",
  "website": "",
  "street_address": "",
  "city": "",
  "state_region": "",
  "postal_code": "",
  "country": "",
  "linkedin": "",
  "raw_text": ""
}

Rules:
- Do not guess.
- If a field is missing or uncertain, use "".
- The company name or logo may appear only on the back.
- Merge information from front and back into one JSON object.
- Do not create two contacts.
- Preserve original spelling, capitalization, and accents.
- Split US/UK-style addresses carefully.
- For US addresses, put the state abbreviation in "state_region", e.g. CA, NY, TX.
- For UK addresses, put county/region in "state_region" if present.
- Keep phone numbers exactly as written, including +1, +44, extensions, parentheses, and dashes.
- Put extensions like ext. 204 in the relevant phone field.
- Respond only with valid JSON. No markdown.
"""


# ======================================================
# SERVICES
# ======================================================

@st.cache_resource
def get_drive_service():
    return build(
        "drive",
        "v3",
        developerKey=st.secrets["GOOGLE_DRIVE_API_KEY"]
    )


@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


# ======================================================
# GOOGLE DRIVE
# ======================================================

def list_drive_images(service, folder_id):
    mime_query = " or ".join(
        [f"mimeType='{mime}'" for mime in SUPPORTED_MIME_TYPES]
    )

    query = (
        f"'{folder_id}' in parents and trashed = false and "
        f"({mime_query})"
    )

    files = []
    page_token = None

    while True:
        response = service.files().list(
            q=query,
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
            pageSize=1000,
        ).execute()

        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")

        if not page_token:
            break

    return sorted(files, key=lambda x: x["name"])


def download_drive_file(service, file_id):
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    return buffer.getvalue()


# ======================================================
# DATA PROCESSING
# ======================================================

def clean_json(text):
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    elif text.startswith("```"):
        text = text.replace("```", "").strip()

    return json.loads(text)


def build_card_pairs(files):
    cards = defaultdict(lambda: {"front": None, "back": None})

    for file in files:
        name = file["name"]
        stem = re.sub(
            r"\.(jpg|jpeg|png|webp)$",
            "",
            name,
            flags=re.IGNORECASE
        )

        lower_stem = stem.lower()

        if lower_stem.endswith(FRONT_SUFFIX):
            card_id = re.sub(
                f"{FRONT_SUFFIX}$",
                "",
                stem,
                flags=re.IGNORECASE
            )
            cards[card_id]["front"] = file

        elif lower_stem.endswith(BACK_SUFFIX):
            card_id = re.sub(
                f"{BACK_SUFFIX}$",
                "",
                stem,
                flags=re.IGNORECASE
            )
            cards[card_id]["back"] = file

    return dict(cards)


def load_existing_csv(uploaded_file):
    if uploaded_file is None:
        return set()

    content = uploaded_file.read().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))

    return {
        row["card_id"]
        for row in reader
        if row.get("card_id")
    }


def rows_to_csv(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=COLONNES)
    writer.writeheader()

    for row in rows:
        writer.writerow({col: row.get(col, "") for col in COLONNES})

    return output.getvalue().encode("utf-8-sig")


def estimate_time(number_of_cards, delay):
    seconds = number_of_cards * (delay + 3)
    minutes = seconds / 60

    if minutes < 1:
        return f"{int(seconds)} secondes environ"

    return f"{minutes:.1f} minutes environ"


# ======================================================
# GEMINI EXTRACTION
# ======================================================

def extract_card(client, service, card_id, front_file, back_file=None, use_back=True):
    row = {col: "" for col in COLONNES}
    row["card_id"] = card_id
    row["front_image"] = front_file["name"] if front_file else ""
    row["back_image"] = back_file["name"] if back_file and use_back else ""

    try:
        if not front_file:
            row["erreur"] = "Image front manquante"
            return row

        contents = []

        front_bytes = download_drive_file(service, front_file["id"])
        contents.append(
            types.Part.from_bytes(
                data=front_bytes,
                mime_type=front_file["mimeType"],
            )
        )

        if use_back and back_file:
            back_bytes = download_drive_file(service, back_file["id"])
            contents.append(
                types.Part.from_bytes(
                    data=back_bytes,
                    mime_type=back_file["mimeType"],
                )
            )

        contents.append(PROMPT)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
        )

        data = clean_json(response.text)
        row.update(data)
        row["erreur"] = ""

    except Exception as e:
        row["erreur"] = str(e)

    return row


# ======================================================
# STREAMLIT UI
# ======================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📇",
    layout="wide",
)

st.title("📇 Business Card Extractor")
st.caption("Extraction automatique de cartes de visite depuis un dossier Google Drive public.")

try:
    FOLDER_ID = st.secrets["FOLDER_ID"]
except KeyError:
    st.error("FOLDER_ID est manquant dans les secrets Streamlit.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Paramètres")

    mode = st.radio(
        "Mode de traitement",
        [
            "Traiter toutes les cartes",
            "Traiter uniquement les nouvelles cartes"
        ]
    )

    old_csv = None
    if mode == "Traiter uniquement les nouvelles cartes":
        old_csv = st.file_uploader(
            "Importer l'ancien CSV",
            type=["csv"]
        )

    use_back = st.checkbox(
        "Utiliser le verso quand disponible",
        value=True
    )

    delay = st.slider(
        "Pause entre deux cartes",
        min_value=0.0,
        max_value=MAX_DELAY,
        value=DEFAULT_DELAY,
        step=0.5
    )

    show_raw_text = st.checkbox(
        "Afficher raw_text dans le tableau",
        value=False
    )

    show_errors_only = st.checkbox(
        "Afficher uniquement les lignes en erreur",
        value=False
    )

st.subheader("📂 Dossier Google Drive")
st.write("L'application utilise le dossier Google Drive configuré dans les secrets.")

drive_service = get_drive_service()
gemini_client = get_gemini_client()

with st.spinner("Lecture du dossier Google Drive..."):
    files = list_drive_images(drive_service, FOLDER_ID)

if not files:
    st.error("Aucune image trouvée. Vérifie que le dossier Drive est public et que FOLDER_ID est correct.")
    st.stop()

cards = build_card_pairs(files)

valid_cards = {
    card_id: pair
    for card_id, pair in cards.items()
    if pair.get("front") is not None
}

cards_without_front = {
    card_id: pair
    for card_id, pair in cards.items()
    if pair.get("front") is None and pair.get("back") is not None
}

already_processed = load_existing_csv(old_csv)

cards_to_process = valid_cards

if mode == "Traiter uniquement les nouvelles cartes":
    cards_to_process = {
        card_id: pair
        for card_id, pair in valid_cards.items()
        if card_id not in already_processed
    }

col1, col2, col3, col4 = st.columns(4)

col1.metric("Images détectées", len(files))
col2.metric("Cartes valides", len(valid_cards))
col3.metric("Cartes à traiter", len(cards_to_process))
col4.metric("Sans recto", len(cards_without_front))

st.info(f"Temps estimé : {estimate_time(len(cards_to_process), delay)}")

if cards_without_front:
    with st.expander("Voir les cartes avec verso mais sans recto"):
        st.write(list(cards_without_front.keys()))

if mode == "Traiter uniquement les nouvelles cartes" and old_csv is None:
    st.warning("Importe un ancien CSV pour détecter les nouvelles cartes.")

start = st.button("🚀 Lancer l'extraction", type="primary")

if start:
    if mode == "Traiter uniquement les nouvelles cartes" and old_csv is None:
        st.error("Ancien CSV manquant.")
        st.stop()

    if not cards_to_process:
        st.info("Aucune carte à traiter.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()
    rows = []

    for index, (card_id, pair) in enumerate(cards_to_process.items(), start=1):
        status.write(f"Traitement {index}/{len(cards_to_process)} : {card_id}")

        row = extract_card(
            client=gemini_client,
            service=drive_service,
            card_id=card_id,
            front_file=pair["front"],
            back_file=pair.get("back"),
            use_back=use_back,
        )

        rows.append(row)
        progress.progress(index / len(cards_to_process))

        if delay > 0:
            time.sleep(delay)

    st.success("Extraction terminée.")

    display_rows = rows

    if show_errors_only:
        display_rows = [
            row for row in display_rows
            if row.get("erreur")
        ]

    if not show_raw_text:
        display_rows = [
            {k: v for k, v in row.items() if k != "raw_text"}
            for row in display_rows
        ]

    st.dataframe(display_rows, width="stretch")

    csv_bytes = rows_to_csv(rows)

    st.download_button(
        label="📥 Télécharger le CSV",
        data=csv_bytes,
        file_name=CSV_NAME,
        mime="text/csv",
    )
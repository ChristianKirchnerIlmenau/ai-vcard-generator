import os
import subprocess
import sys
# !pip install scipy --quiet (Nicht in Skripten verwendbar)
# !pip install tenacity --quiet (Nicht in Skripten verwendbar)
# !pip install tiktoken --quiet (Nicht in Skripten verwendbar)
# !pip install termcolor --quiet (Nicht in Skripten verwendbar)
# !pip install openai --quiet (Nicht in Skripten verwendbar)
import re
import easyocr
import json
from typing import Dict
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored

# load api key from bash env
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

GPT_MODEL = "gpt-4o"
client = OpenAI()

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


# EasyOCR-Reader initialisieren
reader = easyocr.Reader(['en', 'de'], gpu=False)

def analyze_text_with_ai(text: str) -> Dict[str, str]:
    """
    Nutzt generative KI zur besseren Analyse der OCR-Daten.
    """
    prompt = f"""
    Analysiere folgenden Text und extrahiere die relevanten Informationen
    für eine Visitenkarte:

    {text}

    Gib mir eine strukturierte JSON-Ausgabe mit den Feldern:
    - company
    - name
    - surname
    - email
    - phone
    - mobile
    - fax
    - address
    - title
    - position
    """

    chat_response = chat_completion_request(
       [{'role': 'user', 'content': prompt}]
    )

    # debug
    #print(f"Chat Response: {chat_response}")
    if not chat_response:
        print(f"Fehler: Chat Response ist leer!")
        sys.exit(1)

    if chat_response and chat_response.choices and hasattr(chat_response.choices[0].message, 'content') and chat_response.choices[0].message.content.strip():
        try:
            response_content = chat_response.choices[0].message.content.strip()
            if response_content.startswith('```json') and response_content.endswith('```'):
                response_content = response_content[7:-3].strip()
            return json.loads(response_content)
        except json.JSONDecodeError:
            print("Fehler: Ungültige JSON-Antwort von OpenAI.")
            return {
                "company": "Unbekannt",
                "name": "Unbekannt",
                "surname": "",
                "email": "keine_email@example.com",
                "phone": "keine_nummer",
                "mobile": "N/A",
                "fax": "N/A",
                "address": "Unbekannte Adresse",
                "title": "Kein Titel",
                "position": "N/A"
            }
    else:
        print("Warnung: Leere Antwort von OpenAI erhalten.")
        return {
            "company": "Unbekannt",
            "name": "Unbekannt",
            "surname": "",
            "email": "keine_email@example.com",
            "phone": "keine_nummer",
            "mobile": "N/A",
            "fax": "N/A",
            "address": "Unbekannte Adresse",
            "title": "Kein Titel",
            "position": "N/A"
        }

# Funktion, um Text aus einem Bild mit EasyOCR zu extrahieren
def extract_text_from_image(image_path: str) -> str:
    result = reader.readtext(image_path, detail=0)
    return "\n".join(result)

# KI-gestützte Funktion zur Extraktion der Kundendaten mit OpenAI
def extract_customer_data(ocr_text: str) -> Dict[str, str]:
    return analyze_text_with_ai(ocr_text)

# Funktion, um eine vCard-Datei zu erstellen
def create_vcard(data: Dict[str, str], filename: str):
    vcard_content = f"""BEGIN:VCARD
VERSION:3.0
FN:{str(data.get('name', 'Unbekannt')).strip()} {str(data.get('surname', '')).strip()}
ORG:{str(data.get('company', 'Unbekannt')).strip()}
EMAIL:{str(data.get('email', 'keine_email@example.com')).strip()}
TEL:{str(data.get('phone', 'keine_nummer')).strip()}
TEL;TYPE=mobile:{str(data.get('mobile', 'N/A')).strip()}
TEL;TYPE=fax:{str(data.get('fax', 'N/A')).strip()}
ADR:{str(data.get('address', 'Unbekannte Adresse')).strip()}
TITLE:{str(data.get('title', 'Kein Titel')).strip()}
ROLE:{str(data.get('position', 'N/A')).strip()}
END:VCARD"""
    with open(filename, 'w') as file:
        file.write(vcard_content)

def main(folder_path: str):
    # Prüfe, ob der Pfad existiert und ein Ordner ist
    if not os.path.exists(folder_path):
        print(f"Fehler: Der angegebene Pfad '{folder_path}' existiert nicht.")
        sys.exit(1)
    if not os.path.isdir(folder_path):
        print(f"Fehler: '{folder_path}' ist kein gültiger Ordner.")
        sys.exit(1)

    print(f"\nVerwende Ordnerpfad: {folder_path}")

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        print(f"Fehler: Keine .jpg oder .png Dateien im Ordner '{folder_path}' gefunden.")
        sys.exit(1)

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        print(f"\n\nVerarbeite Datei: \t\t\t{image_path}")

        extracted_text = extract_text_from_image(image_path)
        customer_data = extract_customer_data(extracted_text)

        try:
            customer_data = extract_customer_data(extracted_text)
        except Exception as e:
            print(f"Fehler bei der OpenAI-Auswertung: {e}")
            customer_data = {
                "company": "Unbekannt",
                "name": "Unbekannt",
                "email": "keine_email@example.com",
                "phone": "keine_nummer",
                "address": "Unbekannte Adresse",
                "title": "Kein Titel"
    }
        print(f"\nGefundene Daten: \t\t\t{customer_data}")

        # vCard-Datei mit Firmenname und Name des Kontakts speichern
        safe_filename = f"{customer_data.get('company', 'Unbekannt').replace(' ', '_')}_{customer_data.get('name', 'Unbekannt').replace(' ', '_')}.vcf"
        vcard_filepath = os.path.join(folder_path, safe_filename)
        create_vcard(customer_data, vcard_filepath)

        print(f"vCard gespeichert als: {vcard_filepath}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py \"<path_to_your_images_folder>\"")
        sys.exit(1)

    folder_path = sys.argv[1]
    main(folder_path)

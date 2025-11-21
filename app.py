import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import io
import os
import tempfile
import gdown
import json
from torchvision import transforms, models
import re
import cv2
import pytesseract

# Configuration de la page
st.set_page_config(
    page_title="OCR et Détection de Falsification",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f1f1f;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .json-display {
        background-color: #f4f4f4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #2196F3;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
    }
    .classification-result {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 600;
    }
    .normal {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        color: #155724;
    }
    .forgery {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Classe du modèle de classification
class DocClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.base_model = models.efficientnet_v2_m(weights=None)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

# Fonction pour télécharger depuis Google Drive
@st.cache_resource
def download_from_drive(file_id, output_path):
    """Télécharge un fichier depuis Google Drive"""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        st.error(f"Erreur lors du téléchargement: {str(e)}")
        return False

# Fonction pour charger le modèle de classification
@st.cache_resource
def load_classifier_model(model_id):
    """Charge le modèle EfficientNetV2 pour la classification"""
    try:
        with st.spinner("Chargement du modèle de classification..."):
            temp_dir = tempfile.mkdtemp()
            model_path = os.path.join(temp_dir, "classifier.pth")
            
            if not download_from_drive(model_id, model_path):
                return None
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = DocClassifier(num_classes=5).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            st.success("Modèle de classification chargé avec succès")
            return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle de classification: {str(e)}")
        return None

# Fonction pour charger le modèle OCR avec Unsloth
@st.cache_resource
def load_ocr_model_unsloth(model_id):
    """Charge le modèle Qwen2.5-VL fine-tuné avec Unsloth"""
    try:
        with st.spinner("Installation et chargement d'Unsloth..."):
            # Installation d'Unsloth
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "unsloth"])
            
            from unsloth import FastVisionModel
            import torch
            
            temp_dir = tempfile.mkdtemp()
            model_path = os.path.join(temp_dir, "qwen_model")
            
            if model_id:
                # Télécharger depuis Google Drive
                if not download_from_drive(model_id, model_path + ".zip"):
                    return None, None
                
                # Décompresser
                import zipfile
                if os.path.exists(model_path + ".zip"):
                    with zipfile.ZipFile(model_path + ".zip", 'r') as zip_ref:
                        zip_ref.extractall(model_path)
            else:
                # Utiliser le modèle de base
                model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
            
            # Charger avec Unsloth
            model, tokenizer = FastVisionModel.from_pretrained(
                model_path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            
            st.success("Modèle OCR (Unsloth) chargé avec succès")
            return model, tokenizer
            
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle OCR Unsloth: {str(e)}")
        return None, None

# Fonctions de prétraitement d'image pour OCR de secours
def preprocess_image(image):
    """Prétraite l'image pour améliorer l'OCR"""
    img = np.array(image)
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    gray = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return Image.fromarray(thresh)

def extract_text_tesseract(image):
    """Extrait le texte avec Tesseract OCR (solution de secours)"""
    try:
        processed_image = preprocess_image(image)
        custom_config = r'--oem 3 --psm 6 -l eng+fra+spa+rus'
        text = pytesseract.image_to_string(np.array(processed_image), config=custom_config)
        return text
    except Exception as e:
        return f"Erreur Tesseract: {str(e)}"

# Prompts pour l'extraction OCR (identique à votre notebook)
EXTRACTION_PROMPT = """Analyze the ID card image with extreme precision and extract ONLY the visibly present information.

**CARD TYPE IDENTIFICATION:**
First determine the exact card type by examining these key characteristics:
- **USA**: Contains physical attributes (height, weight, eye color), address field, and MM/DD/YYYY date format
- **RUS**: Written in Cyrillic characters, includes patronymic_name field, uses DD.MM.YYYY date format with dots
- **EST**: Shows "PD" type designation, "EST" country codes, uses DD.MM.YYYY date format with dots
- **ESP**: Contains second_surname field, uses DD MM YYYY date format with spaces as separators

**EXACT FIELD REQUIREMENTS FOR EACH CARD TYPE:**

**USA CARD - Output this exact structure:**
{
    "name": "Complete full name as printed on card (first, middle, last)",
    "address": "Complete street address including city, state, and ZIP code",
    "birthday": "Date of birth in MM/DD/YYYY format only",
    "gender": "Single character: M for Male or F for Female",
    "class": "License class designation (D, C, etc.)",
    "issue_date": "Issue date in MM/DD/YYYY format only",
    "expire_date": "Expiration date in MM/DD/YYYY format only",
    "license_number": "Driver license number exactly as printed",
    "height": "Height measurement with units (e.g., 5'-07\")",
    "weight": "Weight measurement with units (e.g., 115 lb)",
    "eye_color": "Eye color abbreviation (BRO, BLU, GRN, etc.)",
    "DD": "Document discriminator number"
}

**RUS CARD - Output this exact structure:**
{
    "country_code": "Always 'RUS'",
    "surname": "Family name in Cyrillic characters exactly as printed",
    "given_name": "First name in Cyrillic characters exactly as printed",
    "birthday": "Date of birth in DD.MM.YYYY format with dots",
    "issue_date": "Issue date in DD.MM.YYYY format with dots",
    "expire_date": "Expiration date in DD.MM.YYYY format with dots",
    "gender": "Either 'ЖЕН' for female or 'МУЖ' for male",
    "patronymic_name": "Patronymic name in Cyrillic characters exactly as printed",
    "place_of_birth": "City/place of birth in Cyrillic characters exactly as printed",
    "card_num": "Card identification number with spaces as shown"
}

**EST CARD - Output this exact structure:**
{
    "type": "Always 'PD'",
    "country_code": "Always 'EST'",
    "country": "Always 'EST'",
    "surname": "Family name exactly as printed in Latin characters",
    "given_name": "First name exactly as printed in Latin characters",
    "birthday": "Date of birth in DD.MM.YYYY format with dots",
    "gender": "Either 'M' for male or 'N/F' for female",
    "issue_date": "Issue date in DD.MM.YYYY format with dots",
    "expire_date": "Expiration date in DD.MM.YYYY format with dots",
    "place_of_birth": "Country of birth (typically 'EST')",
    "issue_authority": "Issuing authority abbreviation (e.g., 'PPA')",
    "personal_num": "Personal identification number without spaces",
    "card_num": "Card number exactly as printed"
}

**ESP CARD - Output this exact structure:**
{
    "country_code": "Always 'ESP'",
    "surname": "Primary family name exactly as printed",
    "given_name": "First name exactly as printed",
    "birthday": "Date of birth in DD MM YYYY format with spaces",
    "issue_date": "Issue date in DD MM YYYY format with spaces",
    "expire_date": "Expiration date in DD MM YYYY format with spaces",
    "gender": "Single character: 'M' for male or 'F' for female",
    "second_surname": "Secondary family name exactly as printed",
    "card_num": "Card identification number with letters/numbers as shown",
    "personal_num": "Personal identification number exactly as printed"
}

**CRITICAL EXTRACTION RULES:**
- Extract ONLY text that is clearly visible and legible in the image
- DO NOT invent, assume, or hallucinate any information
- If a field is not visible or cannot be read, OMIT it completely from the JSON output
- Preserve exact spelling, capitalization, spacing, and special characters
- Maintain the precise date format specified for each card type
- Output ONLY the raw JSON object without any additional text, explanations, or formatting
- Do not translate any text - preserve original language and characters

RESPOND ONLY WITH THE REQUIRED JSON OBJECT AND WITH THE EXACT STRUCTURE AS MENTIONED.
"""

# Fonction d'extraction OCR avec Unsloth
def extract_info_ocr_unsloth(image, ocr_model, tokenizer):
    """Extrait les informations avec le modèle Unsloth"""
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": EXTRACTION_PROMPT}
                ]
            }
        ]
        
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(
            text=[input_text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(ocr_model.device)
        
        with torch.no_grad():
            output = ocr_model.generate(
                **inputs,
                max_new_tokens=256,
                use_cache=True
            )
        
        input_length = inputs.input_ids.shape[1]
        response_ids = output[0][input_length:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Extraire le JSON de la réponse
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                json_object = json.loads(json_string)
                return json_object
            else:
                return {"error": "Aucun JSON trouvé dans la réponse", "raw_response": response}
        except json.JSONDecodeError:
            return {"error": "Format JSON invalide", "raw_response": response}
            
    except Exception as e:
        return {"error": f"Erreur lors de l'extraction: {str(e)}"}

# Fonction de classification
def classify_document(image, classifier_model):
    """Classifie le document (normal ou falsifié)"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        transform = transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = classifier_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
        
        class_names = ['normal', 'forgery_1', 'forgery_2', 'forgery_3', 'forgery_4']
        all_probs = {class_names[i]: float(probabilities[0][i].item() * 100) for i in range(5)}
        
        return {
            'class': class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': all_probs
        }
    except Exception as e:
        return {'error': f"Erreur lors de la classification: {str(e)}"}

# Interface principale
def main():
    # En-tête
    st.markdown('<p class="main-header">OCR et Détection de Falsification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Extraction automatique d\'informations et détection de documents falsifiés</p>', unsafe_allow_html=True)
    
    # Barre latérale
    with st.sidebar:
        st.header("Configuration des Modèles")
        
        st.markdown("""
        <div class="info-box">
        <strong>Instructions:</strong><br>
        Entrez les identifiants Google Drive des modèles entraînés.
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Modèle OCR (Qwen2.5-VL Unsloth)")
        ocr_model_id = st.text_input(
            "ID Google Drive - Modèle OCR fine-tuné",
            placeholder="1ABC...XYZ (optionnel)",
            help="Laissez vide pour utiliser le modèle de base"
        )
        
        st.markdown("---")
        
        st.subheader("Modèle de Classification")
        classifier_id = st.text_input(
            "ID Google Drive - EfficientNetV2",
            placeholder="1ABC...XYZ",
            help="ID du fichier best_model.pth"
        )
        
        st.markdown("---")
        
        st.info("""
        **Note:** Le modèle OCR utilise Unsloth pour le fine-tuning.
        Si le modèle OCR n'est pas disponible, Tesseract sera utilisé comme solution de secours.
        """)
        
        load_button = st.button("Charger les Modèles", type="primary")
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.85rem; color: #666;">
        <strong>Technologies</strong><br>
        - OCR: Qwen2.5-VL-7B + Unsloth<br>
        - Classification: EfficientNetV2<br>
        - Framework: PyTorch + Streamlit
        </div>
        """, unsafe_allow_html=True)
    
    # Charger les modèles
    if load_button:
        if not classifier_id:
            st.error("Veuillez fournir l'ID du modèle de classification.")
        else:
            st.session_state.ocr_model, st.session_state.ocr_tokenizer = load_ocr_model_unsloth(ocr_model_id if ocr_model_id else None)
            st.session_state.classifier_model = load_classifier_model(classifier_id)
    
    # Vérifier si les modèles sont chargés
    if 'classifier_model' not in st.session_state or st.session_state.classifier_model is None:
        st.markdown("""
        <div class="warning-box">
        <strong>Aucun modèle chargé</strong><br>
        Veuillez configurer et charger les modèles dans la barre latérale pour commencer.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Fonctionnalités OCR
            
            **Extraction automatique d'informations**
            - Reconnaissance de 4 types de cartes d'identité
            - Support multilingue (anglais, russe, espagnol, estonien)
            - Extraction de tous les champs pertinents
            - Format JSON structuré
            
            **Précision élevée**
            - Modèle fine-tuné sur données spécifiques
            - Validation loss de 0.02
            - Gestion des formats de dates variés
            """)
        
        with col2:
            st.markdown("""
            ### Détection de Falsification
            
            **Classification en 5 catégories**
            - Normal (document authentique)
            - Forgery 1, 2, 3, 4 (différents types de falsification)
            
            **Performance**
            - Précision globale: 75.3%
            - Modèle basé sur EfficientNetV2
            - Score de confiance pour chaque prédiction
            """)
        
        return
    
    # Interface principale d'analyse
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Télécharger une image de carte d'identité",
        type=['jpg', 'jpeg', 'png'],
        help="Formats supportés: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Image téléchargée", use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="result-box">
            <h3>Informations sur le fichier</h3>
            """, unsafe_allow_html=True)
            
            st.write(f"**Nom:** {uploaded_file.name}")
            st.write(f"**Dimensions:** {image.size[0]} x {image.size[1]} pixels")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Analyser le Document", type="primary"):
            with st.spinner("Analyse en cours..."):
                
                # Classification
                st.markdown('<p class="section-header">1. Détection de Falsification</p>', unsafe_allow_html=True)
                
                classification_result = classify_document(image, st.session_state.classifier_model)
                
                if 'error' in classification_result:
                    st.error(classification_result['error'])
                else:
                    is_normal = classification_result['class'] == 'normal'
                    class_css = 'normal' if is_normal else 'forgery'
                    
                    status_text = "Document Authentique" if is_normal else f"Document Falsifié ({classification_result['class']})"
                    
                    st.markdown(f"""
                    <div class="classification-result {class_css}">
                    <h3 style="margin: 0;">{status_text}</h3>
                    <p style="margin: 0.5rem 0 0 0;">Confiance: {classification_result['confidence']:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Distribution des probabilités
                    st.markdown("**Distribution des probabilités:**")
                    for class_name, prob in classification_result['all_probabilities'].items():
                        st.markdown(f"""
                        <div style="margin: 0.5rem 0;">
                        <strong>{class_name}:</strong>
                        <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px; margin-top: 5px;">
                            <div style="background-color: {'#28a745' if class_name == 'normal' else '#dc3545'}; width: {prob}%; height: 100%; border-radius: 10px; display: flex; align-items: center; padding-left: 10px; color: white; font-weight: bold;">
                                {prob:.2f}%
                            </div>
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Extraction OCR
                st.markdown('<p class="section-header">2. Extraction des Informations (OCR)</p>', unsafe_allow_html=True)
                
                if 'ocr_model' in st.session_state and st.session_state.ocr_model is not None:
                    # Utiliser le modèle Unsloth
                    ocr_result = extract_info_ocr_unsloth(
                        image,
                        st.session_state.ocr_model,
                        st.session_state.ocr_tokenizer
                    )
                else:
                    # Solution de secours avec Tesseract
                    st.warning("Utilisation de Tesseract OCR (solution de secours)")
                    extracted_text = extract_text_tesseract(image)
                    ocr_result = {"extracted_text": extracted_text, "method": "tesseract"}
                
                if 'error' in ocr_result:
                    st.warning(f"OCR: {ocr_result['error']}")
                    if 'raw_response' in ocr_result:
                        st.text_area("Réponse brute:", ocr_result['raw_response'], height=200)
                else:
                    if 'method' in ocr_result and ocr_result['method'] == 'tesseract':
                        st.markdown("**Texte extrait (Tesseract):**")
                        st.text_area("", ocr_result['extracted_text'], height=200)
                    else:
                        st.markdown("**Informations extraites:**")
                        st.markdown(f"""
                        <div class="json-display">
                        <pre>{json.dumps(ocr_result, indent=2, ensure_ascii=False)}</pre>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Affichage formaté des informations clés
                        if isinstance(ocr_result, dict):
                            st.markdown("**Résumé des informations:**")
                            col1, col2 = st.columns(2)
                            
                            keys = list(ocr_result.keys())
                            mid = len(keys) // 2
                            
                            with col1:
                                for key in keys[:mid]:
                                    st.write(f"**{key}:** {ocr_result[key]}")
                            
                            with col2:
                                for key in keys[mid:]:
                                    st.write(f"**{key}:** {ocr_result[key]}")
                
                # Bouton de téléchargement des résultats
                st.markdown("---")
                results = {
                    "filename": uploaded_file.name,
                    "classification": classification_result,
                    "ocr_extraction": ocr_result
                }
                
                json_str = json.dumps(results, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Télécharger les résultats (JSON)",
                    data=json_str,
                    file_name=f"analyse_{uploaded_file.name.split('.')[0]}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()

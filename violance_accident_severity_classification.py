# """
# ═══════════════════════════════════════════════════════════
# MULTI-AGENT EMERGENCY DETECTION SYSTEM
# Streamlit Web Application - FIXED VERSION

# Agents:
# 1. Violence Detection (99.90%)
# 2. Accident Detection (99.91%)
# 3. Severity Classification (83.81%)

# Author: Shiv
# ═══════════════════════════════════════════════════════════
# """

# import streamlit as st
# import torch
# import torch.nn as nn
# import cv2
# import numpy as np
# from PIL import Image
# import open_clip
# from transformers import BlipProcessor, BlipForConditionalGeneration
# import tempfile
# import os
# from datetime import datetime

# # ═══════════════════════════════════════════════════════════
# # PAGE CONFIG
# # ═══════════════════════════════════════════════════════════

# st.set_page_config(
#     page_title="Emergency Detection System",
#     page_icon="🚨",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ═══════════════════════════════════════════════════════════
# # CUSTOM CSS
# # ═══════════════════════════════════════════════════════════

# st.markdown("""
# <style>
#     .main {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     }
    
#     h1 {
#         color: #ffffff;
#         text-align: center;
#         padding: 20px;
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         border-radius: 10px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
    
#     h2 {
#         color: #667eea;
#         border-bottom: 3px solid #667eea;
#         padding-bottom: 10px;
#     }
    
#     .stButton>button {
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         padding: 15px 30px;
#         border-radius: 25px;
#         font-size: 16px;
#         font-weight: bold;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.2);
#     }
    
#     .success-box {
#         background: #d4edda;
#         padding: 20px;
#         border-radius: 10px;
#         border-left: 5px solid #28a745;
#         margin: 10px 0;
#     }
    
#     .danger-box {
#         background: #f8d7da;
#         padding: 20px;
#         border-radius: 10px;
#         border-left: 5px solid #dc3545;
#         margin: 10px 0;
#     }
    
#     .warning-box {
#         background: #fff3cd;
#         padding: 20px;
#         border-radius: 10px;
#         border-left: 5px solid #ffc107;
#         margin: 10px 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ═══════════════════════════════════════════════════════════
# # MODEL ARCHITECTURES
# # ═══════════════════════════════════════════════════════════

# class ViolenceDetector(nn.Module):
#     def __init__(self, input_dim=1024, hidden1=512, hidden2=256, num_classes=2, dropout=0.3):
#         super(ViolenceDetector, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden1)
#         self.bn1 = nn.BatchNorm1d(hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.bn2 = nn.BatchNorm1d(hidden2)
#         self.fc3 = nn.Linear(hidden2, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# class AccidentDetector(nn.Module):
#     def __init__(self, input_dim=1024, hidden1=512, hidden2=256, num_classes=2, dropout=0.3):
#         super(AccidentDetector, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden1)
#         self.bn1 = nn.BatchNorm1d(hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.bn2 = nn.BatchNorm1d(hidden2)
#         self.fc3 = nn.Linear(hidden2, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# class SeverityClassifier(nn.Module):
#     def __init__(self, input_dim=512, hidden1=256, hidden2=128, num_classes=2, dropout=0.3):
#         super(SeverityClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden1)
#         self.bn1 = nn.BatchNorm1d(hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.bn2 = nn.BatchNorm1d(hidden2)
#         self.fc3 = nn.Linear(hidden2, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# # ═══════════════════════════════════════════════════════════
# # LOAD MODELS (CACHED)
# # ═══════════════════════════════════════════════════════════

# @st.cache_resource
# def load_all_models():
#     """Load all models - EXACT MATCH TO TRAINING"""
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     base_path = r"C:\Users\shiva\OneDrive\Desktop\cv_project_env"
#     models_path = os.path.join(base_path, "models")
    
#     # Load OpenCLIP (EXACT MATCH: laion2b_s34b_b79k)
#     clip_model, _, preprocess = open_clip.create_model_and_transforms(
#         'ViT-B-32',
#         pretrained='laion2b_s34b_b79k'  # ⚠️ CRITICAL: Match training!
#     )
#     clip_model = clip_model.to(device)
#     clip_model.eval()
    
#     tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
#     # Load BLIP
#     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
#     blip_model = blip_model.to(device)
#     blip_model.eval()
    
#     # Load Violence Detector
#     violence_model = ViolenceDetector().to(device)
#     violence_model.load_state_dict(
#         torch.load(os.path.join(models_path, "violence_agent_best.pth"),
#                   map_location=device, weights_only=True)
#     )
#     violence_model.eval()
    
#     # Load Accident Detector
#     accident_model = AccidentDetector().to(device)
#     accident_model.load_state_dict(
#         torch.load(os.path.join(models_path, "accident_agent_best.pth"),
#                   map_location=device, weights_only=True)
#     )
#     accident_model.eval()
    
#     # Load Severity Classifier
#     severity_model = SeverityClassifier().to(device)
#     severity_model.load_state_dict(
#         torch.load(os.path.join(models_path, "severity_classifier_improved.pth"),
#                   map_location=device, weights_only=True)
#     )
#     severity_model.eval()
    
#     return {
#         'device': device,
#         'clip_model': clip_model,
#         'preprocess': preprocess,
#         'tokenizer': tokenizer,
#         'blip_model': blip_model,
#         'blip_processor': blip_processor,
#         'violence_model': violence_model,
#         'accident_model': accident_model,
#         'severity_model': severity_model
#     }

# # ═══════════════════════════════════════════════════════════
# # FEATURE EXTRACTION - EXACT MATCH TO TRAINING
# # ═══════════════════════════════════════════════════════════

# def extract_features_from_frame(frame, models):
#     """
#     Extract features - EXACT MATCH TO TRAINING
    
#     ⚠️ CRITICAL:
#     - Image embedding: NORMALIZED
#     - Text embedding: NOT NORMALIZED
#     """
    
#     device = models['device']
    
#     # Convert to PIL
#     pil_image = Image.fromarray(frame.astype('uint8'))
    
#     # ═══════════════════════════════════════════════════════
#     # IMAGE EMBEDDING (NORMALIZED)
#     # ═══════════════════════════════════════════════════════
    
#     image_input = models['preprocess'](pil_image).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         image_features = models['clip_model'].encode_image(image_input)
#         # ⚠️ NORMALIZE IMAGE (matches training)
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
#     image_embedding = image_features.cpu().numpy()[0]
    
#     # ═══════════════════════════════════════════════════════
#     # GENERATE CAPTION
#     # ═══════════════════════════════════════════════════════
    
#     blip_inputs = models['blip_processor'](pil_image, return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         caption_ids = models['blip_model'].generate(**blip_inputs, max_length=50)
    
#     caption = models['blip_processor'].decode(caption_ids[0], skip_special_tokens=True)
    
#     # ═══════════════════════════════════════════════════════
#     # TEXT EMBEDDING (NOT NORMALIZED)
#     # ═══════════════════════════════════════════════════════
    
#     text_input = models['tokenizer']([caption]).to(device)
    
#     with torch.no_grad():
#         text_features = models['clip_model'].encode_text(text_input)
#         # ⚠️ NO NORMALIZATION FOR TEXT (matches training)
    
#     text_embedding = text_features.cpu().numpy()[0]
    
#     # Combine for multimodal
#     combined_features = np.concatenate([image_embedding, text_embedding])
    
#     return {
#         'image_embedding': image_embedding,
#         'text_embedding': text_embedding,
#         'combined_features': combined_features,
#         'caption': caption
#     }

# # ═══════════════════════════════════════════════════════════
# # PREDICTION FUNCTIONS
# # ═══════════════════════════════════════════════════════════

# def predict_violence(combined_features, model, device):
#     """Predict violence"""
    
#     model.eval()
#     features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         outputs = model(features_tensor)
#         probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
#     return {
#         'detected': int(np.argmax(probs)) == 1,
#         'confidence': float(probs[np.argmax(probs)]) * 100,
#         'probabilities': {
#             'safe': float(probs[0]) * 100,
#             'violence': float(probs[1]) * 100
#         }
#     }

# def predict_accident(combined_features, model, device):
#     """Predict accident"""
    
#     model.eval()
#     features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         outputs = model(features_tensor)
#         probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
#     return {
#         'detected': int(np.argmax(probs)) == 1,
#         'confidence': float(probs[np.argmax(probs)]) * 100,
#         'probabilities': {
#             'no_accident': float(probs[0]) * 100,
#             'accident': float(probs[1]) * 100
#         }
#     }

# def predict_severity(image_embedding, model, device):
#     """Predict severity (only vision)"""
    
#     model.eval()
#     image_tensor = torch.FloatTensor(image_embedding).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
#     severity_pred = int(np.argmax(probs))
#     severity_label = "LOW" if severity_pred == 0 else "HIGH"
#     severity_desc = "MINOR/MODERATE" if severity_pred == 0 else "MAJOR"
    
#     return {
#         'level': severity_label,
#         'description': severity_desc,
#         'confidence': float(probs[severity_pred]) * 100,
#         'probabilities': {
#             'low': float(probs[0]) * 100,
#             'high': float(probs[1]) * 100
#         }
#     }

# def extract_frames(video_path, fps=1):
#     """Extract frames from video"""
    
#     cap = cv2.VideoCapture(video_path)
#     video_fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_interval = max(1, int(video_fps / fps))
    
#     frames = []
#     frame_count = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         if frame_count % frame_interval == 0:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame_rgb)
        
#         frame_count += 1
    
#     cap.release()
#     return frames

# # ═══════════════════════════════════════════════════════════
# # MAIN APP
# # ═══════════════════════════════════════════════════════════

# def main():
    
#     st.markdown("<h1>🚨 Multi-Agent Emergency Detection System</h1>", unsafe_allow_html=True)
    
#     st.markdown("""
#     <div class="success-box">
#     <h3>🎯 System Capabilities</h3>
#     <ul>
#         <li><b>Violence Detection:</b> 99.90% accuracy</li>
#         <li><b>Accident Detection:</b> 99.91% accuracy</li>
#         <li><b>Severity Classification:</b> 83.81% accuracy</li>
#     </ul>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Sidebar
#     st.sidebar.title("⚙️ Configuration")
#     st.sidebar.markdown("---")
    
#     detection_mode = st.sidebar.selectbox(
#         "Select Detection Mode",
#         ["All Agents", "Violence Only", "Accident Only", "Accident + Severity"]
#     )
    
#     fps = st.sidebar.slider("Frames per Second", 1, 5, 1)
    
#     st.sidebar.markdown("---")
#     st.sidebar.info("""
#     **Model Performance:**
#     - Violence: 99.90%
#     - Accident: 99.91%
#     - Severity: 83.81%
#     """)
    
#     # Load models
#     with st.spinner("Loading AI models..."):
#         models = load_all_models()
    
#     st.success(f"✅ Models loaded on {models['device'].upper()}!")
    
#     # File uploader
#     st.markdown("### 📤 Upload Video")
#     uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
#     if uploaded_file is not None:
        
#         tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#         tfile.write(uploaded_file.read())
#         video_path = tfile.name
        
#         st.markdown("### 📹 Uploaded Video")
#         st.video(video_path)
        
#         if st.button("🚀 Analyze Video", type="primary"):
            
#             with st.spinner("Extracting frames..."):
#                 frames = extract_frames(video_path, fps)
            
#             st.success(f"✅ Extracted {len(frames)} frames")
            
#             results_violence = []
#             results_accident = []
#             results_severity = []
            
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             for idx, frame in enumerate(frames):
                
#                 status_text.text(f"Processing frame {idx+1}/{len(frames)}...")
                
#                 feature_data = extract_features_from_frame(frame, models)
                
#                 # Violence
#                 if detection_mode in ["All Agents", "Violence Only"]:
#                     violence_result = predict_violence(
#                         feature_data['combined_features'],
#                         models['violence_model'],
#                         models['device']
#                     )
#                     violence_result['caption'] = feature_data['caption']
#                     violence_result['frame_num'] = idx
#                     results_violence.append(violence_result)
                
#                 # Accident
#                 if detection_mode in ["All Agents", "Accident Only", "Accident + Severity"]:
#                     accident_result = predict_accident(
#                         feature_data['combined_features'],
#                         models['accident_model'],
#                         models['device']
#                     )
#                     accident_result['caption'] = feature_data['caption']
#                     accident_result['frame_num'] = idx
#                     results_accident.append(accident_result)
                    
#                     # Severity if accident
#                     if accident_result['detected'] and detection_mode in ["All Agents", "Accident + Severity"]:
#                         severity_result = predict_severity(
#                             feature_data['image_embedding'],
#                             models['severity_model'],
#                             models['device']
#                         )
#                         severity_result['caption'] = feature_data['caption']
#                         severity_result['frame_num'] = idx
#                         results_severity.append(severity_result)
                
#                 progress_bar.progress((idx + 1) / len(frames))
            
#             status_text.text("✅ Analysis complete!")
            
#             # Display Results
#             st.markdown("---")
#             st.markdown("## 📊 Analysis Results")
            
#             # Violence Results
#             if results_violence:
#                 st.markdown("### 🔴 Violence Detection")
                
#                 violence_count = sum(1 for r in results_violence if r['detected'])
#                 violence_pct = (violence_count / len(results_violence)) * 100
                
#                 col1, col2, col3 = st.columns(3)
#                 col1.metric("Frames Analyzed", len(results_violence))
#                 col2.metric("Violence Detected", violence_count)
#                 col3.metric("Violence %", f"{violence_pct:.1f}%")
                
#                 if violence_count > 0:
#                     st.markdown("""
#                     <div class="danger-box">
#                     <h4>⚠️ VIOLENCE DETECTED!</h4>
#                     <p>Violent activity detected. Immediate attention required.</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     for r in results_violence:
#                         if r['detected']:
#                             st.markdown(f"- **Frame {r['frame_num']}:** {r['confidence']:.1f}% - *{r['caption']}*")
#                 else:
#                     st.markdown("""
#                     <div class="success-box">
#                     <h4>✅ NO VIOLENCE DETECTED</h4>
#                     </div>
#                     """, unsafe_allow_html=True)
            
#             # Accident Results
#             if results_accident:
#                 st.markdown("### 🚗 Accident Detection")
                
#                 accident_count = sum(1 for r in results_accident if r['detected'])
#                 accident_pct = (accident_count / len(results_accident)) * 100
                
#                 col1, col2, col3 = st.columns(3)
#                 col1.metric("Frames Analyzed", len(results_accident))
#                 col2.metric("Accidents Detected", accident_count)
#                 col3.metric("Accident %", f"{accident_pct:.1f}%")
                
#                 if accident_count > 0:
#                     st.markdown("""
#                     <div class="danger-box">
#                     <h4>🚨 ACCIDENT DETECTED!</h4>
#                     <p>Traffic accident detected. Emergency response may be required.</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     for r in results_accident:
#                         if r['detected']:
#                             st.markdown(f"- **Frame {r['frame_num']}:** {r['confidence']:.1f}% - *{r['caption']}*")
#                 else:
#                     st.markdown("""
#                     <div class="success-box">
#                     <h4>✅ NO ACCIDENTS DETECTED</h4>
#                     </div>
#                     """, unsafe_allow_html=True)
            
#             # Severity Results
#             if results_severity:
#                 st.markdown("### ⚠️ Severity Classification")
                
#                 high_count = sum(1 for r in results_severity if r['level'] == 'HIGH')
#                 low_count = len(results_severity) - high_count
                
#                 col1, col2, col3 = st.columns(3)
#                 col1.metric("Total Accidents", len(results_severity))
#                 col2.metric("Low Severity", low_count)
#                 col3.metric("High Severity", high_count)
                
#                 if high_count > 0:
#                     st.markdown("""
#                     <div class="danger-box">
#                     <h4>🔴 MAJOR ACCIDENTS DETECTED</h4>
#                     <p>High severity. Immediate emergency response required.</p>
#                     </div>
#                     """, unsafe_allow_html=True)
#                 else:
#                     st.markdown("""
#                     <div class="warning-box">
#                     <h4>🟡 MINOR/MODERATE ACCIDENTS</h4>
#                     <p>Low severity. Monitor situation.</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 for r in results_severity:
#                     emoji = "🔴" if r['level'] == 'HIGH' else "🟡"
#                     st.markdown(f"{emoji} **Frame {r['frame_num']}:** {r['description']} ({r['confidence']:.1f}%)")
            
#             # Summary
#             st.markdown("---")
#             st.markdown("### 📈 Summary")
#             st.markdown(f"**Analysis Mode:** {detection_mode}")
#             st.markdown(f"**Frames Processed:** {len(frames)}")
#             st.markdown(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# if __name__ == "__main__":
#     main()





"""
═══════════════════════════════════════════════════════════
MULTI-AGENT EMERGENCY DETECTION SYSTEM
Smart Detection - Shows only relevant results

Violence code: UNCHANGED from working app.py
Accident & Severity: Added without touching violence
═══════════════════════════════════════════════════════════
"""

import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import open_clip
from transformers import BlipProcessor, BlipForConditionalGeneration
import tempfile
import os
from datetime import datetime

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Emergency Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    h1 {
        color: #ffffff;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: bold;
    }
    
    .success-box {
        background: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    
    .danger-box {
        background: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    
    .info-box {
        background: #d1ecf1;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════

# class ViolenceDetector(nn.Module):
#     """Violence Detection - UNCHANGED from working app.py"""
#     def __init__(self, input_dim=1024, hidden1=512, hidden2=256, num_classes=2, dropout=0.3):
#         super(ViolenceDetector, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden1)
#         self.bn1 = nn.BatchNorm1d(hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.bn2 = nn.BatchNorm1d(hidden2)
#         self.fc3 = nn.Linear(hidden2, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# class AccidentDetector(nn.Module):
#     def __init__(self, input_dim=1024, hidden1=512, hidden2=256, num_classes=2, dropout=0.3):
#         super(AccidentDetector, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden1)
#         self.bn1 = nn.BatchNorm1d(hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.bn2 = nn.BatchNorm1d(hidden2)
#         self.fc3 = nn.Linear(hidden2, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# class SeverityClassifier(nn.Module):
#     def __init__(self, input_dim=512, hidden1=256, hidden2=128, num_classes=2, dropout=0.3):
#         super(SeverityClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden1)
#         self.bn1 = nn.BatchNorm1d(hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.bn2 = nn.BatchNorm1d(hidden2)
#         self.fc3 = nn.Linear(hidden2, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# # ═══════════════════════════════════════════════════════════
# # LOAD MODELS - UNCHANGED from app.py
# # ═══════════════════════════════════════════════════════════

# @st.cache_resource
# def load_models():
#     """Load all models - EXACT from working app.py"""
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     base_path = r"C:\Users\shiva\OneDrive\Desktop\cv_project_env"
#     models_path = os.path.join(base_path, "models")
    
#     # OpenCLIP (EXACT from app.py)
#     clip_model, _, preprocess = open_clip.create_model_and_transforms(
#         'ViT-B-32',
#         pretrained='laion2b_s34b_b79k'
#     )
#     clip_model = clip_model.to(device)
#     clip_model.eval()
    
#     tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
#     # BLIP
#     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
#     blip_model = blip_model.to(device)
#     blip_model.eval()
    
#     # Violence Model
#     violence_model = ViolenceDetector().to(device)
#     violence_model.load_state_dict(
#         torch.load(os.path.join(models_path, "violence_agent_best.pth"),
#                   map_location=device, weights_only=True)
#     )
#     violence_model.eval()
    
#     # Accident Model
#     accident_model = AccidentDetector().to(device)
#     accident_model.load_state_dict(
#         torch.load(os.path.join(models_path, "accident_agent_best.pth"),
#                   map_location=device, weights_only=True)
#     )
#     accident_model.eval()
    
#     # Severity Model
#     severity_model = SeverityClassifier().to(device)
#     severity_model.load_state_dict(
#         torch.load(os.path.join(models_path, "severity_classifier_improved.pth"),
#                   map_location=device, weights_only=True)
#     )
#     severity_model.eval()
    
#     return {
#         'device': device,
#         'clip_model': clip_model,
#         'preprocess': preprocess,
#         'tokenizer': tokenizer,
#         'blip_model': blip_model,
#         'blip_processor': blip_processor,
#         'violence_model': violence_model,
#         'accident_model': accident_model,
#         'severity_model': severity_model
#     }

# # ═══════════════════════════════════════════════════════════
# # FEATURE EXTRACTION - EXACT from app.py (UNCHANGED)
# # ═══════════════════════════════════════════════════════════

# def extract_features_from_frame(frame, models):
#     """
#     Extract features - EXACT from working app.py
#     Image: NORMALIZED
#     Text: NOT NORMALIZED
#     """
    
#     device = models['device']
#     pil_image = Image.fromarray(frame.astype('uint8'))
    
#     # Image embedding (NORMALIZED)
#     image_input = models['preprocess'](pil_image).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         image_features = models['clip_model'].encode_image(image_input)
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
#     image_embedding = image_features.cpu().numpy()[0]
    
#     # Caption
#     blip_inputs = models['blip_processor'](pil_image, return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         caption_ids = models['blip_model'].generate(**blip_inputs, max_length=50)
    
#     caption = models['blip_processor'].decode(caption_ids[0], skip_special_tokens=True)
    
#     # Text embedding (NOT NORMALIZED)
#     text_input = models['tokenizer']([caption]).to(device)
    
#     with torch.no_grad():
#         text_features = models['clip_model'].encode_text(text_input)
    
#     text_embedding = text_features.cpu().numpy()[0]
    
#     # Combine
#     combined_features = np.concatenate([image_embedding, text_embedding])
    
#     return {
#         'image_embedding': image_embedding,
#         'text_embedding': text_embedding,
#         'combined_features': combined_features,
#         'caption': caption
#     }

# # ═══════════════════════════════════════════════════════════
# # PREDICTIONS - UNCHANGED from app.py
# # ═══════════════════════════════════════════════════════════

# def predict_violence(combined_features, model, device):
#     """Predict violence - EXACT from app.py"""
    
#     model.eval()
#     features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         outputs = model(features_tensor)
#         probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
#     return {
#         'prediction': int(np.argmax(probs)),
#         'confidence': float(probs[np.argmax(probs)]),
#         'violence_prob': float(probs[1]),
#         'safe_prob': float(probs[0])
#     }

# def predict_accident(combined_features, model, device):
#     """Predict accident"""
    
#     model.eval()
#     features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         outputs = model(features_tensor)
#         probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
#     return {
#         'prediction': int(np.argmax(probs)),
#         'confidence': float(probs[np.argmax(probs)]),
#         'accident_prob': float(probs[1]),
#         'safe_prob': float(probs[0])
#     }

# def predict_severity(image_embedding, model, device):
#     """Predict severity"""
    
#     model.eval()
#     image_tensor = torch.FloatTensor(image_embedding).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
#     severity_pred = int(np.argmax(probs))
    
#     return {
#         'level': "LOW" if severity_pred == 0 else "HIGH",
#         'description': "MINOR/MODERATE" if severity_pred == 0 else "MAJOR",
#         'confidence': float(probs[severity_pred]),
#         'low_prob': float(probs[0]),
#         'high_prob': float(probs[1])
#     }

# def extract_frames(video_path, fps=1):
#     """Extract frames - UNCHANGED"""
    
#     cap = cv2.VideoCapture(video_path)
#     video_fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_interval = max(1, int(video_fps / fps))
    
#     frames = []
#     frame_count = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         if frame_count % frame_interval == 0:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame_rgb)
        
#         frame_count += 1
    
#     cap.release()
#     return frames

# # ═══════════════════════════════════════════════════════════
# # MAIN APP - SMART DETECTION
# # ═══════════════════════════════════════════════════════════

# def main():
    
#     st.markdown("<h1>🚨 Multi-Agent Emergency Detection System</h1>", unsafe_allow_html=True)
    
#     st.markdown("""
#     <div class="info-box">
#     <h3>🎯 Smart Detection System</h3>
#     <p>Automatically detects and shows only relevant emergencies:</p>
#     <ul>
#         <li><b>Violence Detection:</b> 99.90% accuracy</li>
#         <li><b>Accident Detection:</b> 99.91% accuracy</li>
#         <li><b>Severity Classification:</b> 83.81% accuracy</li>
#     </ul>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Sidebar
#     st.sidebar.title("⚙️ Settings")
#     fps = st.sidebar.slider("Frames per Second", 1, 5, 1)
    
#     st.sidebar.markdown("---")
#     st.sidebar.info("""
#     **Model Performance:**
#     - Violence: 99.90%
#     - Accident: 99.91%
#     - Severity: 83.81%
#     """)
    
#     # Load models
#     with st.spinner("Loading AI models..."):
#         models = load_models()
    
#     st.success(f"✅ Models loaded on {models['device'].upper()}!")
    
#     # Upload
#     st.markdown("### 📤 Upload Video")
#     uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
#     if uploaded_file is not None:
        
#         tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#         tfile.write(uploaded_file.read())
#         video_path = tfile.name
        
#         st.markdown("### 📹 Uploaded Video")
#         st.video(video_path)
        
#         if st.button("🚀 Analyze Video", type="primary"):
            
#             with st.spinner("Extracting frames..."):
#                 frames = extract_frames(video_path, fps)
            
#             st.success(f"✅ Extracted {len(frames)} frames")
            
#             # Process all frames
#             all_violence = []
#             all_accident = []
#             all_severity = []
            
#             progress_bar = st.progress(0)
            
#             for idx, frame in enumerate(frames):
                
#                 feature_data = extract_features_from_frame(frame, models)
                
#                 # Violence prediction
#                 violence_result = predict_violence(
#                     feature_data['combined_features'],
#                     models['violence_model'],
#                     models['device']
#                 )
#                 violence_result['caption'] = feature_data['caption']
#                 violence_result['frame_num'] = idx
#                 all_violence.append(violence_result)
                
#                 # Accident prediction
#                 accident_result = predict_accident(
#                     feature_data['combined_features'],
#                     models['accident_model'],
#                     models['device']
#                 )
#                 accident_result['caption'] = feature_data['caption']
#                 accident_result['frame_num'] = idx
#                 all_accident.append(accident_result)
                
#                 # Severity if accident detected
#                 if accident_result['prediction'] == 1:
#                     severity_result = predict_severity(
#                         feature_data['image_embedding'],
#                         models['severity_model'],
#                         models['device']
#                     )
#                     severity_result['caption'] = feature_data['caption']
#                     severity_result['frame_num'] = idx
#                     all_severity.append(severity_result)
                
#                 progress_bar.progress((idx + 1) / len(frames))
            
#             # ═══════════════════════════════════════════════════════
#             # SMART DISPLAY - Show only what's detected!
#             # ═══════════════════════════════════════════════════════
            
#             st.markdown("---")
#             st.markdown("## 📊 Detection Results")
            
#             # Count detections
#             violence_count = sum(1 for r in all_violence if r['prediction'] == 1)
#             accident_count = sum(1 for r in all_accident if r['prediction'] == 1)
            
#             # Summary metrics
#             col1, col2, col3 = st.columns(3)
#             col1.metric("Frames Analyzed", len(frames))
#             col2.metric("Violence Detected", violence_count, delta=None if violence_count == 0 else "⚠️")
#             col3.metric("Accidents Detected", accident_count, delta=None if accident_count == 0 else "⚠️")
            
#             # ═══════════════════════════════════════════════════════
#             # VIOLENCE SECTION - Only show if detected
#             # ═══════════════════════════════════════════════════════
            
#             if violence_count > 0:
#                 st.markdown("---")
#                 st.markdown("### 🔴 Violence Detection")
                
#                 violence_pct = (violence_count / len(frames)) * 100
                
#                 st.markdown(f"""
#                 <div class="danger-box">
#                 <h4>⚠️ VIOLENCE DETECTED!</h4>
#                 <p><b>{violence_count} out of {len(frames)} frames ({violence_pct:.1f}%)</b></p>
#                 <p>Violent activity detected. Immediate attention required.</p>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 # Show violent frames
#                 st.markdown("#### 🎯 Violent Frames:")
#                 for r in all_violence:
#                     if r['prediction'] == 1:
#                         st.markdown(f"""
#                         - **Frame {r['frame_num']}:** Violence probability {r['violence_prob']*100:.1f}%
#                           - *Caption: {r['caption']}*
#                           - Confidence: {r['confidence']*100:.1f}%
#                         """)
            
#             # ═══════════════════════════════════════════════════════
#             # ACCIDENT SECTION - Only show if detected
#             # ═══════════════════════════════════════════════════════
            
#             if accident_count > 0:
#                 st.markdown("---")
#                 st.markdown("### 🚗 Accident Detection")
                
#                 accident_pct = (accident_count / len(frames)) * 100
                
#                 st.markdown(f"""
#                 <div class="danger-box">
#                 <h4>🚨 ACCIDENT DETECTED!</h4>
#                 <p><b>{accident_count} out of {len(frames)} frames ({accident_pct:.1f}%)</b></p>
#                 <p>Traffic accident detected. Emergency response may be required.</p>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 # Show accident frames
#                 st.markdown("#### 🎯 Accident Frames:")
#                 for r in all_accident:
#                     if r['prediction'] == 1:
#                         st.markdown(f"""
#                         - **Frame {r['frame_num']}:** Accident probability {r['accident_prob']*100:.1f}%
#                           - *Caption: {r['caption']}*
#                           - Confidence: {r['confidence']*100:.1f}%
#                         """)
                
#                 # ═══════════════════════════════════════════════════
#                 # SEVERITY SECTION - Only show if accidents detected
#                 # ═══════════════════════════════════════════════════
                
#                 if len(all_severity) > 0:
#                     st.markdown("---")
#                     st.markdown("### ⚠️ Severity Classification")
                    
#                     high_count = sum(1 for r in all_severity if r['level'] == 'HIGH')
#                     low_count = len(all_severity) - high_count
                    
#                     col1, col2, col3 = st.columns(3)
#                     col1.metric("Accident Frames", len(all_severity))
#                     col2.metric("Low Severity", low_count)
#                     col3.metric("High Severity", high_count)
                    
#                     if high_count > 0:
#                         st.markdown(f"""
#                         <div class="danger-box">
#                         <h4>🔴 MAJOR ACCIDENTS DETECTED</h4>
#                         <p><b>{high_count} high severity accident(s)</b></p>
#                         <p>Immediate emergency response required!</p>
#                         </div>
#                         """, unsafe_allow_html=True)
#                     else:
#                         st.markdown(f"""
#                         <div class="warning-box">
#                         <h4>🟡 MINOR/MODERATE ACCIDENTS</h4>
#                         <p><b>{low_count} low severity accident(s)</b></p>
#                         <p>Monitor the situation.</p>
#                         </div>
#                         """, unsafe_allow_html=True)
                    
#                     # Show severity details
#                     st.markdown("#### 📋 Severity Details:")
#                     for r in all_severity:
#                         emoji = "🔴" if r['level'] == 'HIGH' else "🟡"
#                         st.markdown(f"""
#                         {emoji} **Frame {r['frame_num']}:** {r['description']}
#                         - Confidence: {r['confidence']*100:.1f}%
#                         - *Caption: {r['caption']}*
#                         """)
            
#             # ═══════════════════════════════════════════════════════
#             # SAFE - Only show if nothing detected
#             # ═══════════════════════════════════════════════════════
            
#             if violence_count == 0 and accident_count == 0:
#                 st.markdown("---")
#                 st.markdown(f"""
#                 <div class="success-box">
#                 <h2>✅ ALL CLEAR - NO EMERGENCIES DETECTED</h2>
#                 <p><b>Analyzed {len(frames)} frames</b></p>
#                 <ul>
#                     <li>No violence detected (99.90% accuracy model)</li>
#                     <li>No accidents detected (99.91% accuracy model)</li>
#                 </ul>
#                 <p>Video appears safe.</p>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             # Summary
#             st.markdown("---")
#             st.markdown("### 📈 Analysis Summary")
#             st.markdown(f"**Frames Processed:** {len(frames)}")
#             st.markdown(f"**Processing FPS:** {fps}")
#             st.markdown(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
#             # Cleanup
#             try:
#                 os.unlink(video_path)
#             except:
#                 pass

# if __name__ == "__main__":
#     main()

# """
# ═══════════════════════════════════════════════════════════
# MULTI-AGENT EMERGENCY DETECTION SYSTEM
# User selects which agents to run

# Violence code: UNCHANGED from working app.py
# ═══════════════════════════════════════════════════════════
# """

import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import open_clip
from transformers import BlipProcessor, BlipForConditionalGeneration
import tempfile
import os
from datetime import datetime

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Emergency Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    h1 {
        color: #ffffff;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: bold;
    }
    
    .success-box {
        background: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    
    .danger-box {
        background: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    
    .info-box {
        background: #d1ecf1;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# # ═══════════════════════════════════════════════════════════
# # MODEL ARCHITECTURES
# # ═══════════════════════════════════════════════════════════

# class ViolenceDetector(nn.Module):
#     def __init__(self, input_dim=1024, hidden1=512, hidden2=256, num_classes=2, dropout=0.3):
#         super(ViolenceDetector, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden1)
#         self.bn1 = nn.BatchNorm1d(hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.bn2 = nn.BatchNorm1d(hidden2)
#         self.fc3 = nn.Linear(hidden2, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# class AccidentDetector(nn.Module):
#     def __init__(self, input_dim=1024, hidden1=512, hidden2=256, num_classes=2, dropout=0.3):
#         super(AccidentDetector, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden1)
#         self.bn1 = nn.BatchNorm1d(hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.bn2 = nn.BatchNorm1d(hidden2)
#         self.fc3 = nn.Linear(hidden2, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# class SeverityClassifier(nn.Module):
#     def __init__(self, input_dim=512, hidden1=256, hidden2=128, num_classes=2, dropout=0.3):
#         super(SeverityClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden1)
#         self.bn1 = nn.BatchNorm1d(hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.bn2 = nn.BatchNorm1d(hidden2)
#         self.fc3 = nn.Linear(hidden2, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# # ═══════════════════════════════════════════════════════════
# # LOAD MODELS - UNCHANGED
# # ═══════════════════════════════════════════════════════════

# @st.cache_resource
# def load_models():
#     """Load all models - EXACT from working app.py"""
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     base_path = r"C:\Users\shiva\OneDrive\Desktop\cv_project_env"
#     models_path = os.path.join(base_path, "models")
    
#     # OpenCLIP
#     clip_model, _, preprocess = open_clip.create_model_and_transforms(
#         'ViT-B-32',
#         pretrained='laion2b_s34b_b79k'
#     )
#     clip_model = clip_model.to(device)
#     clip_model.eval()
    
#     tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
#     # BLIP
#     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
#     blip_model = blip_model.to(device)
#     blip_model.eval()
    
#     # Violence Model
#     violence_model = ViolenceDetector().to(device)
#     violence_model.load_state_dict(
#         torch.load(os.path.join(models_path, "violence_agent_best.pth"),
#                   map_location=device, weights_only=True)
#     )
#     violence_model.eval()
    
#     # Accident Model
#     accident_model = AccidentDetector().to(device)
#     accident_model.load_state_dict(
#         torch.load(os.path.join(models_path, "accident_agent_best.pth"),
#                   map_location=device, weights_only=True)
#     )
#     accident_model.eval()
    
#     # Severity Model
#     severity_model = SeverityClassifier().to(device)
#     severity_model.load_state_dict(
#         torch.load(os.path.join(models_path, "severity_classifier_improved.pth"),
#                   map_location=device, weights_only=True)
#     )
#     severity_model.eval()
    
#     return {
#         'device': device,
#         'clip_model': clip_model,
#         'preprocess': preprocess,
#         'tokenizer': tokenizer,
#         'blip_model': blip_model,
#         'blip_processor': blip_processor,
#         'violence_model': violence_model,
#         'accident_model': accident_model,
#         'severity_model': severity_model
#     }

# # ═══════════════════════════════════════════════════════════
# # FEATURE EXTRACTION - UNCHANGED
# # ═══════════════════════════════════════════════════════════

# def extract_features_from_frame(frame, models):
#     """Extract features - EXACT from working app.py"""
    
#     device = models['device']
#     pil_image = Image.fromarray(frame.astype('uint8'))
    
#     # Image embedding (NORMALIZED)
#     image_input = models['preprocess'](pil_image).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         image_features = models['clip_model'].encode_image(image_input)
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
#     image_embedding = image_features.cpu().numpy()[0]
    
#     # Caption
#     blip_inputs = models['blip_processor'](pil_image, return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         caption_ids = models['blip_model'].generate(**blip_inputs, max_length=50)
    
#     caption = models['blip_processor'].decode(caption_ids[0], skip_special_tokens=True)
    
#     # Text embedding (NOT NORMALIZED)
#     text_input = models['tokenizer']([caption]).to(device)
    
#     with torch.no_grad():
#         text_features = models['clip_model'].encode_text(text_input)
    
#     text_embedding = text_features.cpu().numpy()[0]
    
#     # Combine
#     combined_features = np.concatenate([image_embedding, text_embedding])
    
#     return {
#         'image_embedding': image_embedding,
#         'text_embedding': text_embedding,
#         'combined_features': combined_features,
#         'caption': caption
#     }

# # ═══════════════════════════════════════════════════════════
# # PREDICTIONS - UNCHANGED
# # ═══════════════════════════════════════════════════════════

# def predict_violence(combined_features, model, device):
#     """Predict violence - EXACT from app.py"""
    
#     model.eval()
#     features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         outputs = model(features_tensor)
#         probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
#     return {
#         'prediction': int(np.argmax(probs)),
#         'confidence': float(probs[np.argmax(probs)]),
#         'violence_prob': float(probs[1]),
#         'safe_prob': float(probs[0])
#     }

# def predict_accident(combined_features, model, device):
#     """Predict accident"""
    
#     model.eval()
#     features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         outputs = model(features_tensor)
#         probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
#     return {
#         'prediction': int(np.argmax(probs)),
#         'confidence': float(probs[np.argmax(probs)]),
#         'accident_prob': float(probs[1]),
#         'safe_prob': float(probs[0])
#     }

# def predict_severity(image_embedding, model, device):
#     """Predict severity"""
    
#     model.eval()
#     image_tensor = torch.FloatTensor(image_embedding).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
#     severity_pred = int(np.argmax(probs))
    
#     return {
#         'level': "LOW" if severity_pred == 0 else "HIGH",
#         'description': "MINOR/MODERATE" if severity_pred == 0 else "MAJOR",
#         'confidence': float(probs[severity_pred]),
#         'low_prob': float(probs[0]),
#         'high_prob': float(probs[1])
#     }

# def extract_frames(video_path, fps=1):
#     """Extract frames"""
    
#     cap = cv2.VideoCapture(video_path)
#     video_fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_interval = max(1, int(video_fps / fps))
    
#     frames = []
#     frame_count = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         if frame_count % frame_interval == 0:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame_rgb)
        
#         frame_count += 1
    
#     cap.release()
#     return frames

# # ═══════════════════════════════════════════════════════════
# # MAIN APP
# # ═══════════════════════════════════════════════════════════

# def main():
    
#     st.markdown("<h1>🚨 Multi-Agent Emergency Detection System</h1>", unsafe_allow_html=True)
    
#     st.markdown("""
#     <div class="info-box">
#     <h3>🎯 Select Detection Mode</h3>
#     <p>Choose which emergency type to detect in your video</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # ═══════════════════════════════════════════════════════
#     # DETECTION MODE SELECTION
#     # ═══════════════════════════════════════════════════════
    
#     st.sidebar.title("⚙️ Configuration")
#     st.sidebar.markdown("---")
    
#     detection_mode = st.sidebar.radio(
#         "🎯 Select Detection Type:",
#         [
#             "Violence Detection Only",
#             "Accident Detection Only", 
#             "Accident + Severity Analysis",
#             "All Agents (Violence + Accident + Severity)"
#         ],
#         help="Choose which agents to run on your video"
#     )
    
#     fps = st.sidebar.slider("Frames per Second", 1, 5, 1)
    
#     st.sidebar.markdown("---")
#     st.sidebar.info(f"""
#     **Active Mode:** 
#     {detection_mode}
    
#     **Model Performance:**
#     - Violence: 99.90%
#     - Accident: 99.91%
#     - Severity: 83.81%
#     """)
    
#     # Load models
#     with st.spinner("Loading AI models..."):
#         models = load_models()
    
#     st.success(f"✅ Models loaded on {models['device'].upper()}!")
    
#     # Upload
#     st.markdown("### 📤 Upload Video")
#     uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
#     if uploaded_file is not None:
        
#         tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#         tfile.write(uploaded_file.read())
#         video_path = tfile.name
        
#         st.markdown("### 📹 Uploaded Video")
#         st.video(video_path)
        
#         if st.button("🚀 Analyze Video", type="primary"):
            
#             with st.spinner("Extracting frames..."):
#                 frames = extract_frames(video_path, fps)
            
#             st.success(f"✅ Extracted {len(frames)} frames")
            
#             # Results storage
#             all_violence = []
#             all_accident = []
#             all_severity = []
            
#             progress_bar = st.progress(0)
            
#             # ═══════════════════════════════════════════════════════
#             # PROCESS BASED ON MODE
#             # ═══════════════════════════════════════════════════════
            
#             for idx, frame in enumerate(frames):
                
#                 feature_data = extract_features_from_frame(frame, models)
                
#                 # Violence (only if mode includes it)
#                 if detection_mode in ["Violence Detection Only", "All Agents (Violence + Accident + Severity)"]:
#                     violence_result = predict_violence(
#                         feature_data['combined_features'],
#                         models['violence_model'],
#                         models['device']
#                     )
#                     violence_result['caption'] = feature_data['caption']
#                     violence_result['frame_num'] = idx
#                     all_violence.append(violence_result)
                
#                 # Accident (only if mode includes it)
#                 if detection_mode in ["Accident Detection Only", "Accident + Severity Analysis", "All Agents (Violence + Accident + Severity)"]:
#                     accident_result = predict_accident(
#                         feature_data['combined_features'],
#                         models['accident_model'],
#                         models['device']
#                     )
#                     accident_result['caption'] = feature_data['caption']
#                     accident_result['frame_num'] = idx
#                     all_accident.append(accident_result)
                    
#                     # Severity (only if accident mode AND accident detected)
#                     if detection_mode in ["Accident + Severity Analysis", "All Agents (Violence + Accident + Severity)"]:
#                         if accident_result['prediction'] == 1:
#                             severity_result = predict_severity(
#                                 feature_data['image_embedding'],
#                                 models['severity_model'],
#                                 models['device']
#                             )
#                             severity_result['caption'] = feature_data['caption']
#                             severity_result['frame_num'] = idx
#                             all_severity.append(severity_result)
                
#                 progress_bar.progress((idx + 1) / len(frames))
            
#             # ═══════════════════════════════════════════════════════
#             # DISPLAY RESULTS
#             # ═══════════════════════════════════════════════════════
            
#             st.markdown("---")
#             st.markdown("## 📊 Detection Results")
            
#             # ═══ VIOLENCE RESULTS ═══
#             if len(all_violence) > 0:
#                 st.markdown("### 🔴 Violence Detection")
                
#                 violence_count = sum(1 for r in all_violence if r['prediction'] == 1)
#                 violence_pct = (violence_count / len(all_violence)) * 100
                
#                 col1, col2, col3 = st.columns(3)
#                 col1.metric("Frames Analyzed", len(all_violence))
#                 col2.metric("Violence Detected", violence_count)
#                 col3.metric("Violence %", f"{violence_pct:.1f}%")
                
#                 if violence_count > 0:
#                     st.markdown(f"""
#                     <div class="danger-box">
#                     <h4>⚠️ VIOLENCE DETECTED!</h4>
#                     <p><b>{violence_count} out of {len(frames)} frames ({violence_pct:.1f}%)</b></p>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     st.markdown("#### 🎯 Violent Frames:")
#                     for r in all_violence:
#                         if r['prediction'] == 1:
#                             st.markdown(f"""
#                             - **Frame {r['frame_num']}:** {r['violence_prob']*100:.1f}% probability
#                               - *{r['caption']}*
#                             """)
#                 else:
#                     st.markdown("""
#                     <div class="success-box">
#                     <h4>✅ NO VIOLENCE DETECTED</h4>
#                     </div>
#                     """, unsafe_allow_html=True)
            
#             # ═══ ACCIDENT RESULTS ═══
#             if len(all_accident) > 0:
#                 st.markdown("---")
#                 st.markdown("### 🚗 Accident Detection")
                
#                 accident_count = sum(1 for r in all_accident if r['prediction'] == 1)
#                 accident_pct = (accident_count / len(all_accident)) * 100
                
#                 col1, col2, col3 = st.columns(3)
#                 col1.metric("Frames Analyzed", len(all_accident))
#                 col2.metric("Accidents Detected", accident_count)
#                 col3.metric("Accident %", f"{accident_pct:.1f}%")
                
#                 if accident_count > 0:
#                     st.markdown(f"""
#                     <div class="danger-box">
#                     <h4>🚨 ACCIDENT DETECTED!</h4>
#                     <p><b>{accident_count} out of {len(frames)} frames ({accident_pct:.1f}%)</b></p>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     st.markdown("#### 🎯 Accident Frames:")
#                     for r in all_accident:
#                         if r['prediction'] == 1:
#                             st.markdown(f"""
#                             - **Frame {r['frame_num']}:** {r['accident_prob']*100:.1f}% probability
#                               - *{r['caption']}*
#                             """)
#                 else:
#                     st.markdown("""
#                     <div class="success-box">
#                     <h4>✅ NO ACCIDENTS DETECTED</h4>
#                     </div>
#                     """, unsafe_allow_html=True)
            
#             # ═══ SEVERITY RESULTS ═══
#             if len(all_severity) > 0:
#                 st.markdown("---")
#                 st.markdown("### ⚠️ Severity Classification")
                
#                 high_count = sum(1 for r in all_severity if r['level'] == 'HIGH')
#                 low_count = len(all_severity) - high_count
                
#                 col1, col2, col3 = st.columns(3)
#                 col1.metric("Accident Frames", len(all_severity))
#                 col2.metric("Low Severity", low_count)
#                 col3.metric("High Severity", high_count)
                
#                 if high_count > 0:
#                     st.markdown(f"""
#                     <div class="danger-box">
#                     <h4>🔴 MAJOR ACCIDENTS</h4>
#                     <p>{high_count} high severity</p>
#                     </div>
#                     """, unsafe_allow_html=True)
#                 else:
#                     st.markdown(f"""
#                     <div class="warning-box">
#                     <h4>🟡 MINOR/MODERATE</h4>
#                     <p>{low_count} low severity</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 st.markdown("#### 📋 Details:")
#                 for r in all_severity:
#                     emoji = "🔴" if r['level'] == 'HIGH' else "🟡"
#                     st.markdown(f"{emoji} **Frame {r['frame_num']}:** {r['description']} ({r['confidence']*100:.1f}%)")
            
#             # Summary
#             st.markdown("---")
#             st.markdown("### 📈 Summary")
#             st.markdown(f"**Mode:** {detection_mode}")
#             st.markdown(f"**Frames:** {len(frames)}")
#             st.markdown(f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
#             try:
#                 os.unlink(video_path)
#             except:
#                 pass

# if __name__ == "__main__":
#     main()






# """
# ═══════════════════════════════════════════════════════════
# MULTI-AGENT EMERGENCY DETECTION SYSTEM
# Violence code: EXACT COPY from working app.py (NO CHANGES!)
# ═══════════════════════════════════════════════════════════
# """

# import streamlit as st
# import torch
# import torch.nn as nn
# import cv2
# import numpy as np
# from PIL import Image
# import open_clip
# from transformers import BlipProcessor, BlipForConditionalGeneration
# import tempfile
# import os
# from datetime import datetime

# # ═══════════════════════════════════════════════════════════
# # PAGE CONFIG
# # ═══════════════════════════════════════════════════════════

# st.set_page_config(
#     page_title="Emergency Detection System",
#     page_icon="🚨",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ═══════════════════════════════════════════════════════════
# # CUSTOM CSS
# # ═══════════════════════════════════════════════════════════

# st.markdown("""
# <style>
#     .main {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     }
    
#     h1 {
#         color: #ffffff;
#         text-align: center;
#         padding: 20px;
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         border-radius: 10px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
    
#     .stButton>button {
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         padding: 15px 30px;
#         border-radius: 25px;
#         font-size: 16px;
#         font-weight: bold;
#     }
    
#     .success-box {
#         background: #d4edda;
#         padding: 20px;
#         border-radius: 10px;
#         border-left: 5px solid #28a745;
#         margin: 10px 0;
#     }
    
#     .danger-box {
#         background: #f8d7da;
#         padding: 20px;
#         border-radius: 10px;
#         border-left: 5px solid #dc3545;
#         margin: 10px 0;
#     }
    
#     .warning-box {
#         background: #fff3cd;
#         padding: 20px;
#         border-radius: 10px;
#         border-left: 5px solid #ffc107;
#         margin: 10px 0;
#     }
    
#     .info-box {
#         background: #d1ecf1;
#         padding: 20px;
#         border-radius: 10px;
#         border-left: 5px solid #17a2b8;
#         margin: 10px 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ═══════════════════════════════════════════════════════════
# # MODEL ARCHITECTURES
# # ═══════════════════════════════════════════════════════════

# class ViolenceDetector(nn.Module):
#     def __init__(self, input_dim=1024, hidden1=512, hidden2=256, num_classes=2, dropout=0.3):
#         super(ViolenceDetector, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden1)
#         self.bn1 = nn.BatchNorm1d(hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.bn2 = nn.BatchNorm1d(hidden2)
#         self.fc3 = nn.Linear(hidden2, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# class AccidentDetector(nn.Module):
#     def __init__(self, input_dim=1024, hidden1=512, hidden2=256, num_classes=2, dropout=0.3):
#         super(AccidentDetector, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden1)
#         self.bn1 = nn.BatchNorm1d(hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.bn2 = nn.BatchNorm1d(hidden2)
#         self.fc3 = nn.Linear(hidden2, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# class SeverityClassifier(nn.Module):
#     def __init__(self, input_dim=512, hidden1=256, hidden2=128, num_classes=2, dropout=0.3):
#         super(SeverityClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden1)
#         self.bn1 = nn.BatchNorm1d(hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.bn2 = nn.BatchNorm1d(hidden2)
#         self.fc3 = nn.Linear(hidden2, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# # ═══════════════════════════════════════════════════════════
# # LOAD MODELS - EXACT FROM APP.PY
# # ═══════════════════════════════════════════════════════════

# @st.cache_resource
# def load_models():
#     """Load all models - EXACT from app.py"""
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     base_path = r"C:\Users\shiva\OneDrive\Desktop\cv_project_env"
#     models_path = os.path.join(base_path, "models")
    
#     # OpenCLIP (EXACT from app.py)
#     clip_model, _, preprocess = open_clip.create_model_and_transforms(
#         'ViT-B-32',
#         pretrained='laion2b_s34b_b79k'
#     )
#     clip_model = clip_model.to(device)
#     clip_model.eval()
    
#     tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
#     # BLIP
#     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
#     blip_model = blip_model.to(device)
#     blip_model.eval()
    
#     # Violence Model
#     violence_model = ViolenceDetector().to(device)
#     violence_model.load_state_dict(
#         torch.load(os.path.join(models_path, "violence_agent_best.pth"),
#                   map_location=device, weights_only=True)
#     )
#     violence_model.eval()
    
#     # Accident Model
#     accident_model = AccidentDetector().to(device)
#     accident_model.load_state_dict(
#         torch.load(os.path.join(models_path, "accident_agent_best.pth"),
#                   map_location=device, weights_only=True)
#     )
#     accident_model.eval()
    
#     # Severity Model
#     severity_model = SeverityClassifier().to(device)
#     severity_model.load_state_dict(
#         torch.load(os.path.join(models_path, "severity_classifier_improved.pth"),
#                   map_location=device, weights_only=True)
#     )
#     severity_model.eval()
    
#     return {
#         'device': device,
#         'clip_model': clip_model,
#         'preprocess': preprocess,
#         'tokenizer': tokenizer,
#         'blip_model': blip_model,
#         'blip_processor': blip_processor,
#         'violence_model': violence_model,
#         'accident_model': accident_model,
#         'severity_model': severity_model
#     }

# # ═══════════════════════════════════════════════════════════
# # FEATURE EXTRACTION - EXACT COPY FROM APP.PY (NO NORMALIZATION!)
# # ═══════════════════════════════════════════════════════════

# def extract_features_from_frame(frame, models):
#     """
#     Extract features WITHOUT normalization
#     EXACT COPY FROM WORKING APP.PY - NO CHANGES!
#     """
    
#     device = models['device']
    
#     if isinstance(frame, np.ndarray):
#         pil_image = Image.fromarray(frame.astype('uint8'))
#     else:
#         pil_image = frame
    
#     # ═══════════════════════════════════════════════════════
#     # IMAGE EMBEDDING (NO NORMALIZATION!)
#     # ═══════════════════════════════════════════════════════
    
#     image_input = models['preprocess'](pil_image).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         image_features = models['clip_model'].encode_image(image_input)
#         # NO NORMALIZATION HERE!
    
#     image_embedding = image_features.cpu().numpy()[0]
    
#     # ═══════════════════════════════════════════════════════
#     # CAPTION
#     # ═══════════════════════════════════════════════════════
    
#     blip_inputs = models['blip_processor'](pil_image, return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         caption_ids = models['blip_model'].generate(**blip_inputs, max_length=50)
    
#     caption = models['blip_processor'].decode(caption_ids[0], skip_special_tokens=True)
    
#     # ═══════════════════════════════════════════════════════
#     # TEXT EMBEDDING (NO NORMALIZATION!)
#     # ═══════════════════════════════════════════════════════
    
#     text_input = models['tokenizer']([caption]).to(device)
    
#     with torch.no_grad():
#         text_features = models['clip_model'].encode_text(text_input)
#         # NO NORMALIZATION HERE!
    
#     text_embedding = text_features.cpu().numpy()[0]
    
#     # ═══════════════════════════════════════════════════════
#     # COMBINE
#     # ═══════════════════════════════════════════════════════
    
#     combined_features = np.concatenate([image_embedding, text_embedding])
    
#     img_norm = np.linalg.norm(image_embedding)
#     text_norm = np.linalg.norm(text_embedding)
    
#     return {
#         'combined_features': combined_features,
#         'caption': caption,
#         'img_norm': float(img_norm),
#         'text_norm': float(text_norm),
#         'image_embedding': image_embedding,
#         'text_embedding': text_embedding
#     }

# # ═══════════════════════════════════════════════════════════
# # PREDICTIONS - EXACT FROM APP.PY
# # ═══════════════════════════════════════════════════════════

# def predict_violence(combined_features, model, device):
#     """Predict violence - EXACT from app.py"""
    
#     model.eval()
#     features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         outputs = model(features_tensor)
#         probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
#     return {
#         'prediction': int(np.argmax(probs)),
#         'confidence': float(probs[np.argmax(probs)]),
#         'violence_prob': float(probs[1]),
#         'safe_prob': float(probs[0])
#     }

# def predict_accident(combined_features, model, device):
#     """Predict accident"""
    
#     model.eval()
#     features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         outputs = model(features_tensor)
#         probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
#     return {
#         'prediction': int(np.argmax(probs)),
#         'confidence': float(probs[np.argmax(probs)]),
#         'accident_prob': float(probs[1]),
#         'safe_prob': float(probs[0])
#     }

# def predict_severity(image_embedding, model, device):
#     """Predict severity - uses only image embedding"""
    
#     model.eval()
#     image_tensor = torch.FloatTensor(image_embedding).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
#     severity_pred = int(np.argmax(probs))
    
#     return {
#         'level': "LOW" if severity_pred == 0 else "HIGH",
#         'description': "MINOR/MODERATE" if severity_pred == 0 else "MAJOR",
#         'confidence': float(probs[severity_pred]),
#         'low_prob': float(probs[0]),
#         'high_prob': float(probs[1])
#     }

# def extract_frames(video_path, fps=1):
#     """Extract frames - EXACT from app.py"""
    
#     cap = cv2.VideoCapture(video_path)
#     video_fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_interval = max(1, int(video_fps / fps))
    
#     frames = []
#     frame_count = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         if frame_count % frame_interval == 0:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame_rgb)
        
#         frame_count += 1
    
#     cap.release()
#     return frames

# # ═══════════════════════════════════════════════════════════
# # MAIN APP
# # ═══════════════════════════════════════════════════════════

# def main():
    
#     st.markdown("<h1>🚨 Multi-Agent Emergency Detection System</h1>", unsafe_allow_html=True)
    
#     st.markdown("""
#     <div class="info-box">
#     <h3>🎯 Select Detection Mode</h3>
#     <p>Choose which emergency type to detect in your video</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # ═══════════════════════════════════════════════════════
#     # SIDEBAR
#     # ═══════════════════════════════════════════════════════
    
#     st.sidebar.title("⚙️ Configuration")
#     st.sidebar.markdown("---")
    
#     detection_mode = st.sidebar.radio(
#         "🎯 Select Detection Type:",
#         [
#             "Violence Detection Only",
#             "Accident Detection Only", 
#             "Accident + Severity Analysis",
#             "All Agents (Violence + Accident + Severity)"
#         ],
#         help="Choose which agents to run"
#     )
    
#     fps = st.sidebar.slider("Frames per Second", 1, 5, 1)
    
#     st.sidebar.markdown("---")
#     st.sidebar.info(f"""
#     **Active Mode:** 
#     {detection_mode}
    
#     **Model Performance:**
#     - Violence: 99.90%
#     - Accident: 99.91%
#     - Severity: 83.81%
#     """)
    
#     # Load models
#     with st.spinner("Loading AI models..."):
#         models = load_models()
    
#     st.success(f"✅ Models loaded on {models['device'].upper()}!")
    
#     # Upload
#     st.markdown("### 📤 Upload Video")
#     uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
#     if uploaded_file is not None:
        
#         tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#         tfile.write(uploaded_file.read())
#         video_path = tfile.name
        
#         st.markdown("### 📹 Uploaded Video")
#         st.video(video_path)
        
#         if st.button("🚀 Analyze Video", type="primary"):
            
#             with st.spinner("Extracting frames..."):
#                 frames = extract_frames(video_path, fps)
            
#             st.success(f"✅ Extracted {len(frames)} frames")
            
#             # Results storage
#             all_violence = []
#             all_accident = []
#             all_severity = []
            
#             progress_bar = st.progress(0)
            
#             # ═══════════════════════════════════════════════════════
#             # PROCESS BASED ON MODE
#             # ═══════════════════════════════════════════════════════
            
#             for idx, frame in enumerate(frames):
                
#                 feature_data = extract_features_from_frame(frame, models)
                
#                 # Violence (only if selected)
#                 if detection_mode in ["Violence Detection Only", "All Agents (Violence + Accident + Severity)"]:
#                     violence_result = predict_violence(
#                         feature_data['combined_features'],
#                         models['violence_model'],
#                         models['device']
#                     )
#                     violence_result['caption'] = feature_data['caption']
#                     violence_result['frame_num'] = idx
#                     violence_result['img_norm'] = feature_data['img_norm']
#                     violence_result['text_norm'] = feature_data['text_norm']
#                     all_violence.append(violence_result)
                
#                 # Accident (only if selected)
#                 if detection_mode in ["Accident Detection Only", "Accident + Severity Analysis", "All Agents (Violence + Accident + Severity)"]:
#                     accident_result = predict_accident(
#                         feature_data['combined_features'],
#                         models['accident_model'],
#                         models['device']
#                     )
#                     accident_result['caption'] = feature_data['caption']
#                     accident_result['frame_num'] = idx
#                     all_accident.append(accident_result)
                    
#                     # Severity (only if accident mode AND accident detected)
#                     if detection_mode in ["Accident + Severity Analysis", "All Agents (Violence + Accident + Severity)"]:
#                         if accident_result['prediction'] == 1:
#                             severity_result = predict_severity(
#                                 feature_data['image_embedding'],
#                                 models['severity_model'],
#                                 models['device']
#                             )
#                             severity_result['caption'] = feature_data['caption']
#                             severity_result['frame_num'] = idx
#                             all_severity.append(severity_result)
                
#                 progress_bar.progress((idx + 1) / len(frames))
            
#             # ═══════════════════════════════════════════════════════
#             # DISPLAY RESULTS
#             # ═══════════════════════════════════════════════════════
            
#             st.markdown("---")
#             st.markdown("## 📊 Detection Results")
            
#             # ═══ VIOLENCE RESULTS ═══
#             if len(all_violence) > 0:
#                 st.markdown("### 🔴 Violence Detection")
                
#                 violence_count = sum(1 for r in all_violence if r['prediction'] == 1)
#                 violence_pct = (violence_count / len(all_violence)) * 100
                
#                 col1, col2, col3 = st.columns(3)
#                 col1.metric("Frames Analyzed", len(all_violence))
#                 col2.metric("Violence Detected", violence_count)
#                 col3.metric("Violence %", f"{violence_pct:.1f}%")
                
#                 if violence_count > 0:
#                     st.markdown(f"""
#                     <div class="danger-box">
#                     <h4>⚠️ VIOLENCE DETECTED!</h4>
#                     <p><b>{violence_count} out of {len(frames)} frames ({violence_pct:.1f}%)</b></p>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     st.markdown("#### 🎯 Violent Frames:")
#                     for r in all_violence:
#                         if r['prediction'] == 1:
#                             st.markdown(f"""
#                             - **Frame {r['frame_num']}:** {r['violence_prob']*100:.1f}% probability (Confidence: {r['confidence']*100:.1f}%)
#                               - *{r['caption']}*
#                               - Img norm: {r['img_norm']:.2f}, Text norm: {r['text_norm']:.2f}
#                             """)
#                 else:
#                     st.markdown("""
#                     <div class="success-box">
#                     <h4>✅ NO VIOLENCE DETECTED</h4>
#                     </div>
#                     """, unsafe_allow_html=True)
            
#             # ═══ ACCIDENT RESULTS ═══
#             if len(all_accident) > 0:
#                 st.markdown("---")
#                 st.markdown("### 🚗 Accident Detection")
                
#                 accident_count = sum(1 for r in all_accident if r['prediction'] == 1)
#                 accident_pct = (accident_count / len(all_accident)) * 100
                
#                 col1, col2, col3 = st.columns(3)
#                 col1.metric("Frames Analyzed", len(all_accident))
#                 col2.metric("Accidents Detected", accident_count)
#                 col3.metric("Accident %", f"{accident_pct:.1f}%")
                
#                 if accident_count > 0:
#                     st.markdown(f"""
#                     <div class="danger-box">
#                     <h4>🚨 ACCIDENT DETECTED!</h4>
#                     <p><b>{accident_count} out of {len(frames)} frames ({accident_pct:.1f}%)</b></p>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     st.markdown("#### 🎯 Accident Frames:")
#                     for r in all_accident:
#                         if r['prediction'] == 1:
#                             st.markdown(f"""
#                             - **Frame {r['frame_num']}:** {r['accident_prob']*100:.1f}% probability
#                               - *{r['caption']}*
#                             """)
#                 else:
#                     st.markdown("""
#                     <div class="success-box">
#                     <h4>✅ NO ACCIDENTS DETECTED</h4>
#                     </div>
#                     """, unsafe_allow_html=True)
            
#             # ═══ SEVERITY RESULTS ═══
#             if len(all_severity) > 0:
#                 st.markdown("---")
#                 st.markdown("### ⚠️ Severity Classification")
                
#                 high_count = sum(1 for r in all_severity if r['level'] == 'HIGH')
#                 low_count = len(all_severity) - high_count
                
#                 col1, col2, col3 = st.columns(3)
#                 col1.metric("Accident Frames", len(all_severity))
#                 col2.metric("Low Severity", low_count)
#                 col3.metric("High Severity", high_count)
                
#                 if high_count > 0:
#                     st.markdown(f"""
#                     <div class="danger-box">
#                     <h4>🔴 MAJOR ACCIDENTS</h4>
#                     <p>{high_count} high severity</p>
#                     </div>
#                     """, unsafe_allow_html=True)
#                 else:
#                     st.markdown(f"""
#                     <div class="warning-box">
#                     <h4>🟡 MINOR/MODERATE</h4>
#                     <p>{low_count} low severity</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 st.markdown("#### 📋 Details:")
#                 for r in all_severity:
#                     emoji = "🔴" if r['level'] == 'HIGH' else "🟡"
#                     st.markdown(f"{emoji} **Frame {r['frame_num']}:** {r['description']} ({r['confidence']*100:.1f}%)")
            
#             # Summary
#             st.markdown("---")
#             st.markdown("### 📈 Summary")
#             st.markdown(f"**Mode:** {detection_mode}")
#             st.markdown(f"**Frames:** {len(frames)}")
#             st.markdown(f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
#             try:
#                 os.unlink(video_path)
#             except:
#                 pass

# if __name__ == "__main__":
#     main()


# """
# ═══════════════════════════════════════════════════════════
# MULTI-AGENT EMERGENCY DETECTION SYSTEM
# Optimized UI - Compact video display
# ═══════════════════════════════════════════════════════════
# """

import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import open_clip
from transformers import BlipProcessor, BlipForConditionalGeneration
import tempfile
import os
from datetime import datetime

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Emergency Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════
# CUSTOM CSS - OPTIMIZED
# ═══════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    h1 {
        color: #ffffff;
        text-align: center;
        padding: 15px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    h2 {
        color: #667eea;
        border-bottom: 2px solid #667eea;
        padding-bottom: 8px;
        margin-top: 15px;
    }
    
    h3 {
        color: #764ba2;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 25px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
    }
    
    .success-box {
        background: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 8px 0;
    }
    
    .danger-box {
        background: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 8px 0;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 8px 0;
    }
    
    .info-box {
        background: #d1ecf1;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 8px 0;
    }
    
    /* Compact video player */
    .stVideo {
        max-height: 400px !important;
    }
    
    /* Reduce spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Compact expander */
    .streamlit-expanderHeader {
        font-size: 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# MODEL ARCHITECTURES (SAME AS BEFORE)
# ═══════════════════════════════════════════════════════════

class ViolenceDetector(nn.Module):
    def __init__(self, input_dim=1024, hidden1=512, hidden2=256, num_classes=2, dropout=0.3):
        super(ViolenceDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class AccidentDetector(nn.Module):
    def __init__(self, input_dim=1024, hidden1=512, hidden2=256, num_classes=2, dropout=0.3):
        super(AccidentDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class SeverityClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden1=256, hidden2=128, num_classes=2, dropout=0.3):
        super(SeverityClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ═══════════════════════════════════════════════════════════
# LOAD MODELS (SAME AS BEFORE)
# ═══════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = r"C:\Users\shiva\OneDrive\Desktop\cv_project_env"
    models_path = os.path.join(base_path, "models")
    
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = blip_model.to(device)
    blip_model.eval()
    
    violence_model = ViolenceDetector().to(device)
    violence_model.load_state_dict(
        torch.load(os.path.join(models_path, "violence_agent_best.pth"),
                  map_location=device, weights_only=True)
    )
    violence_model.eval()
    
    accident_model = AccidentDetector().to(device)
    accident_model.load_state_dict(
        torch.load(os.path.join(models_path, "accident_agent_best.pth"),
                  map_location=device, weights_only=True)
    )
    accident_model.eval()
    
    severity_model = SeverityClassifier().to(device)
    severity_model.load_state_dict(
        torch.load(os.path.join(models_path, "severity_classifier_improved.pth"),
                  map_location=device, weights_only=True)
    )
    severity_model.eval()
    
    return {
        'device': device,
        'clip_model': clip_model,
        'preprocess': preprocess,
        'tokenizer': tokenizer,
        'blip_model': blip_model,
        'blip_processor': blip_processor,
        'violence_model': violence_model,
        'accident_model': accident_model,
        'severity_model': severity_model
    }

# ═══════════════════════════════════════════════════════════
# FEATURE EXTRACTION (NO NORMALIZATION!)
# ═══════════════════════════════════════════════════════════

def extract_features_from_frame(frame, models):
    device = models['device']
    
    if isinstance(frame, np.ndarray):
        pil_image = Image.fromarray(frame.astype('uint8'))
    else:
        pil_image = frame
    
    image_input = models['preprocess'](pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = models['clip_model'].encode_image(image_input)
    
    image_embedding = image_features.cpu().numpy()[0]
    
    blip_inputs = models['blip_processor'](pil_image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        caption_ids = models['blip_model'].generate(**blip_inputs, max_length=50)
    
    caption = models['blip_processor'].decode(caption_ids[0], skip_special_tokens=True)
    
    text_input = models['tokenizer']([caption]).to(device)
    
    with torch.no_grad():
        text_features = models['clip_model'].encode_text(text_input)
    
    text_embedding = text_features.cpu().numpy()[0]
    combined_features = np.concatenate([image_embedding, text_embedding])
    
    return {
        'combined_features': combined_features,
        'caption': caption,
        'image_embedding': image_embedding,
        'text_embedding': text_embedding
    }

# ═══════════════════════════════════════════════════════════
# PREDICTIONS (SAME AS BEFORE)
# ═══════════════════════════════════════════════════════════

def predict_violence(combined_features, model, device):
    model.eval()
    features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(features_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    return {
        'prediction': int(np.argmax(probs)),
        'confidence': float(probs[np.argmax(probs)]),
        'violence_prob': float(probs[1]),
        'safe_prob': float(probs[0])
    }

def predict_accident(combined_features, model, device):
    model.eval()
    features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(features_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    return {
        'prediction': int(np.argmax(probs)),
        'confidence': float(probs[np.argmax(probs)]),
        'accident_prob': float(probs[1]),
        'safe_prob': float(probs[0])
    }

def predict_severity(image_embedding, model, device):
    model.eval()
    image_tensor = torch.FloatTensor(image_embedding).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    severity_pred = int(np.argmax(probs))
    
    return {
        'level': "LOW" if severity_pred == 0 else "HIGH",
        'description': "MINOR/MODERATE" if severity_pred == 0 else "MAJOR",
        'confidence': float(probs[severity_pred]),
        'low_prob': float(probs[0]),
        'high_prob': float(probs[1])
    }

def extract_frames(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_count += 1
    
    cap.release()
    return frames

# ═══════════════════════════════════════════════════════════
# MAIN APP - COMPACT UI
# ═══════════════════════════════════════════════════════════

def main():
    
    st.markdown("<h1>🚨 Multi-Agent Emergency Detection System</h1>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════
    # SIDEBAR - COMPACT
    # ═══════════════════════════════════════════════════════
    
    st.sidebar.title("⚙️ Settings")
    
    detection_mode = st.sidebar.radio(
        "Detection Type:",
        [
            "Violence Only",
            "Accident Only", 
            "Accident + Severity",
            "All Agents"
        ],
        help="Select which agents to run"
    )
    
    fps = st.sidebar.slider("FPS", 1, 5, 1)
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Performance:**
    - Violence: 99.90%
    - Accident: 99.91%
    - Severity: 83.81%
    """)
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    st.success(f"✅ Models ready ({models['device'].upper()})")
    
    # ═══════════════════════════════════════════════════════
    # COMPACT UPLOAD SECTION
    # ═══════════════════════════════════════════════════════
    
    uploaded_file = st.file_uploader("📤 Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file is not None:
        
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # ═══════════════════════════════════════════════════════
        # COMPACT VIDEO PREVIEW (COLLAPSED BY DEFAULT)
        # ═══════════════════════════════════════════════════════
        
        with st.expander("📹 Preview Video", expanded=False):
            st.video(video_path)
        
        # ═══════════════════════════════════════════════════════
        # ANALYZE BUTTON
        # ═══════════════════════════════════════════════════════
        
        if st.button("🚀 Analyze Video"):
            
            with st.spinner("Extracting frames..."):
                frames = extract_frames(video_path, fps)
            
            st.info(f"Analyzing {len(frames)} frames...")
            
            all_violence = []
            all_accident = []
            all_severity = []
            
            progress_bar = st.progress(0)
            
            for idx, frame in enumerate(frames):
                
                feature_data = extract_features_from_frame(frame, models)
                
                if detection_mode in ["Violence Only", "All Agents"]:
                    violence_result = predict_violence(
                        feature_data['combined_features'],
                        models['violence_model'],
                        models['device']
                    )
                    violence_result['caption'] = feature_data['caption']
                    violence_result['frame_num'] = idx
                    all_violence.append(violence_result)
                
                if detection_mode in ["Accident Only", "Accident + Severity", "All Agents"]:
                    accident_result = predict_accident(
                        feature_data['combined_features'],
                        models['accident_model'],
                        models['device']
                    )
                    accident_result['caption'] = feature_data['caption']
                    accident_result['frame_num'] = idx
                    all_accident.append(accident_result)
                    
                    if detection_mode in ["Accident + Severity", "All Agents"]:
                        if accident_result['prediction'] == 1:
                            severity_result = predict_severity(
                                feature_data['image_embedding'],
                                models['severity_model'],
                                models['device']
                            )
                            severity_result['caption'] = feature_data['caption']
                            severity_result['frame_num'] = idx
                            all_severity.append(severity_result)
                
                progress_bar.progress((idx + 1) / len(frames))
            
            # ═══════════════════════════════════════════════════════
            # COMPACT RESULTS DISPLAY
            # ═══════════════════════════════════════════════════════
            
            st.markdown("---")
            
            # VIOLENCE
            if len(all_violence) > 0:
                violence_count = sum(1 for r in all_violence if r['prediction'] == 1)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Violence Frames", f"{violence_count}/{len(all_violence)}")
                with col2:
                    pct = (violence_count / len(all_violence)) * 100
                    st.metric("Violence %", f"{pct:.1f}%")
                
                if violence_count > 0:
                    st.markdown(f"""
                    <div class="danger-box">
                    <b>⚠️ VIOLENCE DETECTED</b> ({violence_count} frames)
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("🔴 View Violent Frames"):
                        for r in all_violence:
                            if r['prediction'] == 1:
                                st.markdown(f"**Frame {r['frame_num']}:** {r['violence_prob']*100:.1f}% - *{r['caption']}*")
                else:
                    st.markdown('<div class="success-box"><b>✅ NO VIOLENCE</b></div>', unsafe_allow_html=True)
            
            # ACCIDENT
            if len(all_accident) > 0:
                accident_count = sum(1 for r in all_accident if r['prediction'] == 1)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accident Frames", f"{accident_count}/{len(all_accident)}")
                with col2:
                    pct = (accident_count / len(all_accident)) * 100
                    st.metric("Accident %", f"{pct:.1f}%")
                
                if accident_count > 0:
                    st.markdown(f"""
                    <div class="danger-box">
                    <b>🚨 ACCIDENT DETECTED</b> ({accident_count} frames)
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("🚗 View Accident Frames"):
                        for r in all_accident:
                            if r['prediction'] == 1:
                                st.markdown(f"**Frame {r['frame_num']}:** {r['accident_prob']*100:.1f}% - *{r['caption']}*")
                else:
                    st.markdown('<div class="success-box"><b>✅ NO ACCIDENTS</b></div>', unsafe_allow_html=True)
            
            # SEVERITY
            if len(all_severity) > 0:
                high_count = sum(1 for r in all_severity if r['level'] == 'HIGH')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("High Severity", high_count)
                with col2:
                    st.metric("Low Severity", len(all_severity) - high_count)
                
                with st.expander("⚠️ View Severity Details"):
                    for r in all_severity:
                        emoji = "🔴" if r['level'] == 'HIGH' else "🟡"
                        st.markdown(f"{emoji} **Frame {r['frame_num']}:** {r['description']} ({r['confidence']*100:.1f}%)")
            
            try:
                os.unlink(video_path)
            except:
                pass

if __name__ == "__main__":
    main()
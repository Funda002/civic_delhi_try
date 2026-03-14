# 🏛️ Delhi Civic Grievance Intelligent System

## 🌟 Overview
The **Delhi Civic Grievance Intelligent System** is an advanced AI-driven platform designed to streamline the reporting, classification, and resolution of civic infrastructure and service issues in Delhi. 

In a fast-growing urban landscape, civic departments often face a deluge of complaints ranging from sanitation and waste management to electrical failures and illegal construction. Processing these manually is time-consuming and prone to routing errors. This system leverages state-of-the-art **Deep Learning** and **Computer Vision** models to transform unstructured images and user reports into actionable, routed intelligence.

---

## 🚀 The Problem We Solve
Reporting a civic issue should be as simple as taking a photo. However, the existing infrastructure for redressal often suffers from:
* **Inefficient Routing:** Complaints are often sent to the wrong department, causing delays.
* **Lack of Location Data:** Vague descriptions without specific ward or locality mapping hinder quick resolution.
* **Information Overload:** Officials are overwhelmed by raw, unorganized data.

Our solution automates the **End-to-End lifecycle** of a grievance: **Capture → Analyze → Categorize → Pinpoint Location → Route to Responsible Department.**

---

## 🧠 Key Features

### 1. Multimodal AI Analysis
The system doesn't just "see" an image; it understands it.
* **Computer Vision (BLIP):** Automatically generates captions for images to describe the scene (e.g., "a street filled with garbage").
* **OCR (EasyOCR):** Extracts text from street signs or notices found in photos, providing critical context for location detection.
* **Text Processing:** Allows users to manually add notes, which are merged with image insights for a holistic understanding of the problem.

### 2. Intelligent Categorization
Powered by **SetFit (Sentence-Transformer-based Few-shot Text Classification)**, the model classifies complaints into 12 specific municipal categories, ensuring the report lands in the correct inbox immediately.

### 3. Hierarchical Location Resolution
Utilizing a custom **Delhi Gazetteer**, the system maps natural language inputs to specific localities, zones, and wards, ensuring the grievance reaches the local authorities responsible for that specific geography.

### 4. Political Accountability
The system retrieves the councillor’s details for the detected ward, fostering transparency and accountability by letting users know exactly who is responsible for their area.

---

## 🛠️ Technology Stack
* **AI/ML:** SetFit (for classification), BLIP (for captioning), EasyOCR (for text extraction), Transformers.
* **Backend/Frontend:** Gradio (an intuitive Python-based UI framework).
* **Data:** Custom CSV-based gazetteers mapping Delhi localities, wards, and councillors.
* **Environment:** PyTorch, Pandas, and NumPy for high-performance data manipulation.

---
## 🧠 System Architecture

```text
    Citizen Upload
         │
         ▼
    Image + Text Input
         │
         ▼
    Multimodal AI Analysis
         │
     ┌───────────────┬───────────────┐
     │               │               │
     ▼               ▼               ▼
    BLIP         EasyOCR        User Notes
    Caption      Text Extract   Complaint
     │               │               │
     └───────────────┴───────────────┘
                 ▼
          Combined Context
                 ▼
          SetFit Classifier
          (Issue Category)
                 ▼
         Location Resolver
          (Delhi Gazetteer)
                 ▼
       Ward + Zone Detection
                 ▼
         Councillor Lookup
                 ▼
        Department Routing

```
### step 1:
## 📂 Project Structure

```text
/
├── app_new.py              # Main execution script (Gradio UI + Routing)
├── hierarchy_resolver.py   # Maps localities to wards/zones
├── locality_resolver.py    # Matches input text to Delhi locations
├── data/                   # Dataset repository
│   ├── delhi_localities_gazetteer.csv
│   └── ward_data.csv       # Councillor & Ward mapping
├── setfit_delhi_civic_model_1e/ # Trained classification model
└── requirements.txt        # System dependencies
```

### 🛠️ Step 2: Create a Virtual Environment

    python -m venv myenv

### 🛠️ Step 3: Install Dependencies

    pip install -r requirements.txt

### 🛠️ Step 4: Launch the System

    python app_new.py
### The system will initialize, load the AI models into memory, and launch a local web server at 
    http://127.0.0.1:7860

### 💡 How to Use

* **1️⃣ Upload/Input:** Provide an image of the civic issue and/or type the description of the problem.
* **2️⃣ Classify:** Click the **"Classify & Route"** button. The AI will analyze the visual and textual data.
* **3️⃣ Review:** Expand the "Technical Analysis" tab to see what the AI saw (OCR & Image Description).
* **4️⃣ Action:** The system will display the department responsible for solving your issue, along with the contact details of your local councillor.
* **5️⃣ Correct:** If the AI makes a location mistake, use the "Select Location" dropdown to manually override and ensure the routing is 100% accurate.

---
### demo of how this works--image 
![image alt](https://github.com/Funda002/civic_delhi_try/blob/43f231d411f5b2cec35790c8f5db310fd9a6a1d0/Screenshot%202026-03-14%20160047.png)

## 🤝 Contribution & Acknowledgements
This project is an evolving initiative to bridge the gap between technology and municipal governance in Delhi. Contributions are welcome! Whether you are interested in improving the accuracy of the location resolver, training a larger classification model, or enhancing the UI, please feel free to submit a Pull Request.

**❤️ Built for Smart Urban Governance**

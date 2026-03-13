import gradio as gr
from transformers import pipeline
from PIL import Image
import easyocr
import numpy as np
from setfit import SetFitModel
from departments import DEPARTMENT_MAP
from location_detector import detect_location
from ward_mapper import WardInfoProvider

# Load Models
print("Booting Systems...")
model = SetFitModel.from_pretrained("setfit_delhi_civic_model_1e")
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
reader = easyocr.Reader(['en', 'hi']) # Hindi support for street signs
ward_info = WardInfoProvider()

def process_input(text, image):
    try:
        # 1. Image Processing
        caption, ocr_text = "", ""
        if image is not None:
            pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
            caption = captioner(pil_image)[0]["generated_text"]
            ocr_text = " ".join([t[1] for t in reader.readtext(image)])
        
        # 2. Text Classification
        final_text = f"{text or ''} {caption} {ocr_text}".strip()
        if not final_text: return "No Input", "None", "N/A", "Awaiting data...", gr.update(visible=False)
        
        category = model.predict([final_text])[0]
        dept = DEPARTMENT_MAP.get(category, "General Administration")
        
        # 3. Location Search
        loc_data = detect_location(final_text)
        if loc_data and loc_data.get("location_found"):
            details = ward_info.get_ward_details(loc_data['ward_no'])
            if details:
                loc_str = f"Ward {details['ward_no']}: {details['ward_name']}"
                summary = (f"✅ Grievance: {category}\n📍 Location: {loc_str}\n"
                           f"👨‍💼 Official: {details['councillor']} ({details['party']})\n"
                           f"📞 Mobile: {details['mobile']}")
                return category, loc_str, dept, summary, gr.update(visible=False)
        
        return category, "Still not found", dept, "⚠️ Location unknown. Please use the bot.", gr.update(visible=True)
    except Exception as e:
        return "Error", "Error", "Error", f"Fault: {str(e)}", gr.update(visible=True)

def process_bot_reply(bot_input, orig_category, orig_dept):
    loc_data = detect_location(bot_input)
    if loc_data and loc_data.get("location_found"):
        details = ward_info.get_ward_details(loc_data['ward_no'])
        if details:
            loc_str = f"Ward {details['ward_no']}: {details['ward_name']}"
            summary = f"✅ Resolved!\n📍 Ward: {loc_str}\n👨‍💼 Routed to: {details['councillor']}"
            return loc_str, summary, gr.update(visible=False)
    return "Not found", "Please try a valid Ward Name (e.g. 'Lalita Park')", gr.update(visible=True)

# UI Construction
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🏛️ MCD Civic Grievance System (V3)")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 1. User Input")
            input_text = gr.Textbox(label="Grievance Description", lines=3)
            input_img = gr.Image(label="Evidence Photo")
            submit_btn = gr.Button("Analyze Complaint", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 🧠 2. AI Classification")
            out_cat = gr.Textbox(label="Predicted Category")
            out_loc = gr.Textbox(label="Detected Ward/Location")
            
            with gr.Group(visible=False) as bot_section:
                bot_in = gr.Textbox(label="🤖 Bot: Where is this located?")
                bot_btn = gr.Button("Confirm Location")

        with gr.Column(scale=1):
            gr.Markdown("### 📋 3. Dispatch Status")
            out_dept = gr.Textbox(label="Responsible Department")
            summary_out = gr.Textbox(label="System Log", lines=6)

    submit_btn.click(process_input, [input_text, input_img], [out_cat, out_loc, out_dept, summary_out, bot_section])
    bot_btn.click(process_bot_reply, [bot_in, out_cat, out_dept], [out_loc, summary_out, bot_section])

interface.launch()
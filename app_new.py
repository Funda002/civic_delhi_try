import gradio as gr
from PIL import Image
import easyocr
import pandas as pd
from setfit import SetFitModel
from transformers import pipeline
import numpy as np
from locality_resolver import match_locality, build_hierarchy

# 1. Configuration
DEPARTMENT_MAP = {
    "Infrastructure Failures": "Engineering Department",
    "Electrical Infrastructure": "Electrical & Mechanical Department",
    "Sanitation & Waste": "Environmental Management (DEMS)",
    "Public Health Risks": "Public Health Department",
    "Green & Urban Ecology": "Horticulture Department",
    "Animal-Related Issues": "Veterinary Department",
    "Illegal Construction & Encroachment": "Building/Licensing/Town Planning Dept",
    "Municipal Traffic Obstruction": "Engineering/Licensing Department",
    "Property, Land & Tax Issues": "Assessment & Collection Department",
    "Education & Civic Institutions": "Education Department",
    "Community & Public Spaces": "Community Services Department",
    "Governance, Corruption & Process Failures": "Vigilance/Law/Finance Department"
}

# 2. Initialize Models
print("Loading models and data... please wait.")
parent_map, leaf_map = build_hierarchy("data/delhi_localities_gazetteer.csv")
ward_df = pd.read_csv("data/ward_data.csv")

reader = easyocr.Reader(['en'], gpu=False) 
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
model = SetFitModel.from_pretrained("setfit_delhi_civic_model_1e")

def get_politician_info(ward_no):
    row = ward_df[ward_df['ward_no'] == int(ward_no)]
    if not row.empty:
        r = row.iloc[0]
        return (f"WARD NAME: {r['ward_name']}\n"
                f"WARD NUMBER: {r['ward_no']}\n"
                f"COUNCILLOR: {r['councillor']}\n"
                f"PARTY: {r['party']}")
    return "Politician info not found."

def process_and_classify(image, user_text):
    ocr_text = "No image"
    image_description = "No image"
    final_input = user_text
    
    if image is not None:
        ocr_results = reader.readtext(image, detail=0)
        ocr_text = " ".join(ocr_results) if ocr_results else "No printed text found."
        pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
        caption_result = captioner(pil_image)
        image_description = caption_result[0]['generated_text']
        final_input = f"Image shows: {image_description}. Signs: {ocr_text}. Note: {user_text}"
    
    prediction = model.predict([final_input])[0]
    loc, ward_info = match_locality(final_input, parent_map, leaf_map)
    
    if loc:
        dept = DEPARTMENT_MAP.get(prediction, "General Department")
        poli_msg = get_politician_info(ward_info['ward_no'])
        dept_msg = (f"ISSUE DETAILS:\n{final_input}\n\n"
                    f"TO: {dept}\n"
                    f"ACTION: Please investigate this {prediction} report immediately.")
        return ocr_text, image_description, final_input, prediction, loc.upper(), dept_msg, poli_msg, gr.update(visible=False)
    else:
        return ocr_text, image_description, final_input, prediction, "NOT DETECTED", "Location not found.", "N/A", gr.update(visible=True)

def finalize_routing(selected_loc, prediction, input_text):
    ward_info = build_hierarchy("data/delhi_localities_gazetteer.csv")[1].get(selected_loc.lower())
    poli_msg = get_politician_info(ward_info['ward_no']) if ward_info else "N/A"
    dept = DEPARTMENT_MAP.get(prediction, "General Department")
    dept_msg = (f"ISSUE DETAILS:\n{input_text}\n\nTO: {dept}\nACTION: Route to {dept} immediately.")
    return selected_loc.upper(), dept_msg, poli_msg

# 3. UI Layout
with gr.Blocks(theme=gr.themes.Base(), css="footer {visibility: hidden;}") as demo:
    gr.Markdown("# 🏛️ Delhi Civic Grievance System")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Image")
            text_input = gr.Textbox(lines=3, label="Additional Details")
            submit_btn = gr.Button("Classify & Route", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Accordion("View Technical Analysis (OCR & AI)", open=False):
                ocr_display = gr.Textbox(label="OCR Output", interactive=False)
                caption_display = gr.Textbox(label="AI Image Description", interactive=False)
            
            with gr.Row():
                model_input_display = gr.Textbox(label="Model Input", interactive=False)
                prediction_display = gr.Textbox(label="Prediction", interactive=False)
                loc_display = gr.Textbox(label="Detected Location", interactive=False)
                loc_dropdown = gr.Dropdown(choices=sorted(list(leaf_map.keys())), label="Select Location", visible=False)

            with gr.Row():
                dept_display = gr.Textbox(label="Department Routing Message", interactive=False, lines=6)
                poli_display = gr.Textbox(label="Politician/Councillor Info", interactive=False, lines=6)

    submit_btn.click(
        fn=process_and_classify, 
        inputs=[image_input, text_input], 
        outputs=[ocr_display, caption_display, model_input_display, prediction_display, loc_display, dept_display, poli_display, loc_dropdown]
    )
    loc_dropdown.change(
        fn=finalize_routing, 
        inputs=[loc_dropdown, prediction_display, model_input_display], 
        outputs=[loc_display, dept_display, poli_display]
    )

if __name__ == "__main__":
    demo.launch()
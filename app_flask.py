from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import easyocr
import pandas as pd
from setfit import SetFitModel
from transformers import pipeline
import numpy as np

from locality_resolver import match_locality, build_hierarchy

app = Flask(__name__)
CORS(app)

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

print("Loading models and data...")

parent_map, leaf_map = build_hierarchy("data/delhi_localities_gazetteer.csv")
ward_df = pd.read_csv("data/ward_data.csv")

reader = easyocr.Reader(["en"], gpu=False)
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
model = SetFitModel.from_pretrained("setfit_delhi_civic_model_1e")

print("Models loaded successfully!")


def get_politician_info(ward_no):
    row = ward_df[ward_df["ward_no"] == int(ward_no)]

    if row.empty:
        return None

    r = row.iloc[0]

    return {
        "ward_name": r["ward_name"],
        "ward_no": int(r["ward_no"]),
        "councillor": r["councillor"],
        "party": r["party"]
    }


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Delhi Civic Grievance API is running",
        "main_endpoint": "/classify",
        "method": "POST"
    })


@app.route("/classify", methods=["POST"])
def classify_complaint():
    try:
        user_text = request.form.get("user_text", "")
        image_file = request.files.get("image")

        ocr_text = "No image provided"
        image_description = "No image provided"
        final_input = user_text

        if image_file:
            image = Image.open(image_file).convert("RGB")
            image_np = np.array(image)

            ocr_results = reader.readtext(image_np, detail=0)
            ocr_text = " ".join(ocr_results) if ocr_results else "No printed text found"

            caption_result = captioner(image)
            image_description = caption_result[0]["generated_text"]

            final_input = (
                f"Image shows: {image_description}. "
                f"Signs: {ocr_text}. "
                f"Note: {user_text}"
            )

        if not final_input.strip():
            return jsonify({
                "error": "Please provide an image or complaint text"
            }), 400

        prediction = model.predict([final_input])[0]

        loc, ward_info = match_locality(final_input, parent_map, leaf_map)

        department = DEPARTMENT_MAP.get(prediction, "General Department")

        response = {
            "technical_analysis": {
                "ocr_text": ocr_text,
                "image_description": image_description,
                "model_input": final_input
            },
            "classification": {
                "predicted_category": prediction,
                "department": department
            },
            "location": {
                "detected_location": loc.upper() if loc else "NOT DETECTED",
                "ward_info": ward_info if ward_info else None
            },
            "councillor_info": None,
            "routing_message": None,
            "manual_location_required": False
        }

        if loc and ward_info:
            politician = get_politician_info(ward_info["ward_no"])

            response["councillor_info"] = politician
            response["routing_message"] = (
                f"ISSUE DETAILS:\n{final_input}\n\n"
                f"TO: {department}\n"
                f"ACTION: Please investigate this {prediction} report immediately."
            )
        else:
            response["manual_location_required"] = True
            response["routing_message"] = "Location not detected. Please select location manually."

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/finalize-routing", methods=["POST"])
def finalize_routing():
    try:
        data = request.json

        selected_location = data.get("selected_location")
        prediction = data.get("prediction")
        input_text = data.get("input_text")

        if not selected_location or not prediction or not input_text:
            return jsonify({
                "error": "selected_location, prediction, and input_text are required"
            }), 400

        ward_info = leaf_map.get(selected_location.lower())

        if not ward_info:
            return jsonify({
                "error": "Selected location not found"
            }), 404

        department = DEPARTMENT_MAP.get(prediction, "General Department")
        politician = get_politician_info(ward_info["ward_no"])

        return jsonify({
            "selected_location": selected_location.upper(),
            "department": department,
            "ward_info": ward_info,
            "councillor_info": politician,
            "routing_message": (
                f"ISSUE DETAILS:\n{input_text}\n\n"
                f"TO: {department}\n"
                f"ACTION: Route to {department} immediately."
            )
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)

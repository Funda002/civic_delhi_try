import warnings
from setfit import SetFitModel

# Hide annoying background warnings
warnings.filterwarnings("ignore")

def main():
    model_directory = "setfit_delhi_civic_model_1e"

    print("Loading the trained grievance model. This might take a few seconds...")

    try:
        model = SetFitModel.from_pretrained(model_directory)
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 5 Complex Examples in Hinglish, Hindi, and English
    texts_to_classify = [
        # 1. Hinglish (Drainage overflow + Official inaction)
        "Pichle ek mahine se gali number 4 mein naali block hai jiske wajah se poora ganda paani road par aa gaya hai. Complaint register ki thi par abhi tak koi MCD wala check karne nahi aaya.",
        
        # 2. Hindi (Garbage dump + Health risk/Disease + Ignoring complaints)
        "हमारे इलाके के पार्क में कूड़े का पहाड़ बन गया है और वहां मरे हुए जानवर पड़े हैं। डेंगू और मलेरिया फैलने का डर है, लेकिन निगम पार्षद कोई सुनवाई नहीं कर रहे हैं।",
        
        # 3. English (Illegal construction + Encroachment + Pedestrian hazard)
        "There is illegal commercial construction happening in the residential area of block C. They have encroached upon the pedestrian footpath, forcing senior citizens to walk on the busy main road.",
        
        # 4. Hinglish (Broken streetlights + Crime/Safety + Unreachable helpline)
        "Main road ki saari street lights pichle 2 hafte se kharab hain. Andhera itna hota hai ki raat ko chori badh gayi hai aur MCD ka helpline number hamesha busy aata hai.",
        
        # 5. Mixed English/Hinglish (Stray animals + Taxpayers grievance)
        "Hum log time par property tax pay karte hain phir bhi roads par itne stray cattle aur kutte hain. Yesterday a child was bitten by a stray dog near the primary school and MCD is ignoring this."

        # 1. Hinglish -> Expected: Sanitation & Waste
        "Pichle ek hafte se hamari society ke bahar se kachra nahi uthaya gaya hai. MCD ki gaadi aana band ho gayi hai aur dustbin overflow kar raha hai.",
        
        # 2. English -> Expected: Animal-Related Issues
        "There is a pack of aggressive stray dogs near the metro station. They have been chasing two-wheelers and biting pedestrians. Please send the animal control team.",
        
        # 3. Hindi -> Expected: Green & Urban Ecology
        "कल रात की तेज आंधी में हमारे घर के सामने एक बहुत बड़ा पेड़ गिर गया है। इसकी वजह से रास्ता पूरी तरह बंद हो गया है, कृपया इसे जल्द हटाएं।",
        
        # 4. Hinglish -> Expected: Education & Civic Institutions
        "Ward 52 ke MCD primary school mein bacchon ke liye peene ka paani nahi hai. Classrooms ki chhat se paani tapak raha hai, bachon ko dikkat ho rahi hai.",
        
        # 5. Hindi/Hinglish -> Expected: Community & Public Spaces
        "Sector 7 ka public park bilkul barbad ho chuka hai. Bacchon ke jhule toot gaye hain aur wahan maintenance bilkul zero hai."
    ]

    # Loop through all 5 examples and print the prediction
    for i, text in enumerate(texts_to_classify, 1):
        print("-" * 70)
        print(f"Example {i}")
        print(f"Input Text: \"{text}\"")
        
        # Make the prediction
        prediction = model.predict([text])
        
        print(f"\n=> Predicted Classification: {prediction[0]}")
        print("-" * 70 + "\n")

# This standard Python block prevents the background memory crash on Windows
if __name__ == "__main__":
    main()
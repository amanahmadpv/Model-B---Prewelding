import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title
st.title("🔧 Welding Rod Consumption Predictor - Model B (Pre-Welding)")

st.markdown("Enter details just before welding to estimate rod consumption (kg/ton)")

# Load model and encoders
model = joblib.load('XGBoost_prewelding_final.pkl')
top_features = joblib.load('features_b.pkl')

fp_encoder = joblib.load('fp_encoder_b.pkl')
grade_type_encoder = joblib.load('grade_type_encoder.pkl')
smp_bulk_encoder = joblib.load('smp_bulk_encoder.pkl')
heat_encoder = joblib.load('heat_type_encoder.pkl')
family_encoder = joblib.load('family_encoder.pkl')
desc_encoder = joblib.load('desc_encoder.pkl')

# Helper function to encode or fallback
def encode_label(value, encoder, encoder_name):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        st.warning(f"⚠️ Unknown {encoder_name} '{value}' — using fallback.")
        return -1  # Fallback for unseen values

# --- Input fields ---
fp_no = st.text_input("New FP No.")
desc = st.text_input("Description")
grade_type = st.selectbox("Grade Type", grade_type_encoder.classes_)
smp_bulk = st.selectbox("Sample Type (Bulk/Grade Sample)", smp_bulk_encoder.classes_)
heat_type = st.selectbox("Heat Type", heat_encoder.classes_)
family = st.selectbox("Family", family_encoder.classes_)

tot_cast_wt = st.number_input("Total Despatch Cast Weight (Tons)", min_value=0.0, step=0.1)
rt_req = st.selectbox("RT Required?", [0, 1])
goug_qty = st.number_input("Gouging Rod Quantity", min_value=0.0, step=0.1)
nozzle = st.number_input("Nozzle Count", min_value=0)
tapping_temp = st.number_input("Tapping Temperature", min_value=0.0, step=1.0)
pouring_temp = st.number_input("Pouring Temperature", min_value=0.0, step=1.0)

# --- Encode inputs ---
if st.button("Predict TOT (Kgs/T)"):
    try:
        input_dict = {
            'new_fp_no_encoded': encode_label(fp_no, fp_encoder, "FP No."),
            'Desc_encoded': encode_label(desc, desc_encoder, "Description"),
            'Grade_encoded': encode_label(grade_type, grade_type_encoder, "Grade Type"),
            'Smp_encoded': encode_label(smp_bulk, smp_bulk_encoder, "Sample Type"),
            'Heat_encoded': encode_label(heat_type, heat_encoder, "Heat Type"),
            'Family_encoded': encode_label(family, family_encoder, "Family"),
            'TotDespCastWt(Ton)': tot_cast_wt,
            'RT Req': rt_req,
            'GougRodQty': goug_qty,
            'Nozzle': nozzle,
            'TappingTemp': tapping_temp,
            'PouringTemp': pouring_temp,
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])[top_features]

        # Predict (log scale)
        pred_log = model.predict(input_df)[0]

        # Convert to actual value
        predicted_kg_per_ton = np.expm1(pred_log)

        st.success(f"🔍 Estimated Welding Rod Consumption: **{predicted_kg_per_ton:.2f} kg/ton**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

import streamlit as st
import pandas as pd
import numpy as np
from matminer.featurizers.composition import ElementProperty
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load model dan scaler
gbr = joblib.load("gbr_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

# Featurizer
featurizer = ElementProperty.from_preset("magpie")

st.title("Band Gap Prediction for Binary Materials")
st.markdown("Masukkan rumus kimia material biner (contoh: GaAs, ZnO, CdTe)")

formula_input = st.text_input("Rumus Kimia Material:")

if st.button("Prediksi Band Gap"):
    try:
        comp = Composition(formula_input)
        df_input = pd.DataFrame({"composition": [comp]})
        df_features = featurizer.featurize_dataframe(df_input, "composition", ignore_errors=True)

        X_input = df_features[selected_features]
        X_scaled = scaler.transform(X_input)
        pred_log = gbr.predict(X_scaled)
        pred = np.expm1(pred_log[0])

        st.success(f"Prediksi Band Gap: {pred:.3f} eV")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

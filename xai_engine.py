import pandas as pd
import numpy as np

# Sayısal değişkenlerin veri setindeki geçerli aralıkları (Clipping)
FEATURE_BOUNDS = {
    "age": (18, 90),
    "credit_amount": (100, 20000), # Model bazındaki değerler
    "duration": (4, 72),
    "installment_rate": (1, 4)
}

# Sayısal değişkenlerde oynama oranları
NUMERIC_PERTURB = {
    "age": 5,
    "credit_amount": 500,
    "duration": 6,
    "installment_rate": 1
}

# Kategorik değişkenler için alternatif güvenli değerler
CATEGORICAL_ALTERNATIVES = {
    "credit_history": ["A34", "A32", "A31"],
    "job": ["A171", "A173", "A172"],
    "housing": ["A152", "A151"]
}


def _predict_score(model, preprocessor, inp: dict):
    """Modelden 0-1900 skor üretir"""
    df = pd.DataFrame([inp])
    proc = preprocessor.transform(df)
    risk = model.predict(proc, verbose=0)[0][0]
    return int((1 - risk) * 1900)


def explain_prediction(model, preprocessor, inp: dict, top_k=6):

    base_score = _predict_score(model, preprocessor, inp)
    effects = []

    # --- SAYISAL DEĞİŞKENLER ---
    for feature, step in NUMERIC_PERTURB.items():
        if feature in inp:
            modified = inp.copy()
            modified[feature] = max(1, inp[feature] + step)

            new_score = _predict_score(model, preprocessor, modified)
            delta = new_score - base_score

            effects.append({
                "feature": feature,
                "delta": delta,
                "direction": "positive" if delta > 0 else "negative"
            })

    # --- KATEGORİK DEĞİŞKENLER ---
    for feature, alternatives in CATEGORICAL_ALTERNATIVES.items():
        if feature in inp:
            original = inp[feature]
            for alt in alternatives:
                if alt != original:
                    modified = inp.copy()
                    modified[feature] = alt

                    new_score = _predict_score(model, preprocessor, modified)
                    delta = new_score - base_score

                    effects.append({
                        "feature": feature,
                        "delta": delta,
                        "direction": "positive" if delta > 0 else "negative"
                    })
                    break  # tek alternatif yeterli

    # Mutlak etkiye göre sırala
    effects = sorted(effects, key=lambda x: abs(x["delta"]), reverse=True)

    return {
        "base_score": base_score,
        "effects": effects[:top_k]
    }

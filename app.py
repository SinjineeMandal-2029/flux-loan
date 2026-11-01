import json, joblib, pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
 
app = Flask(__name__, static_folder=None)
CORS(app)
 
model = joblib.load("model.pkl")
meta = json.load(open("meta.json"))
FEATURES = meta["features"]
CATS = meta["categories"]
EDU = meta["education"]
 
@app.get("/")
def home():
    return send_from_directory(".", "index.html")
 
@app.get("/health")
def health():
    return {"status": "ok"}
 
@app.get("/meta")
def get_meta():
    return jsonify(categories=CATS, education=EDU)
 
@app.post("/predict")
def predict():
    d = request.get_json(force=True)
    row = {f: 0.0 for f in FEATURES}
    for f in FEATURES:
        if f in d and f != "Education_Level":
            try: row[f] = float(d[f])
            except (TypeError, ValueError): pass
    if d.get("Education_Level") in EDU:
        row["Education_Level"] = EDU.index(d["Education_Level"])
    for col, values in CATS.items():
        name = f"{col}_{d.get(col)}"
        if name in row: row[name] = 1.0
    X = pd.DataFrame([row], columns=FEATURES)
    p = float(model.predict_proba(X)[0][1])
    return jsonify(approved=p >= 0.5, probability=p)
 
if __name__ == "__main__":
    app.run(debug=True, port=5000)
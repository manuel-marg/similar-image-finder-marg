import os
import json
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["ALBUMS_FOLDER"] = os.path.join(app.config["UPLOAD_FOLDER"], "albums")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "webp"}

os.makedirs(app.config["ALBUMS_FOLDER"], exist_ok=True)


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"albums": {}, "settings": {"confidence_threshold": 0.85}}


def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


feature_extractor = None


def get_feature_extractor():
    global feature_extractor
    if feature_extractor is None:
        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3)
        )
        feature_extractor = tf.keras.Model(
            inputs=base_model.input, outputs=base_model.output
        )
    return feature_extractor


def extract_features(image_path):
    model = get_feature_extractor()
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    features = model.predict(img_array, verbose=0)
    return features.flatten()


def load_album_features(album_name):
    album_dir = os.path.join(app.config["ALBUMS_FOLDER"], album_name)
    if not os.path.isdir(album_dir):
        return []
    features_list = []
    for fname in sorted(os.listdir(album_dir)):
        if allowed_file(fname):
            fpath = os.path.join(album_dir, fname)
            try:
                feat = extract_features(fpath)
                features_list.append(feat)
            except Exception:
                continue
    return features_list


album_features_cache = {}


def get_album_features(album_name):
    if album_name not in album_features_cache:
        album_features_cache[album_name] = load_album_features(album_name)
    return album_features_cache[album_name]


def invalidate_album_cache(album_name):
    album_features_cache.pop(album_name, None)


def compute_similarity(query_features, reference_features_list):
    if not reference_features_list:
        return 0.0
    query_2d = query_features.reshape(1, -1)
    ref_matrix = np.array(reference_features_list)
    sims = cosine_similarity(query_2d, ref_matrix)[0]
    return float(np.max(sims))


def fire_webhook(webhook_url, payload):
    try:
        resp = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        return {"status": resp.status_code, "ok": resp.ok}
    except Exception as e:
        return {"status": 0, "ok": False, "error": str(e)}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/albums", methods=["GET"])
def list_albums():
    config = load_config()
    albums = []
    for name, meta in config.get("albums", {}).items():
        album_dir = os.path.join(app.config["ALBUMS_FOLDER"], name)
        image_count = 0
        if os.path.isdir(album_dir):
            image_count = len([f for f in os.listdir(album_dir) if allowed_file(f)])
        albums.append({
            "name": name,
            "display_name": meta.get("display_name", name),
            "webhook_url": meta.get("webhook_url", ""),
            "image_count": image_count,
            "created_at": meta.get("created_at", ""),
        })
    return jsonify({"albums": albums})


@app.route("/api/albums", methods=["POST"])
def create_album():
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Se requiere un nombre para el album"}), 400

    raw_name = data["name"].strip()
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw_name).strip("_").lower()
    if not slug:
        return jsonify({"error": "Nombre de album invalido"}), 400

    config = load_config()
    if slug in config.get("albums", {}):
        return jsonify({"error": f"El album '{slug}' ya existe"}), 409

    album_dir = os.path.join(app.config["ALBUMS_FOLDER"], slug)
    os.makedirs(album_dir, exist_ok=True)

    config.setdefault("albums", {})[slug] = {
        "display_name": raw_name,
        "webhook_url": data.get("webhook_url", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_config(config)
    invalidate_album_cache(slug)

    return jsonify({"ok": True, "album": slug}), 201


@app.route("/api/albums/<album_name>", methods=["DELETE"])
def delete_album(album_name):
    config = load_config()
    if album_name not in config.get("albums", {}):
        return jsonify({"error": "Album no encontrado"}), 404

    album_dir = os.path.join(app.config["ALBUMS_FOLDER"], album_name)
    if os.path.isdir(album_dir):
        shutil.rmtree(album_dir)

    del config["albums"][album_name]
    save_config(config)
    invalidate_album_cache(album_name)

    return jsonify({"ok": True})


@app.route("/api/albums/<album_name>/webhook", methods=["PUT"])
def update_webhook(album_name):
    config = load_config()
    if album_name not in config.get("albums", {}):
        return jsonify({"error": "Album no encontrado"}), 404

    data = request.get_json()
    config["albums"][album_name]["webhook_url"] = data.get("webhook_url", "")
    save_config(config)
    return jsonify({"ok": True})


@app.route("/api/albums/<album_name>/upload", methods=["POST"])
def upload_reference_images(album_name):
    config = load_config()
    if album_name not in config.get("albums", {}):
        return jsonify({"error": "Album no encontrado"}), 404

    if "images" not in request.files:
        return jsonify({"error": "No se enviaron archivos"}), 400

    files = request.files.getlist("images")
    uploaded = []
    for f in files:
        if f and f.filename and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
            album_dir = os.path.join(app.config["ALBUMS_FOLDER"], album_name)
            os.makedirs(album_dir, exist_ok=True)
            f.save(os.path.join(album_dir, unique_name))
            uploaded.append(unique_name)

    invalidate_album_cache(album_name)

    return jsonify({"ok": True, "uploaded": len(uploaded), "files": uploaded})


@app.route("/api/albums/<album_name>/images", methods=["GET"])
def list_album_images(album_name):
    config = load_config()
    if album_name not in config.get("albums", {}):
        return jsonify({"error": "Album no encontrado"}), 404

    album_dir = os.path.join(app.config["ALBUMS_FOLDER"], album_name)
    images = []
    if os.path.isdir(album_dir):
        for fname in sorted(os.listdir(album_dir)):
            if allowed_file(fname):
                images.append({
                    "name": fname,
                    "url": f"/static/uploads/albums/{album_name}/{fname}",
                })
    return jsonify({"images": images})


@app.route("/api/albums/<album_name>/images/<filename>", methods=["DELETE"])
def delete_album_image(album_name, filename):
    config = load_config()
    if album_name not in config.get("albums", {}):
        return jsonify({"error": "Album no encontrado"}), 404

    fpath = os.path.join(app.config["ALBUMS_FOLDER"], album_name, secure_filename(filename))
    if os.path.isfile(fpath):
        os.remove(fpath)
        invalidate_album_cache(album_name)
        return jsonify({"ok": True})

    return jsonify({"error": "Imagen no encontrada"}), 404


@app.route("/api/classify", methods=["POST"])
def classify_image():
    if "image" not in request.files:
        return jsonify({"error": "No se envio imagen"}), 400

    file = request.files["image"]
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Formato de archivo no soportado"}), 400

    temp_path = os.path.join(app.config["UPLOAD_FOLDER"], f"temp_{uuid.uuid4().hex}.jpg")
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    file.save(temp_path)

    try:
        query_features = extract_features(temp_path)
    except Exception as e:
        os.remove(temp_path)
        return jsonify({"error": f"Error procesando imagen: {str(e)}"}), 500

    config = load_config()
    threshold = config.get("settings", {}).get("confidence_threshold", 0.85)
    results = []

    for album_name, meta in config.get("albums", {}).items():
        ref_features = get_album_features(album_name)
        if not ref_features:
            results.append({
                "album": album_name,
                "display_name": meta.get("display_name", album_name),
                "similarity": 0.0,
                "matched": False,
                "webhook_fired": False,
            })
            continue

        similarity = compute_similarity(query_features, ref_features)
        matched = similarity >= threshold

        webhook_fired = False
        if matched and meta.get("webhook_url"):
            now = datetime.now(timezone.utc).isoformat()
            payload = {
                "album": album_name,
                "display_name": meta.get("display_name", album_name),
                "similarity_percent": round(similarity * 100, 2),
                "confidence_threshold": round(threshold * 100, 2),
                "matched": True,
                "timestamp": now,
                "source": "similar-image-finder-marg",
                "event": "quality_check",
            }
            wh_result = fire_webhook(meta["webhook_url"], payload)
            webhook_fired = wh_result.get("ok", False)

        results.append({
            "album": album_name,
            "display_name": meta.get("display_name", album_name),
            "similarity": round(similarity, 4),
            "similarity_percent": round(similarity * 100, 2),
            "matched": matched,
            "webhook_fired": webhook_fired,
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    best_match = results[0] if results else None

    os.remove(temp_path)

    return jsonify({
        "ok": True,
        "threshold": round(threshold * 100, 2),
        "best_match": best_match,
        "results": results,
    })


@app.route("/api/settings", methods=["GET"])
def get_settings():
    config = load_config()
    return jsonify(config.get("settings", {}))


@app.route("/api/settings", methods=["PUT"])
def update_settings():
    config = load_config()
    data = request.get_json()
    config.setdefault("settings", {}).update(data)
    save_config(config)
    return jsonify({"ok": True, "settings": config["settings"]})


@app.route("/api/rebuild-cache", methods=["POST"])
def rebuild_cache():
    album_features_cache.clear()
    config = load_config()
    for album_name in config.get("albums", {}):
        get_album_features(album_name)
    return jsonify({"ok": True, "cached_albums": list(album_features_cache.keys())})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


import os
import re
import io
import base64
import traceback

import cv2
import numpy as np
import pytesseract
from flask import Flask, request, jsonify

TESSERACT_CMD = os.environ.get("TESSERACT_CMD")  # e.g. C:\...\tesseract.exe
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

INCLUDE_NULLS = False
MAX_UPLOAD_MB = 10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

def decode_image(file_bytes):
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Unsupported format or corrupted file.")
    return image


def create_variants(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    if w < 600:
        scale = 4
    elif w < 1200:
        scale = 3
    elif w < 2000:
        scale = 2
    else:
        scale = 1

    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return {"denoised": denoised, "otsu": otsu, "adaptive": adaptive, "clahe": enhanced}


def run_ocr(variants):
    results = {}
    for name, img in variants.items():
        text = pytesseract.image_to_string(img, config="--psm 6", lang="eng")
        if text.strip():
            results[name] = text
    return results


def clean_lines(text):
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"\s+", " ", line)
        lines.append(line)
    return lines


def dedupe_lines(lines):
    seen = set()
    out = []
    for line in lines:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            out.append(line)
    return out

DOB_RE = re.compile(
    r"(?:DOB|D\.O\.B|Date\s*of\s*Birth)[^0-9]{0,10}(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
    re.IGNORECASE,
)
YOB_RE = re.compile(r"Year\s*of\s*Birth[^0-9]{0,10}(\d{4})", re.IGNORECASE)
ANY_DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b")
GENDER_RE = re.compile(r"\b(MALE|FEMALE|OTHER|TRANSGENDER)\b", re.IGNORECASE)
PINCODE_RE = re.compile(r"\b[1-9][0-9]{5}\b")
AADHAAR_RE = re.compile(r"\b(\d{4})\s?(\d{4})\s?(\d{4})\b")
RELATION_RE = re.compile(r"\b(S/O|D/O|W/O|C/O|S\.O\.|D\.O\.|W\.O\.|C\.O\.)\b", re.IGNORECASE)

IGNORED_NAME_WORDS = {
    "government", "india", "authority", "unique", "identification",
    "aadhaar", "male", "female", "other", "transgender",
    "dob", "date", "birth", "year", "of",
}

ADDRESS_WHITELIST = {
    "S/O", "D/O", "W/O", "C/O", "PO", "P.O", "VTC", "DIST", "TAL", "PS",
    "RD", "ST", "NR", "OPP", "UP", "MP", "WB", "AP", "TN", "KA", "GJ",
    "RJ", "HR", "PB", "JK", "DL", "HP", "CH", "GA", "KL", "OR", "BR",
    "JH", "CG", "MH", "AS", "NO",
}


def verhoeff_checksum_valid(number):
    d = [
        [0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
        [3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
        [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
        [9,8,7,6,5,4,3,2,1,0],
    ]
    p = [
        [0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
        [8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
        [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8],
    ]
    c = 0
    digits = [int(x) for x in reversed(number)]
    for i, digit in enumerate(digits):
        c = d[c][p[i % 8][digit]]
    return c == 0


def find_aadhaar_number(full_text):
    candidates = AADHAAR_RE.findall(full_text)
    for groups in candidates:
        number = "".join(groups)
        if verhoeff_checksum_valid(number):
            return f"{groups[0]} {groups[1]} {groups[2]}"
    if candidates:
        g = candidates[0]
        return f"{g[0]} {g[1]} {g[2]}"
    return None


def normalize_date(raw):
    parts = re.split(r"[/-]", raw)
    if len(parts) == 3:
        d, mth, y = parts
        return f"{int(d):02d}-{int(mth):02d}-{y}"
    return raw


def find_dob(full_text):
    m = DOB_RE.search(full_text)
    if m:
        return normalize_date(m.group(1))
    m = YOB_RE.search(full_text)
    if m:
        return m.group(1)
    m = ANY_DATE_RE.search(full_text)
    if m:
        return normalize_date(m.group(1))
    return None


def find_gender(full_text):
    m = GENDER_RE.search(full_text)
    return m.group(1).capitalize() if m else None


def find_pincode(full_text):
    matches = PINCODE_RE.findall(full_text)
    return matches[-1] if matches else None


def _extract_name_words(line):
    words = re.findall(r"[A-Za-z]+", line)
    words = [w for w in words if w.lower() not in IGNORED_NAME_WORDS]
    if len(words) >= 2:
        return " ".join(words)
    return None


def find_name(lines):
    dob_idx = None
    for i, line in enumerate(lines):
        if re.search(r"DOB|D\.O\.B|Date\s*of\s*Birth|Year\s*of\s*Birth", line, re.IGNORECASE):
            dob_idx = i
            break

    if dob_idx is not None:
        for i in range(dob_idx - 1, max(-1, dob_idx - 5), -1):
            candidate = _extract_name_words(lines[i])
            if candidate:
                return candidate

    for line in lines:
        if re.search(r"DOB|D\.O\.B|Date\s*of\s*Birth|Year\s*of\s*Birth", line, re.IGNORECASE):
            continue
        if GENDER_RE.search(line):
            continue
        if RELATION_RE.search(line):
            continue
        if AADHAAR_RE.search(line):
            continue
        candidate = _extract_name_words(line)
        if candidate:
            return candidate

    return None


def is_noise_token(token):
    core = re.sub(r"[^A-Za-z]", "", token)
    if not core:
        return False
    if core.upper() in ADDRESS_WHITELIST:
        return False
    if len(core) <= 2:
        return True
    if core.isupper() and len(core) <= 4:
        return True
    return False


def clean_address_text(address):
    address = re.sub(r"[{}|~`^_\\]", "", address)
    tokens = address.split(" ")
    kept = [t for t in tokens if t and not is_noise_token(t)]
    address = " ".join(kept)
    address = re.sub(r"\s+", " ", address).strip()
    address = re.sub(r"\s*,\s*", ", ", address)
    address = re.sub(r"(,\s*)+", ", ", address)
    address = re.sub(r"^,\s*|,\s*$", "", address)
    return address.strip()


def find_address(lines):
    start_line, start_pos = None, None
    for i, line in enumerate(lines):
        m = RELATION_RE.search(line)
        if m:
            start_line, start_pos = i, m.start()
            break

    if start_line is None:
        return None

    address_lines = []
    for offset, line in enumerate(lines[start_line:]):
        if re.fullmatch(r"(?:\d{4}\s*){3}", line):
            continue
        if offset == 0:
            line = line[start_pos:]
        address_lines.append(line)
        if PINCODE_RE.search(line):
            break

    address = " ".join(address_lines)
    address = clean_address_text(address)
    return address or None

def extract_fields(lines, full_text):
    return {
        "name": find_name(lines),
        "dob": find_dob(full_text),
        "gender": find_gender(full_text),
        "aadhaar_number": find_aadhaar_number(full_text),
        "address": find_address(lines),
        "pincode": find_pincode(full_text),
    }


def score_result(fields):
    return sum(1 for v in fields.values() if v)


def process_image_bytes(file_bytes):
    image = decode_image(file_bytes)
    variants = create_variants(image)
    ocr_texts = run_ocr(variants)

    if not ocr_texts:
        raise ValueError("OCR produced no text from the image.")

    best_fields, best_score = None, -1
    for text in ocr_texts.values():
        lines = dedupe_lines(clean_lines(text))
        fields = extract_fields(lines, "\n".join(lines))
        score = score_result(fields)
        if score > best_score:
            best_fields, best_score = fields, score

    merged_lines = dedupe_lines(
        [line for text in ocr_texts.values() for line in clean_lines(text)]
    )
    merged_fields = extract_fields(merged_lines, "\n".join(merged_lines))

    final = dict(best_fields)
    for key, value in merged_fields.items():
        if not final.get(key) and value:
            final[key] = value

    if not INCLUDE_NULLS:
        final = {k: v for k, v in final.items() if v}

    return final

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/extract", methods=["POST"])
def extract():
    try:
        if "file" in request.files:
            file_bytes = request.files["file"].read()

        elif request.is_json and "image_base64" in (request.get_json() or {}):
            b64 = request.get_json()["image_base64"]

            if "," in b64 and b64.strip().startswith("data:"):
                b64 = b64.split(",", 1)[1]

            file_bytes = base64.b64decode(b64)

        else:
            return jsonify({
                "success": False,
                "error": "Send an image as multipart form field 'file' or JSON field 'image_base64'."
            }), 400

        if not file_bytes:
            return jsonify({
                "success": False,
                "error": "Empty image payload."
            }), 400

        result = process_image_bytes(file_bytes)

        return jsonify({
            "success": True,
            "data": result
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 422

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
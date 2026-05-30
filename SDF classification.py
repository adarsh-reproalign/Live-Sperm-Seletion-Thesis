import os
import cv2
import numpy as np
import pandas as pd


ROI_DIR = "sperm_roi"
CSV_DIR = "sperm_csv"
OUTPUT_SUMMARY = "SDF_classification_summary.csv"

summary_results = []


def extract_features(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    # Edge refinement
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    cnt = cv2.approxPolyDP(cnt, epsilon, True)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    circularity = (4 * np.pi * area) / (perimeter**2 + 1e-6)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h > 0 else 0

    hull = cv2.convexHull(cnt)
    solidity = area / (cv2.contourArea(hull) + 1e-6)

    return area, perimeter, circularity, aspect_ratio, solidity


def classify(area, circ, ar, solidity):

    # Strict NORMAL condition
    if (0.75 <= circ <= 1.0) and (0.8 <= ar <= 1.3) and (solidity >= 0.9):
        return "Normal sperm shape"

    # Abnormal cases
    elif ar > 1.5:
        return "Elongated (abnormal) sperm head"

    elif solidity < 0.85:
        return "Irregular / Fragmented sperm shape"

    # Borderline
    else:
        return "Borderline / Uncertain morphology"


for sperm_id in os.listdir(ROI_DIR):

    sperm_path = os.path.join(ROI_DIR, sperm_id)
    if not os.path.isdir(sperm_path):
        continue

    print(f"Processing {sperm_id}...")

    records = []
    total = 0
    normal_count = 0
    abnormal_count = 0

    
    for segment in os.listdir(sperm_path):
        seg_path = os.path.join(sperm_path, segment)

        if not os.path.isdir(seg_path):
            continue

        for img_name in os.listdir(seg_path):
            img_path = os.path.join(seg_path, img_name)

            roi = cv2.imread(img_path)
            if roi is None:
                continue

            feat = extract_features(roi)
            if feat is None:
                continue

            area, per, circ, ar, sol = feat
            label = classify(area, circ, ar, sol)

            total += 1

            if "Normal" in label:
                normal_count += 1
            else:
                abnormal_count += 1

            records.append({
                "sperm_id": sperm_id,
                "segment": segment,
                "image": img_name,
                "area": area,
                "perimeter": per,
                "circularity": circ,
                "aspect_ratio": ar,
                "solidity": sol,
                "classification": label
            })

  
    df_morph = pd.DataFrame(records)
    df_morph.to_csv(
        os.path.join(sperm_path, f"{sperm_id}_analysis.csv"),
        index=False
    )

 
    csv_path = os.path.join(CSV_DIR, f"{sperm_id}.csv")

    M_score = 0
    if os.path.exists(csv_path):
        df_motion = pd.read_csv(csv_path)
        speeds = df_motion["speed"].dropna().values

        if len(speeds) > 0:
            mean_speed = np.mean(speeds)
            stability = np.std(speeds)
            M_score = mean_speed - stability

   
    if total > 0:
        avg_circ = np.mean([r["circularity"] for r in records])
        avg_ar   = np.mean([r["aspect_ratio"] for r in records])
        avg_sol  = np.mean([r["solidity"] for r in records])

        Morph_score = (avg_circ + (1/(avg_ar+1e-6)) + avg_sol) / 3
    else:
        avg_circ = 0
        Morph_score = 0

  
    areas = [r["area"] for r in records]
    S_score = np.var(areas) if len(areas) > 0 else 0

  
    if (M_score < 0) or (Morph_score < 1.0) or (S_score > 50):
        SDF_classification = "Fragmented"
    else:
        SDF_classification = "Non-Fragmented"

   
    summary_results.append({
        "sperm_id": sperm_id,
        "total_frames": total,
        "normal_frames": normal_count,
        "abnormal_frames": abnormal_count,
        "M_score": M_score,
        "Morph_score": Morph_score,
        "Size_variance": S_score,
        "SDF classification": SDF_classification
    })



summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(OUTPUT_SUMMARY, index=False)

print("\nFULL PIPELINE COMPLETED!")
print(f"Saved: {OUTPUT_SUMMARY}")
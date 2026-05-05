import os
import cv2
import numpy as np
import pandas as pd

BASE_DIR = "sperm_roi"


def extract_features(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, binary = cv2.threshold(blur, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

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
    if circ > 0.8:
        return "Normal sperm shape"
    elif ar > 1.5:
        return "Elongated (abnormal) sperm head"
    elif solidity < 0.85:
        return "Irregular / Fragmented sperm shape"
    else:
        return "Borderline / Uncertain morphology"



summary_results = []

for sperm_id in os.listdir(BASE_DIR):

    sperm_path = os.path.join(BASE_DIR, sperm_id)
    if not os.path.isdir(sperm_path):
        continue

    print(f"Processing {sperm_id}...")

    records = []
    normal_count = 0
    abnormal_count = 0
    total = 0

    # Loop segments
    for segment in os.listdir(sperm_path):
        segment_path = os.path.join(sperm_path, segment)

        if not os.path.isdir(segment_path):
            continue

        # Loop images
        for img_name in os.listdir(segment_path):
            img_path = os.path.join(segment_path, img_name)

            roi = cv2.imread(img_path)
            if roi is None:
                continue

            feat = extract_features(roi)
            if feat is None:
                continue

            area, per, circ, ar, sol = feat
            label = classify(area, circ, ar, sol)

            # Count stats
            total += 1
            if "Normal" in label:
                normal_count += 1
            else:
                abnormal_count += 1

            # Save record
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

    # Save per-sperm CSV
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(sperm_path, f"{sperm_id}_analysis.csv"), index=False)

    # Final classification
    if total > 0:
        normal_ratio = normal_count / total
        final_class = "Normal" if normal_ratio > 0.6 else "Abnormal"
    else:
        normal_ratio = 0
        final_class = "Unknown"

    summary_results.append({
        "sperm_id": sperm_id,
        "total_images": total,
        "normal_count": normal_count,
        "abnormal_count": abnormal_count,
        "normal_ratio": normal_ratio,
        "final_class": final_class
    })



summary_df = pd.DataFrame(summary_results)
summary_df.to_csv("sperm_summary.csv", index=False)

print("✅ Analysis completed!")
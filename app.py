import cv2
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
import argparse
import pandas as pd
import os
from inference_sdk import InferenceHTTPClient
from collections import defaultdict
import math

sperm_history = defaultdict(list)   # track_id → list of records
roi_base_dir = "sperm_roi"
os.makedirs(roi_base_dir, exist_ok=True)
# ────────────────────────────────────────────────────────────────
# Kalman Filter Tracker
# ────────────────────────────────────────────────────────────────
class KalmanBoxTracker:
    """Standard bounding-box Kalman filter (constant-velocity model)."""

    def __init__(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w  = bbox[2] - bbox[0]
        h  = bbox[3] - bbox[1]

        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.array([[cx], [cy], [w], [h],
                            [0.], [0.], [0.], [0.]], dtype=float)

        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.eye(4, 8)

        self.Q = np.eye(8) * 1e-2
        self.Q[4:, 4:] *= 10

        self.R = np.eye(4) * 1e-1

        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 1000

        self.history = deque(maxlen=60)
        self.history.append(np.array(bbox, dtype=float))

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._state_to_bbox()

    def update(self, bbox):
        z = np.array([[(bbox[0] + bbox[2]) / 2],
                      [(bbox[1] + bbox[3]) / 2],
                      [bbox[2] - bbox[0]],
                      [bbox[3] - bbox[1]]], dtype=float)
        y  = z - self.H @ self.x
        S  = self.H @ self.P @ self.H.T + self.R
        K  = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.history.append(np.array(bbox, dtype=float))

    def reupdate(self, bbox_start, bbox_end, gap):
        """OMCTrack Observation-Centric Re-Update: fill gap with virtual observations."""
        for t in range(1, gap + 1):
            alpha = t / (gap + 1)
            virt = bbox_start + alpha * (bbox_end - bbox_start)
            self.update(virt.tolist())

    def _state_to_bbox(self):
        cx, cy, w, h = self.x[:4, 0]
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    def get_bbox(self):
        return self._state_to_bbox()


# ────────────────────────────────────────────────────────────────
# Appearance feature (HSV histogram + gradient histogram)
# ────────────────────────────────────────────────────────────────
def extract_feature(frame, bbox):
    x1, y1, x2, y2 = (max(0, int(v)) for v in bbox)
    x2 = min(frame.shape[1] - 1, x2)
    y2 = min(frame.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = cv2.resize(frame[y1:y2, x1:x2], (32, 32))

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    fh  = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    fs  = cv2.calcHist([hsv], [1], None, [8],  [0, 256]).flatten()
    fv  = cv2.calcHist([hsv], [2], None, [8],  [0, 256]).flatten()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)
    fmag = np.histogram(mag.flatten(), bins=12, range=(0, mag.max() + 1e-5))[0].astype(np.float32)
    fang = np.histogram(ang.flatten(), bins=12, range=(-np.pi, np.pi))[0].astype(np.float32)

    feat = np.concatenate([fh, fs, fv, fmag, fang]).astype(np.float32)
    n = np.linalg.norm(feat)
    return feat / (n + 1e-6)


def app_dist(a, b):
    if a is None or b is None:
        return 0.5
    return float(np.clip(1.0 - np.dot(a, b), 0.0, 1.0))


# ────────────────────────────────────────────────────────────────
# IOU helpers
# ────────────────────────────────────────────────────────────────
def iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def iou_batch(tracks_bbox, dets_bbox):
    C = np.zeros((len(tracks_bbox), len(dets_bbox)))
    for i, tb in enumerate(tracks_bbox):
        for j, db in enumerate(dets_bbox):
            C[i, j] = iou(tb, db)
    return C


# ────────────────────────────────────────────────────────────────
# OCM: Observation-Centric Momentum direction cost
# ────────────────────────────────────────────────────────────────
def _center(bbox):
    """Returns [cx, cy] as a 1D float array."""
    return np.array([(bbox[0] + bbox[2]) / 2.0,
                     (bbox[1] + bbox[3]) / 2.0], dtype=float)


def ocm_direction_cost(track_history, det_bbox, delta_t=3):
    hist = list(track_history)
    if len(hist) < 2:
        return 0.0

    ref_idx = max(0, len(hist) - 1 - delta_t)
    track_vec = _center(hist[-1]) - _center(hist[ref_idx])
    track_norm = np.linalg.norm(track_vec)
    if track_norm < 1e-3:
        return 0.0

    intent_vec = _center(np.array(det_bbox)) - _center(hist[-1])
    intent_norm = np.linalg.norm(intent_vec)
    if intent_norm < 1e-3:
        return 0.0

    cos_sim = np.dot(track_vec / track_norm, intent_vec / intent_norm)
    return float(np.clip((1.0 - cos_sim) / 2.0, 0.0, 1.0))


# ────────────────────────────────────────────────────────────────
# Camera Motion Compensation (ECC-based)
# ────────────────────────────────────────────────────────────────
class CameraMotionCompensator:
    def __init__(self, warp_mode=cv2.MOTION_EUCLIDEAN):
        self.warp_mode   = warp_mode
        self.prev_gray   = None
        self.warp_matrix = None

    def compensate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray   = gray
            self.warp_matrix = np.eye(2, 3, dtype=np.float32)
            return self.warp_matrix

        warp     = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
        try:
            _, warp = cv2.findTransformECC(
                self.prev_gray, gray, warp, self.warp_mode, criteria
            )
        except cv2.error:
            warp = np.eye(2, 3, dtype=np.float32)

        self.prev_gray   = gray
        self.warp_matrix = warp
        return warp

    @staticmethod
    def apply_to_bbox(warp, bbox):
        pts = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[3]],
        ], dtype=np.float32).reshape(-1, 1, 2)
        pts_t = cv2.transform(pts, warp).reshape(-1, 2)
        return [pts_t[0, 0], pts_t[0, 1], pts_t[1, 0], pts_t[1, 1]]


# ────────────────────────────────────────────────────────────────
# Single Track object
# ────────────────────────────────────────────────────────────────
class Track:
    NEW    = 'new'
    ACTIVE = 'active'
    LOST   = 'lost'

    def __init__(self, track_id, det, frame, conf):
        self.track_id  = track_id
        self.state     = Track.NEW
        self.kf        = KalmanBoxTracker(det)
        self.conf      = conf
        self.appearance = extract_feature(frame, det)
        # FIX 1: frames_since_update must be initialised to 0.
        # Without this, the attribute is missing and _age_and_transition
        # crashes or treats brand-new tracks as already-stale.
        self.frames_since_update = 0
        self.hit_streak = 1
        self.age        = 0
        self.last_obs   = np.array(det, dtype=float)

    def predict(self, warp=None):
        pred = self.kf.predict()
        if warp is not None:
            pred = CameraMotionCompensator.apply_to_bbox(warp, pred)
            cx = (pred[0] + pred[2]) / 2
            cy = (pred[1] + pred[3]) / 2
            self.kf.x[0, 0] = cx
            self.kf.x[1, 0] = cy
        return pred

    def update(self, det, frame, conf):
        self.kf.update(det)
        new_app = extract_feature(frame, det)
        if new_app is not None and self.appearance is not None:
            self.appearance = 0.85 * self.appearance + 0.15 * new_app
        elif new_app is not None:
            self.appearance = new_app
        self.conf = conf
        # FIX 2: reset on every successful match so the aging logic sees 0
        # and keeps the track in the active pool.
        self.frames_since_update = 0
        self.hit_streak += 1
        self.last_obs = np.array(det, dtype=float)

    def reactivate(self, det, frame, conf, gap):
        if gap > 0 and len(self.kf.history) > 0:
            self.kf.reupdate(self.kf.history[-1], np.array(det, dtype=float), gap)
        self.update(det, frame, conf)
        self.state = Track.ACTIVE
        self.frames_since_update = 0
        self.hit_streak = 1

    def get_bbox(self):
        return self.kf.get_bbox()


# ────────────────────────────────────────────────────────────────
# Tracker
# ────────────────────────────────────────────────────────────────
class ByteTrackOMCTracker:
    """
    3-stage association pipeline:
      Stage 1 (ByteTrack high-conf)  — IOU + appearance + OCM direction
      Stage 2 (ByteTrack low-conf)   — IOU only against unmatched active tracks
      Stage 3 (OMCTrack OPM re-ID)   — appearance-heavy match against lost pool
    """

    def __init__(
        self,
        high_thresh    = 0.50,
        low_thresh     = 0.10,
        match_iou      = 0.30,
        match_iou_low  = 0.20,
        reid_app_th    = 0.60,
        reid_iou_th    = 0.10,
        track_buffer   = 90,
        min_hits       = 1,
        mot_w          = 0.50,
        app_w          = 0.30,
        ocm_w          = 0.20,
        use_cmc        = True,
    ):
        self.high_thresh   = high_thresh
        self.low_thresh    = low_thresh
        self.match_iou     = match_iou
        self.match_iou_low = match_iou_low
        self.reid_app_th   = reid_app_th
        self.reid_iou_th   = reid_iou_th
        self.track_buffer  = track_buffer
        self.min_hits      = min_hits
        self.mot_w         = mot_w
        self.app_w         = app_w
        self.ocm_w         = ocm_w

        self.active_tracks: list[Track] = []
        self.lost_tracks:   list[Track] = []
        self.frame_id = 0
        self.next_id  = 0

        self.cmc = CameraMotionCompensator() if use_cmc else None

    # ── public entry point ────────────────────────────────────────
    def update(self, detections, frame):
        """
        detections: list of [x1, y1, x2, y2, conf, cls]
        returns:    list of [x1, y1, x2, y2, track_id, conf]
        """
        self.frame_id += 1

        warp = self.cmc.compensate(frame) if self.cmc else None

        for t in self.active_tracks + self.lost_tracks:
            t.predict(warp)

        high_dets = [d for d in detections if d[4] >= self.high_thresh]
        low_dets  = [d for d in detections if self.low_thresh <= d[4] < self.high_thresh]

        # Stage 1: high-conf vs active
        unmatched_active, unmatched_high = self._stage1(high_dets, frame)

        # Stage 2: low-conf vs remaining active
        unmatched_active2 = self._stage2(low_dets, unmatched_active, frame)

        # FIX 3: Tracks that were NOT matched in stage 1 or 2 must have
        # their staleness counter incremented NOW, before aging runs.
        # Previously this was skipped entirely, so ghost tracks lived forever.
        for idx in unmatched_active2:
            self.active_tracks[idx].frames_since_update += 1

        # Stage 3: re-ID from lost pool
        still_new = self._stage3_reid(
            [high_dets[i] for i in unmatched_high], frame
        )

        # Initialise brand-new tracks
        for det in still_new:
            t = Track(self.next_id, det[:4], frame, det[4])
            self.next_id += 1
            self.active_tracks.append(t)
            print(f"  [NEW]   Sperm #{t.track_id}  frame={self.frame_id}")

        # Age tracks, move stale ones to lost pool
        self._age_and_transition()

        out = []
        for t in self.active_tracks:
            if t.hit_streak >= self.min_hits or self.frame_id <= self.min_hits:
                bb = t.get_bbox()
                out.append([bb[0], bb[1], bb[2], bb[3], t.track_id, t.conf])
        return out

    # ── Stage 1 ──────────────────────────────────────────────────
    def _stage1(self, high_dets, frame):
        if not self.active_tracks or not high_dets:
            return list(range(len(self.active_tracks))), list(range(len(high_dets)))

        C = np.zeros((len(self.active_tracks), len(high_dets)))
        for i, t in enumerate(self.active_tracks):
            pred = t.get_bbox()
            for j, d in enumerate(high_dets):
                motion_cost = 1.0 - iou(pred, d[:4])
                app_cost    = app_dist(t.appearance, extract_feature(frame, d[:4]))
                ocm_cost    = ocm_direction_cost(t.kf.history, d[:4])
                C[i, j]     = (self.mot_w * motion_cost +
                               self.app_w * app_cost +
                               self.ocm_w * ocm_cost)

        rows, cols = linear_sum_assignment(C)

        matched_t, matched_d = set(), set()
        for r, c in zip(rows, cols):
            d = high_dets[c]
            if iou(self.active_tracks[r].get_bbox(), d[:4]) >= self.match_iou:
                self.active_tracks[r].update(d[:4], frame, d[4])
                self.active_tracks[r].state = Track.ACTIVE
                matched_t.add(r)
                matched_d.add(c)

        return ([r for r in range(len(self.active_tracks)) if r not in matched_t],
                [c for c in range(len(high_dets))           if c not in matched_d])

    # ── Stage 2 ──────────────────────────────────────────────────
    def _stage2(self, low_dets, unmatched_t_idx, frame):
        if not low_dets or not unmatched_t_idx:
            return list(unmatched_t_idx)

        tracks = [self.active_tracks[i] for i in unmatched_t_idx]
        iou_m  = iou_batch([t.get_bbox() for t in tracks],
                            [d[:4] for d in low_dets])
        rows, cols = linear_sum_assignment(1.0 - iou_m)

        still_unmatched = list(unmatched_t_idx)
        for r, c in zip(rows, cols):
            if iou_m[r, c] >= self.match_iou_low:
                d = low_dets[c]
                tracks[r].update(d[:4], frame, d[4])
                tracks[r].state = Track.ACTIVE
                still_unmatched.remove(unmatched_t_idx[r])

        return still_unmatched

    # ── Stage 3: OMCTrack Occlusion Perception Module ────────────
    def _stage3_reid(self, unmatched_dets, frame):
        if not self.lost_tracks or not unmatched_dets:
            return unmatched_dets

        C = np.zeros((len(self.lost_tracks), len(unmatched_dets)))
        for i, t in enumerate(self.lost_tracks):
            pred = t.get_bbox()
            for j, d in enumerate(unmatched_dets):
                app_cost    = app_dist(t.appearance, extract_feature(frame, d[:4]))
                motion_cost = 1.0 - iou(pred, d[:4])
                ocm_cost    = ocm_direction_cost(t.kf.history, d[:4])
                C[i, j] = 0.55 * app_cost + 0.30 * motion_cost + 0.15 * ocm_cost

        rows, cols = linear_sum_assignment(C)

        matched_lost_rows = set()
        matched_d = set()
        for r, c in zip(rows, cols):
            t        = self.lost_tracks[r]
            d        = unmatched_dets[c]
            pred_iou = iou(t.get_bbox(), d[:4])
            app_d    = app_dist(t.appearance, extract_feature(frame, d[:4]))

            if app_d <= self.reid_app_th or pred_iou >= self.reid_iou_th:
                gap = t.frames_since_update
                t.reactivate(d[:4], frame, d[4], gap)
                self.active_tracks.append(t)
                matched_lost_rows.add(r)
                matched_d.add(c)
                print(f"  [RE-ID] Sperm #{t.track_id} recovered  "
                      f"app={app_d:.3f}  iou={pred_iou:.3f}  "
                      f"gap={gap}f  frame={self.frame_id}")

        # FIX 4: Remove re-identified tracks from lost pool correctly.
        # The original code built the set of lost-row indices inside the
        # comprehension using a set of *detection* column indices, which
        # produced wrong removals. We now track matched rows directly.
        self.lost_tracks = [t for i, t in enumerate(self.lost_tracks)
                            if i not in matched_lost_rows]

        return [d for j, d in enumerate(unmatched_dets) if j not in matched_d]

    # ── Aging ─────────────────────────────────────────────────────
    def _age_and_transition(self):
        """
        FIX 5: Simplified, correct aging.

        Original had two problems:
          a) It incremented frames_since_update for already-lost tracks a
             *second* time at the bottom of the method, so lost tracks aged
             at 2× speed and were pruned too aggressively.
          b) The 'grace period' check (frames_since_update <= 2) used the
             counter BEFORE it was incremented for newly-unmatched tracks,
             so unmatched tracks were always kept regardless of age.

        Now: FIX 3 (in update()) handles newly-unmatched tracks. Here we
        only decide which active tracks move to LOST, and prune the lost
        pool by track_buffer frames.
        """
        still_active = []
        newly_lost   = []

        for t in self.active_tracks:
            t.age += 1
            if t.frames_since_update == 0:
                # Matched this frame — keep active
                still_active.append(t)
            elif t.frames_since_update <= 2:
                # Grace period: unmatched for 1–2 frames, stay active
                still_active.append(t)
            else:
                t.state = Track.LOST
                newly_lost.append(t)

        self.active_tracks = still_active

        # Merge into lost pool, then prune by age
        self.lost_tracks = self.lost_tracks + newly_lost
        self.lost_tracks = [t for t in self.lost_tracks
                            if t.frames_since_update <= self.track_buffer]

        # Increment staleness for tracks that remain in the lost pool
        # (only once, only here — not duplicated elsewhere)
        for t in self.lost_tracks:
            t.frames_since_update += 1

    @property
    def num_unique(self):
        return self.next_id


# ────────────────────────────────────────────────────────────────
# Roboflow client
# ────────────────────────────────────────────────────────────────
CLIENT   = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="ANLwTWOL88uZwQ3Pm3tO"
)
MODEL_ID  = "sperm-detection-visem/1"
FRAME_DIR = "saved_frames"
os.makedirs(FRAME_DIR, exist_ok=True)


def run(model_path, video_path, output_path=None, conf_thresh=0.25):
    tracker = ByteTrackOMCTracker(
        high_thresh   = 0.45,
        low_thresh    = 0.10,
        match_iou     = 0.25,
        match_iou_low = 0.15,
        reid_app_th   = 0.62,
        reid_iou_th   = 0.08,
        track_buffer  = 90,
        min_hits      = 1,
        mot_w         = 0.50,
        app_w         = 0.30,
        ocm_w         = 0.20,
        use_cmc       = True,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open {video_path}"); return

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if output_path:
        writer = cv2.VideoWriter(output_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    np.random.seed(7)
    palette: dict = {}
    fc = 0
    MAX_FRAMES = 30
    track_results = []

    print("\n══ ByteTrack + OMCTrack  Hybrid Sperm Tracker ══")
    print(f"   Video : {video_path}")
    print(f"   Model : {model_path}")
    print(f"   CMC   : camera motion compensation ON")
    print(f"   Re-ID : OMCTrack Occlusion Perception Module ON")
    print("─" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if fc <= MAX_FRAMES:
            frame_path = os.path.join(FRAME_DIR, f"frame_{fc:04d}.jpg")
            cv2.imwrite(frame_path, frame)

        fc += 1

        # ── Roboflow inference ────────────────────────────────────
        # FIX 6: dets must be built OUTSIDE the loop over predictions,
        # not reset inside it. The original code had `dets = []` inside
        # the for-loop body (wrong indentation), so only the very last
        # prediction was ever appended. All other detections were silently
        # discarded, starving the tracker and causing massive ID switching.
        dets = []
        result = CLIENT.infer(frame, model_id=MODEL_ID)
        if "predictions" in result:
            for p in result["predictions"]:
                x, y, w, h = p["x"], p["y"], p["width"], p["height"]
                conf = p["confidence"]
                x1, y1 = x - w / 2, y - h / 2
                x2, y2 = x + w / 2, y + h / 2
                dets.append([x1, y1, x2, y2, conf, 0])

        tracks = tracker.update(dets, frame)

        for trk in tracks:
            x1, y1, x2, y2, tid, conf = trk
            x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)

            # Center
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2

            # Previous position
            prev_x, prev_y = None, None
            distance = 0
            speed = 0

            if len(sperm_history[tid]) > 0:
                prev_x = sperm_history[tid][-1]["x"]
                prev_y = sperm_history[tid][-1]["y"]

                distance = math.sqrt((xc - prev_x)**2 + (yc - prev_y)**2)
                speed = distance * fps   # pixels per second

            # Save history
            sperm_history[tid].append({
                "frame": fc,
                "x": xc,
                "y": yc,
                "prev_x": prev_x,
                "prev_y": prev_y,
                "distance": distance,
                "speed": speed
            })

            # ───── ROI SAVE ─────
            roi = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
            if roi.size > 0:

                # Create folder per sperm + per 30 frames
                segment_id = fc // 30
                save_dir = os.path.join(roi_base_dir, f"sperm_{tid}", f"segment_{segment_id}")
                os.makedirs(save_dir, exist_ok=True)

                roi_path = os.path.join(save_dir, f"frame_{fc}.jpg")
                cv2.imwrite(roi_path, roi)

            # ───── DRAW ─────
            if tid not in palette:
                palette[tid] = tuple(np.random.randint(60, 220, 3).tolist())
            col = palette[tid]

            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"{tid}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

            # Trajectory
            t_obj = next((t for t in tracker.active_tracks if t.track_id == tid), None)
            if t_obj and len(t_obj.kf.history) > 1:
                # FIX 7: _center() returns a float array — cast to int for cv2.line.
                pts = []
                for b in list(t_obj.kf.history):
                    c = _center(b)
                    pts.append((int(c[0]), int(c[1])))
              

        lost_n = len(tracker.lost_tracks)
        cv2.putText(
            frame,
            f"Frame {fc:5d} | Active {len(tracks)} | Lost-pool {lost_n} | "
            f"Total IDs {tracker.num_unique}  | ByteTrack+OMCTrack",
            (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2
        )

        cv2.imshow("ByteTrack + OMCTrack  Sperm Tracker", frame)
        if writer:
            writer.write(frame)
        if fc % 60 == 0:
            print(f"  frame={fc:5d}  active={len(tracks)}  "
                  f"lost_pool={lost_n}  total_ids={tracker.num_unique}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(
        track_results,
        columns=["frameid", "trackerid", "x_center", "y_center", "width", "height"]
    )
    df.to_csv("sperm_tracking_predictions.csv", index=False)
    csv_dir = "sperm_csv"
    os.makedirs(csv_dir, exist_ok=True)

    for tid, records in sperm_history.items():
        df = pd.DataFrame(records)

        df.rename(columns={
            "x": "current_x",
            "y": "current_y"
        }, inplace=True)

        df.to_csv(os.path.join(csv_dir, f"sperm_{tid}.csv"), index=False)

    print("─" * 55)
    print(f"Done. Frames={fc}  Unique IDs issued={tracker.num_unique}")
    if output_path:
        print(f"Saved → {output_path}")


# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ByteTrack + OMCTrack Sperm Tracker")
    ap.add_argument("--model",  default="runs/detect/runs/train/v8_640_l/weights/best.pt")
    ap.add_argument("--video",  default="Sperm_video.mp4")
    ap.add_argument("--output", default="output_Sperm_video.mp4")
    ap.add_argument("--conf",   type=float, default=0.25)
    args = ap.parse_args()
    run(args.model, args.video, args.output, args.conf)
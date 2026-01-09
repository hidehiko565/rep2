# -*- coding: utf-8 -*-
"""
顔・左目・右目・鼻・口のリアルタイム検出。
・顔＝緑枠で括る
・左目・右目・鼻・口は画像で置き換える（/assets フォルダに置くこと）
・年齢・性別表示あり。
"""

import cv2
import numpy as np
import os, platform, time
from enum import Enum, auto

BASE = os.path.dirname(os.path.abspath(__file__))
WINDOW_NAME = "Eyes Overlay / Nose / Mouth + Age&Gender"  # ← 1つに統一

# ===== オーバーレイ画像 =====
L_EYE_IMG_PATH = os.path.join(BASE, "assets", "left_eye.png")
R_EYE_IMG_PATH = os.path.join(BASE, "assets", "right_eye.png")
MOUTH_IMG_PATH = os.path.join(BASE, "assets", "mouth.png")
NOSE_IMG_PATH  = os.path.join(BASE, "assets", "nose.png")

# ===== DNN（年齢・性別） =====
AGE_PROTO    = os.path.join(BASE, "models", "age_deploy.prototxt")
AGE_MODEL    = os.path.join(BASE, "models", "age_net.caffemodel")
GENDER_PROTO = os.path.join(BASE, "models", "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(BASE, "models", "gender_net.caffemodel")

AGE_LIST    = ['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60-100']
GENDER_LIST = ['Male','Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  # BGR

# ===== 表示保持のパラメータ =====
HOLD_SEC = 0.5   # 検出が途切れた後も目画像を出し続ける秒数
SMOOTHING = 0  # 位置の平滑化係数（0=なし, 0.2〜0.5推奨）

class Mode(Enum):
    TEXT = auto()
    ICON = auto()

MODE = Mode.ICON 
#MODE = Mode.TEXT

def log(msg):
    print(msg, flush=True)

def safe_load_net(proto_path, model_path):
    if not (os.path.exists(proto_path) and os.path.exists(model_path)):
        log(f"[WARN] DNNモデルが見つかりません: {proto_path}, {model_path}")
        return None
    try:
        return cv2.dnn.readNetFromCaffe(proto_path, model_path)
    except Exception as e:
        log(f"[WARN] DNNモデルのロードに失敗: {e}")
        return None

def predict_age_gender(face_bgr, age_net, gender_net):
    if face_bgr is None or face_bgr.size == 0: return None
    blob = cv2.dnn.blobFromImage(face_bgr, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False, crop=False)
    age_label = None; age_conf = None
    gender_label = None; gender_conf = None
    if age_net is not None:
        age_net.setInput(blob)
        ap = age_net.forward().flatten()
        ai = int(np.argmax(ap)); age_label = AGE_LIST[ai]; age_conf = float(ap[ai])
    if gender_net is not None:
        gender_net.setInput(blob)
        gp = gender_net.forward().flatten()
        gi = int(np.argmax(gp)); gender_label = GENDER_LIST[gi]; gender_conf = float(gp[gi])
    if age_label is None and gender_label is None: return None
    return age_label, age_conf, gender_label, gender_conf

def pick_largest(rects):
    if len(rects) == 0: return None
    areas = [(w*h, (x,y,w,h)) for (x,y,w,h) in rects]
    areas.sort(key=lambda t: t[0], reverse=True)
    return areas[0][1]

def overlay_image(frame_bgr, overlay_img, x, y, w, h, force_opaque=True):
    if overlay_img is None or overlay_img.size == 0 or w <= 0 or h <= 0: return
    H, W = frame_bgr.shape[:2]
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + w); y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0: return
    target_w = x1 - x0; target_h = y1 - y0
    inter = cv2.INTER_AREA if (overlay_img.shape[1] > target_w or overlay_img.shape[0] > target_h) else cv2.INTER_LINEAR
    resized = cv2.resize(overlay_img, (target_w, target_h), interpolation=inter)
    if resized.shape[2] == 4:  # BGRA
        overlay_rgb = resized[:, :, :3]
        alpha = resized[:, :, 3].astype(np.float32) / 255.0
        if force_opaque: alpha[:] = 1.0
        alpha = alpha[..., None]
        roi = frame_bgr[y0:y1, x0:x1].astype(np.float32)
        blended = overlay_rgb.astype(np.float32) * alpha + roi * (1.0 - alpha)
        frame_bgr[y0:y1, x0:x1] = blended.astype(np.uint8)
    else:
        frame_bgr[y0:y1, x0:x1] = resized

def open_camera():
    sysname = platform.system()
    backends_try = [cv2.CAP_ANY]
    if sysname == "Windows": backends_try = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    elif sysname == "Darwin": backends_try = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else: backends_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
    for be in backends_try:
        cap = cv2.VideoCapture(0, be)
        if cap.isOpened():
            log(f"[INFO] Camera opened with backend={be}")
            return cap
        cap.release()
    return None

def smooth_box(prev, new, alpha):
    """位置の平滑化（移動平均）。prev/new: (x,y,w,h) または None"""
    if prev is None or new is None or alpha <= 0.0:
        return new
    px, py, pw, ph = prev; nx, ny, nw, nh = new
    sx = int(px*(1-alpha) + nx*alpha)
    sy = int(py*(1-alpha) + ny*alpha)
    sw = int(pw*(1-alpha) + nw*alpha)
    sh = int(ph*(1-alpha) + nh*alpha)
    return (sx, sy, sw, sh)

def main():
    # モード：テキストモードかアイコンモードか
    log(f"[INFO] MODE: {'ICON' if MODE == Mode.ICON else 'TEXT'}")

    # 目画像
    left_eye_img  = cv2.imread(L_EYE_IMG_PATH,  cv2.IMREAD_UNCHANGED)
    right_eye_img = cv2.imread(R_EYE_IMG_PATH, cv2.IMREAD_UNCHANGED)
    mouth_img     = cv2.imread(MOUTH_IMG_PATH, cv2.IMREAD_UNCHANGED)
    nose_img      = cv2.imread(NOSE_IMG_PATH, cv2.IMREAD_UNCHANGED)
    log(f"[INFO] left_eye.png: {'OK' if left_eye_img is not None else 'MISSING'}")
    log(f"[INFO] right_eye.png: {'OK' if right_eye_img is not None else 'MISSING'}")
    log(f"[INFO] mouse.png: {'OK' if mouth_img is not None else 'MISSING'}")
    log(f"[INFO] nose.png: {'OK' if nose_img is not None else 'MISSING'}")

    # Haar
    hb = cv2.data.haarcascades
    face_cascade_path  = os.path.join(hb, "haarcascade_frontalface_default.xml")
    eye_cascade_path   = os.path.join(hb, "haarcascade_eye.xml")
    mouth_cascade_path = os.path.join(hb, "haarcascade_mcs_mouth.xml")
    smile_cascade_path = os.path.join(hb, "haarcascade_smile.xml")
    nose_cascade_path  = os.path.join(hb, "haarcascade_mcs_nose.xml")

    face_cascade  = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade   = cv2.CascadeClassifier(eye_cascade_path)
    mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
    smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
    nose_cascade  = cv2.CascadeClassifier(nose_cascade_path)

    log(f"[INFO] face_cascade  loaded: {not face_cascade.empty()}")
    log(f"[INFO] eye_cascade   loaded: {not eye_cascade.empty()}")
    log(f"[INFO] mouth_cascade loaded: {not mouth_cascade.empty()}")
    log(f"[INFO] nose_cascade  loaded: {not nose_cascade.empty()}")

    if face_cascade.empty() or eye_cascade.empty() or nose_cascade.empty():
        raise RuntimeError("Haar Cascade のロードに失敗（顔/目/鼻）。OpenCVのインストールとファイルパスをご確認ください。")

    # DNN
    age_net    = safe_load_net(AGE_PROTO, AGE_MODEL)
    gender_net = safe_load_net(GENDER_PROTO, GENDER_MODEL)
    if age_net is None or gender_net is None:
        log("[INFO] 年齢・性別推定は無効です（モデル未設定）。検出のみ実行します。")

    # カメラ
    cap = open_camera()
    if cap is None:
        raise RuntimeError("カメラを開けませんでした。Zoom/Teamsの終了、OSのカメラ権限、USB接続をご確認ください。")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    log("[INFO] 'q' キーで終了します。")

    # ===== 左右目の保持用ステート =====
    left_state = {"last_box": None, "last_time": 0.0}
    right_state= {"last_box": None, "last_time": 0.0}

    while True:
        ok, frame = cap.read()
        if not ok:
            log("[WARN] フレーム取得に失敗"); break

        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80))
        for (x, y, w, h) in faces:
            # 顔枠（緑）
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            
            # 顔のグレースケールとRGB
            face_gray = gray[y:y + h, x:x + w]
            face_bgr  = frame[y:y + h, x:x + w]

            # ===== 目（上半分→左右分割） =====
            top_h = int(h * 0.55)
            eyes_roi_gray = face_gray[0:top_h, :]
            mid_x = w // 2
            left_roi  = eyes_roi_gray[:, 0:mid_x]
            right_roi = eyes_roi_gray[:, mid_x:w]

            left_eyes  = eye_cascade.detectMultiScale(left_roi,  scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
            right_eyes = eye_cascade.detectMultiScale(right_roi, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
            le_detect = pick_largest(left_eyes)
            re_detect = pick_largest(right_eyes)

            # ---- 左目：検出があれば更新、無ければ保持表示 ----
            if le_detect is not None:
                ex, ey, ew, eh = le_detect
                # ROI→フレーム座標
                box = (x + ex, y + ey, ew, eh)
                # 平滑化
                box = smooth_box(left_state["last_box"], box, SMOOTHING)
                left_state["last_box"] = box
                left_state["last_time"] = now
            # 表示（検出あり or 保持時間以内）
            if left_state["last_box"] is not None and (now - left_state["last_time"] <= HOLD_SEC):
                bx, by, bw, bh = left_state["last_box"]
                if (left_eye_img is not None) and MODE == Mode.ICON:
                    overlay_image(frame, left_eye_img, bx, by, bw, bh, force_opaque=False)
                else:
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 2)
                    cv2.putText(frame, "Left Eye", (x + ex, y + ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # ---- 右目：検出があれば更新、無ければ保持表示 ----
            if re_detect is not None:
                ex, ey, ew, eh = re_detect
                abs_ex = x + mid_x + ex
                box = (abs_ex, y + ey, ew, eh)
                box = smooth_box(right_state["last_box"], box, SMOOTHING)
                right_state["last_box"] = box
                right_state["last_time"] = now
            if right_state["last_box"] is not None and (now - right_state["last_time"] <= HOLD_SEC):
                bx, by, bw, bh = right_state["last_box"]
                if (right_eye_img is not None) and MODE == Mode.ICON:
                    overlay_image(frame, right_eye_img, bx, by, bw, bh, force_opaque=False)
                else:
                    cv2.rectangle(frame, (abs_ex, y + ey), (abs_ex + ew, y + ey + eh), (0, 0, 255), 2)
                    cv2.putText(frame, "Right Eye", (abs_ex, y + ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # ===== 鼻（中央帯） =====
            band_top = int(h * 0.35); band_bottom = int(h * 0.75)
            roi_gray_nose = face_gray[band_top:band_bottom, :]
            noses = nose_cascade.detectMultiScale(roi_gray_nose, scaleFactor=1.2, minNeighbors=8, minSize=(28, 28))
            nose_box = pick_largest(noses)
            nose_bottom_y = None
            if nose_box is not None:
                nx, ny, nw, nh = nose_box
                abs_ny = y + band_top + ny
                if (nose_img is not None) and MODE == Mode.ICON:
                    overlay_image(frame, nose_img, x + nx, abs_ny, nw, nh, force_opaque=False)
                else:
                    cv2.rectangle(frame, (x + nx, abs_ny), (x + nx + nw, abs_ny + nh), (0, 0, 255), 2)
                nose_bottom_y = abs_ny + nh

            # ===== 口（下部 + 幾何フィルタ） =====
            mouth_top_rel = int(h * 0.55)
            roi_gray_mouth = face_gray[mouth_top_rel:h, :]
            if not mouth_cascade.empty():
                mouths_raw = mouth_cascade.detectMultiScale(roi_gray_mouth, scaleFactor=1.2, minNeighbors=15, minSize=(30, 20))
            else:
                smiles_raw = smile_cascade.detectMultiScale(roi_gray_mouth, scaleFactor=1.3, minNeighbors=25, minSize=(30, 20))
                mouths_raw = smiles_raw

            filtered = []
            for (mx, my, mw, mh) in mouths_raw:
                abs_my = y + mouth_top_rel + my
                cy = abs_my + mh / 2.0
                aspect = mw / float(mh + 1e-6)
                cond_center = (cy > (y + h * 0.55))
                cond_nose   = True if nose_bottom_y is None else (cy > nose_bottom_y)
                cond_aspect = (aspect >= 1.2)
                if cond_center and cond_nose and cond_aspect:
                    filtered.append((mx, my, mw, mh))
            mouth_box = pick_largest(filtered)
            if mouth_box is not None:
                mx, my, mw, mh = mouth_box
                abs_my = y + mouth_top_rel + my
                if (mouth_img is not None) and MODE == Mode.ICON:
                    overlay_image(frame, mouth_img, x + mx, abs_my, mw, mh, force_opaque=False)
                else:
                    cv2.rectangle(frame, (x + mx, abs_my), (x + mx + mw, abs_my + mh), (0, 0, 255), 2)

            # ===== 年齢・性別 =====
            info_text = "Age/Gender: N/A"
            res = predict_age_gender(face_bgr, age_net, gender_net)
            if res is not None:
                age_label, age_conf, gender_label, gender_conf = res
                age_part = f"Age: {age_label}" + (f" ({age_conf*100:.0f}%)" if age_conf is not None else "")
                gender_part = f"Gender: {gender_label}" + (f" ({gender_conf*100:.0f}%)" if gender_conf is not None else "")
                info_text = f"{age_part} | {gender_part}"
            (tw, th), base = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            tx, ty = x, y - 10
            if ty - th - base < 0:
                ty = y + h + th + 10
            cv2.rectangle(frame, (tx, ty - th - base), (tx + tw, ty + base), (0, 0, 0), -1)
            cv2.putText(frame, info_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            log("[INFO] Quit"); break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

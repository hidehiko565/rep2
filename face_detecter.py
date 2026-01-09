# -*- coding: utf-8 -*-
"""
リアルタイムでカメラから顔・目・口・鼻を検出し、枠を描画。
顔の枠（緑）、目・口・鼻の枠（赤）。
検出した顔の上に推定した年齢・性別を表示。

必要:
    pip install opencv-python numpy

DNNモデル（Caffe; 年齢・性別）:
    AGE_PROTO    = "models/age_deploy.prototxt"
    AGE_MODEL    = "models/age_net.caffemodel"
    GENDER_PROTO = "models/gender_deploy.prototxt"
    GENDER_MODEL = "models/gender_net.caffemodel"
"""

import cv2
import numpy as np
import os

# ===== DNN（年齢・性別）モデルのパス =====
BASE = os.path.dirname(os.path.abspath(__file__))
AGE_PROTO    = os.path.join(BASE, "models", "age_deploy.prototxt")
AGE_MODEL    = os.path.join(BASE, "models", "age_net.caffemodel")
GENDER_PROTO = os.path.join(BASE, "models", "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(BASE, "models", "gender_net.caffemodel")

AGE_LIST    = ['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60-100']
GENDER_LIST = ['Male','Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  # BGR

def safe_load_net(proto_path, model_path):
    if not (os.path.exists(proto_path) and os.path.exists(model_path)):
        print(f"[WARN] DNNモデルが見つかりません: {proto_path}, {model_path}")
        return None
    try:
        net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        return net
    except Exception as e:
        print(f"[WARN] DNNモデルのロードに失敗しました: {e}")
        return None

def predict_age_gender(face_bgr, age_net, gender_net):
    if face_bgr is None or face_bgr.size == 0:
        return None
    blob = cv2.dnn.blobFromImage(face_bgr, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False, crop=False)

    age_label = None; age_conf = None
    gender_label = None; gender_conf = None

    if age_net is not None:
        age_net.setInput(blob)
        age_preds = age_net.forward().flatten()
        age_idx = int(np.argmax(age_preds))
        age_label = AGE_LIST[age_idx]
        age_conf = float(age_preds[age_idx])

    if gender_net is not None:
        gender_net.setInput(blob)
        gender_preds = gender_net.forward().flatten()
        gender_idx = int(np.argmax(gender_preds))
        gender_label = GENDER_LIST[gender_idx]
        gender_conf = float(gender_preds[gender_idx])

    if age_label is None and gender_label is None:
        return None
    return age_label, age_conf, gender_label, gender_conf

# テキスト描画（顔の上 or 下）
def plot_text(frame, x, y, w, h, text_str):
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 0.6
    text_th = 1
    (text_w, text_h), baseline = cv2.getTextSize(text_str, text_font, text_scale, text_th)

    text_x = x
    text_y = y - 10
    if text_y - text_h - baseline < 0:
        text_y = y + h + text_h + 10
    cv2.rectangle(frame, (text_x, text_y - text_h - baseline), (text_x + text_w, text_y + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text_str, (text_x, text_y), text_font, text_scale, (255, 255, 255), text_th, cv2.LINE_AA)

def main():
    # ===== Haar Cascade（顔・目・口・鼻） =====
    haar_base = cv2.data.haarcascades
    face_cascade_path  = os.path.join(haar_base, "haarcascade_frontalface_default.xml")
    eye_cascade_path   = os.path.join(haar_base, "haarcascade_eye.xml")
    smile_cascade_path = os.path.join(haar_base, "haarcascade_smile.xml")      # 口の近似（笑顔）
    mouth_cascade_path = os.path.join(haar_base, "haarcascade_mcs_mouth.xml")  # 口
    nose_cascade_path  = os.path.join(haar_base, "haarcascade_mcs_nose.xml")   # 鼻

    face_cascade  = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade   = cv2.CascadeClassifier(eye_cascade_path)
    smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
    mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
    nose_cascade  = cv2.CascadeClassifier(nose_cascade_path)

    if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty() or nose_cascade.empty():
        raise RuntimeError("Haar Cascade のロードに失敗しました。OpenCV のインストールとファイルパスをご確認ください。")

    # ===== DNN（年齢・性別） =====
    age_net    = safe_load_net(AGE_PROTO, AGE_MODEL)
    gender_net = safe_load_net(GENDER_PROTO, GENDER_MODEL)
    if age_net is None or gender_net is None:
        print("[INFO] 年齢・性別推定は無効です（モデル未設定）。検出のみ実行します。")

    # ===== カメラ初期化 =====
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("カメラを開けませんでした。接続と権限をご確認ください。")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] 'q' キーで終了します。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] フレームの取得に失敗")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # 顔検出
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            # 顔枠（緑）
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 5)

            face_gray = gray[y:y + h, x:x + w]
            face_bgr  = frame[y:y + h, x:x + w]

            # テキスト描画（顔の上 or 下）
            text_font = cv2.FONT_HERSHEY_SIMPLEX
            text_scale = 0.6
            text_th = 1

            # 最もそれっぽい部品を一つ選ぶ（サイズ最大を採用）
            def pick_largest(rects):
                if len(rects) == 0:
                    return None
                areas = [(ew * eh, (ex, ey, ew, eh)) for (ex, ey, ew, eh) in rects]
                areas.sort(key=lambda t: t[0], reverse=True)
                return areas[0][1]

            # ===== 目（顔の上半分を左右に分割して検出） =====
            top_h = int(h * 0.55)
            eyes_roi_gray = face_gray[0:top_h, :]

            # 左右に分割
            mid_x = w // 2
            left_roi  = eyes_roi_gray[:, 0:mid_x]
            right_roi = eyes_roi_gray[:, mid_x:w]

            # 左目検出（左半分）
            left_eyes = eye_cascade.detectMultiScale(
                left_roi, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20)
            )
            # 右目検出（右半分）
            right_eyes = eye_cascade.detectMultiScale(
                right_roi, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20)
            )
            left_eye  = pick_largest(left_eyes)
            right_eye = pick_largest(right_eyes)

            # 左目の描画（座標は顔ROI→フレームに変換）
            if left_eye is not None:
                ex, ey, ew, eh = left_eye
                # 左半分の原点は (x, y) から見て (x + 0, y + 0)
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 2)
                cv2.putText(frame, "Left Eye", (x + ex, y + ey - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # 右目の描画（右半分の原点は face の左端から mid_x オフセット）
            if right_eye is not None:
                ex, ey, ew, eh = right_eye
                abs_ex = x + mid_x + ex
                cv2.rectangle(frame, (abs_ex, y + ey), (abs_ex + ew, y + ey + eh), (0, 0, 255), 2)
                cv2.putText(frame, "Right Eye", (abs_ex, y + ey - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # ===== 鼻（顔中央付近; 上半分〜下半分の境界周辺） =====
            # 鼻は目と口の間にあることが多いので、顔の中央帯を優先探索
            band_top = int(h * 0.35)
            band_bottom = int(h * 0.75)
            roi_gray_nose = face_gray[band_top:band_bottom, :]
            noses = nose_cascade.detectMultiScale(
                roi_gray_nose,
                scaleFactor=1.2,
                minNeighbors=8,     # 誤検出抑制
                minSize=(28, 28)
            )
            nose_box = pick_largest(noses)
            nose_bottom_y = None   # 鼻の下限、口の判定に利用
            if nose_box is not None:
                nx, ny, nw, nh = nose_box
                abs_ny = y + band_top + ny
                cv2.rectangle(frame, (x + nx, abs_ny), (x + nx + nw, abs_ny + nh), (0, 0, 255), 2)
                nose_bottom_y = abs_ny + nh
                cv2.putText(frame, "Nose", (x + nx, abs_ny - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            # ===== 口（顔の下部に限定 + 幾何フィルタ） =====
            # 下 45% を探索領域に設定
            mouth_top_rel = int(h * 0.55)
            roi_gray_mouth = face_gray[mouth_top_rel:h, :]
            # 口を検出
            mouths_raw = mouth_cascade.detectMultiScale(
                roi_gray_mouth, scaleFactor=1.2, minNeighbors=15, minSize=(30, 20)
            )

            # 幾何フィルタ: 顔の中線より下、鼻下端より下（あれば）、アスペクト比閾値
            filtered = []
            for (mx, my, mw, mh) in mouths_raw:
                abs_my = y + mouth_top_rel + my
                cx = x + mx + mw / 2.0
                cy = abs_my + mh / 2.0

                aspect = mw / float(mh + 1e-6)
                # 条件1: 顔中央線より下
                cond_center = (cy > (y + h * 0.55))
                # 条件2: 鼻が見つかっている場合は鼻下端より下
                cond_nose   = True if nose_bottom_y is None else (cy > nose_bottom_y)
                # 条件3: 口は横長
                cond_aspect = (aspect >= 1.2)

                if cond_center and cond_nose and cond_aspect:
                    filtered.append((mx, my, mw, mh))

            mouth_box = pick_largest(filtered)
            if mouth_box is not None:
                mx, my, mw, mh = mouth_box
                abs_my = y + mouth_top_rel + my
                cv2.rectangle(frame, (x + mx, abs_my), (x + mx + mw, abs_my + mh), (0, 0, 255), 2)
                cv2.putText(frame, "Mouth", (x + mx, abs_my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            # ===== 年齢・性別の推定 =====
            info_text = None
            result = predict_age_gender(face_bgr, age_net, gender_net)
            if result is not None:
                age_label, age_conf, gender_label, gender_conf = result
                age_part = f"Age: {age_label}" + (f" ({age_conf*100:.0f}%)" if age_conf is not None else "")
                gender_part = f"Gender: {gender_label}" + (f" ({gender_conf*100:.0f}%)" if gender_conf is not None else "")
                info_text = f"{age_part} | {gender_part}"
            else:
                info_text = "Age/Gender: N/A"
            plot_text(frame, x, y, w, h, info_text)

        cv2.imshow("Real-time Face/Eyes/Mouth/Nose Detection with Age & Gender", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

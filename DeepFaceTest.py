# 顔画像を使って、お菓子をレコメンドするアプリ
# with TkInter, DeepFace

import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from deepface import DeepFace


# ファイル選択ダイアログを開いて、画像ファイルを選択させる関数
def select_image():
    # ユーザーにファイルを選択させる
    file_path = filedialog.askopenfilename()
    if file_path:
        # 画像ファイルを読み込む
        img = read_img(file_path)
        # 画像から、DeepFaceで顔分析を実行する
        analyze_with_DeepFace(img)
        # 結果をウィンドウに表示する
        display_image(img)

# 画像ファイルを読み込む
def read_img(image_path: str):
    # 画像をOpenCVで読み込む
    try:
        img = cv2.imread(image_path)
    except Exception:
        print("画像の読み込みに失敗しました")

    return img


# 顔検出を行う関数
def detect_faces(image):
    print('detect_faces...')
    # Haar Cascade分類器を使って顔検出を行う
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 顔を検出 (scaleFactor=1.1, minNeighbors=5は標準的な値)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 検出された顔の周りに四角形を描画
    # region を使って最も左上を選ぶ（原点からの距離で算出）
    def upper_left_key(f):
        y = f[1]
        x = f[0]
        ret_val = y**2 + x**2
        print(f"x: {x}, y:{y}, dist:{ret_val}")
        return ret_val
    faces = sorted(faces, key=upper_left_key)

    cnt = 0
    for (x, y, w, h) in faces:
        if cnt == 0:
            rect_color = (0, 255, 0)
        else:
            rect_color = (0, 128, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), rect_color, 3)
        cnt += 1

# 画像をウィンドウに表示する関数
def display_image(image):
    # 画像サイズを調整
    # リサイズ後の縦の長さ
    re_length = 300
    # 現在の縦横のサイズ (h:縦, w:横)
    h, w = image.shape[:2]
    # 変換する倍率を計算
    re_h = re_w = re_length/h
    # アスペクト比を固定して画像を変換
    re_image = cv2.resize(image, dsize=None, fx=re_h , fy=re_w)

    # 顔検出を実行
    detect_faces(re_image)

    # OpenCVの画像をPillow形式に変換
    image_rgb = cv2.cvtColor(re_image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(image_rgb)

    # 画像をtkinterウィジェットに表示できるように変換
    imgtk = ImageTk.PhotoImage(image=im_pil)

    # ラベルに画像を設定して表示
    img_kao.config(image=imgtk)
    img_kao.image = imgtk  # ガベージコレクションされないように参照を保持

# ===== DeepFace ===== 
# 分析結果から推定感情を取り出す
def get_estimated_emotion(face_details):
    emotion_est = face_details.get('emotion', {})
    print(emotion_est)

    if emotion_est:
        # 標準出力には感情の内、最大のものを書いておく
        dom_e = max(emotion_est, key=emotion_est.get)
        ret_str = f"感情: {dom_e} ({emotion_est.get(dom_e, 0):.2f}%)"

    return ret_str

# 分析結果から推定年齢を取り出す
def get_estimated_age(face_details):
    age_est = face_details.get("age")
    ret_str = f"推定年齢: {age_est}"
    return ret_str

# 画像から、DeepFaceで顔分析を実行する関数
def analyze_with_DeepFace(img_bgr):
    # BGR→RGB に変換
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print('analyze_with_DeepFace...')
    result = DeepFace.analyze(
        img,
        actions=['age', 'emotion'],
        detector_backend='retinaface',   # 例: 'retinaface' が強力
        enforce_detection=True           # 例外を出す（動作確認後はFalseでもOK）
    )
    detect_faces = len(result)
    if detect_faces == 0:
        self.label_text.setText("顔が検出できませんでした")
        self.label_photo.clear()
        self.label_candy.clear()
        self.label_score.clear()
        return
    print('顔を'+ str(detect_faces) +'件検出')

    # region を使って最も左上を選ぶ（原点からの距離で算出）
    def upper_left_key(r):
        region = r.get('region', {})
        y, x = [region.get('y', float('inf')), region.get('x', float('inf'))]
        ret_val = y**2 + x**2
        print(f"x: {x}, y:{y}, dist:{ret_val}")
        return ret_val

    results_sorted = sorted(result, key=upper_left_key)
    face = results_sorted[0]

    # 1) 推定感情
    str_emotion = get_estimated_emotion(face)
    print(str_emotion)

    # 2) 推定年齢
    str_age = get_estimated_age(face)
    print(str_age)

    # 画面に表示
    text_emotion.config(text=str_emotion)
    text_age.config(text=str_age)



# ===== アプリ起動 =====
if __name__ == "__main__":
    # GUIアプリケーションの作成
    root = tk.Tk()
    root.title("顔認識アプリケーション")

    # 画像を選択するボタンを作成
    btn = tk.Button(root, text="画像を選択", command=select_image)

    # 画像表示用のラベルを作成
    img_kao = tk.Label(root)

    # テキスト表示用のラベルを作成
    text_emotion = tk.Label(root)
    text_age = tk.Label(root)

    # 配置
    btn.grid(
        column=0,
        columnspan=2,
        row=0,
    )
    img_kao.grid(
        column=0,
        columnspan=2,
        row=1,
    )
    text_emotion.grid(
        column=0,
        row=2,
    )
    text_age.grid(
        column=1,
        row=2,
    )

    # GUIを開始
    root.mainloop()

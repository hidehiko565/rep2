import cv2

# 分類器のパス（OpenCV付属のHaar分類器）
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# 分類器を読み込む
face_cascade = cv2.CascadeClassifier(cascade_path)

# 読み込み確認
if face_cascade.empty():
    raise IOError("分類器の読み込みに失敗しました。パスを確認してください。")

# カメラを起動（0はデフォルトカメラ）
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔検出
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 検出した顔に枠を描画
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 結果を表示
    cv2.imshow('Face Detection', frame)

    # ESCキーで終了
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

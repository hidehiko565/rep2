import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD

import os
from pathlib import Path
from pdf2image import convert_from_path

# メインウィンドウを作成
window = TkinterDnD.Tk()
window.title("PDF Drag and Drop")

# テキストボックスを作成
textbox = tk.Text(window)
textbox.pack()

# ドラッグアンドドロップを有効にする
textbox.drop_target_register(DND_FILES)

#----------
# PDFから画像を抽出
def p2i(file_name):
    # poppler/binを環境変数PATHに追加する
    poppler_dir = Path(__file__).parent.absolute() / "poppler/Library/bin"
    os.environ["PATH"] += os.pathsep + str(poppler_dir)

    # PDFファイルのパス
    pdf_path = Path(file_name)
    print(pdf_path)

    # PDF -> Image に変換（150dpi）
    pages = convert_from_path(str(pdf_path), 150)

    # 画像ファイルを1ページずつ保存
    image_dir = Path("./img_data")
    for i, page in enumerate(pages):
        # PNGで保存
        file_name = pdf_path.stem + "_{:03d}".format(i + 1) + ".png"
        image_path = image_dir / file_name
        page.save(str(image_path), "PNG")

# ドラッグアンドドロップされたときの処理
def drop(event):
    # ドロップされたファイルパスを取得
    filepath = event.data
    # テキストボックスに追加
    textbox.insert(tk.END, filepath + "\n")
    # PDFから画像を抽出
    if filepath.endswith(".pdf"):
        p2i(filepath)
    else:
        print("please d2d *.pdf")
    # イベントを処理済みにする
    return event.action

# ドラッグアンドドロップのイベントをバインドする
textbox.dnd_bind("<<Drop>>", drop)

# メインループを開始
window.mainloop()

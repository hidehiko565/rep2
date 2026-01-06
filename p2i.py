import os
from pathlib import Path
from pdf2image import convert_from_path

# カレントフォルダのファイル名をリストで取得
files = os.listdir()
pdf_files = [f for f in files if f.endswith(".pdf")]
file_name = pdf_files[0]  # 先頭のみ
print(file_name)

#----------
# poppler/binを環境変数PATHに追加する
poppler_dir = Path(__file__).parent.absolute() / "poppler/Library/bin"
os.environ["PATH"] += os.pathsep + str(poppler_dir)

# PDFファイルのパス
pdf_path = Path("./" + file_name)
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

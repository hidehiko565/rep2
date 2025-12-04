### HtmlDiffクラスでファイルを比較
import os
import difflib

dir_path = r'./'

file1_name = 'DigitalWorkshop.py'
file2_name = 'DigitalWorkshop_withDeepFace.py'

file1_path = os.path.join(dir_path, file1_name)
file2_path = os.path.join(dir_path, file2_name)


# ① 文字コードを明示して読み込む（UTF-8前提／BOMつきも考慮するなら 'utf-8-sig'）
#    不明な場合は errors='replace' で置換して落ちないようにする
with open(file1_path, 'r', encoding='utf-8', errors='replace') as f1:
    from_lines = f1.read().splitlines()
with open(file2_path, 'r', encoding='utf-8', errors='replace') as f2:
    to_lines = f2.read().splitlines()

diff = difflib.HtmlDiff()

output_name = 'diff.html'
output_path = os.path.join(dir_path, output_name)

# ② 出力もUTF-8で
with open(output_path, 'w', encoding='utf-8', newline='') as out:
    html = diff.make_file(from_lines, to_lines, fromdesc=file1_name, todesc=file2_name)
    out.write(html)

# -*- coding: utf-8 -*-
"""
テキストファイルを読み取り、指定の画像サイズ・色・フォントでPNG画像に描画するスクリプト。
- 完全ローカル処理（クラウド不要）
- 日本語の自動改行（折り返し）に対応
- 画像サイズ、色、フォント、余白、行間、配置、入出力ファイル名はコード内で指定可能
"""

from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
import os

# =========================
# 設定（ここを編集）
# =========================
# 読み込むテキストファイル
INPUT_TXT_PATH = "input.txt"
# 出力するPNGファイル
OUTPUT_PNG_PATH = "output.png"

# 画像の幅(px)
IMG_WIDTH = 1024
# 画像の高さ(px)  ※AUTO_RESIZE_CANVAS=True の場合は高さ自動調整
IMG_HEIGHT = 1024
# 背景色 (R, G, B, A) / 3タプル(RGB)でもOK
BACKGROUND_COLOR = (0, 0, 0)
# 文字色 (R, G, B, A) / 3タプルでもOK
TEXT_COLOR = (255, 0, 0)

# 日本語対応フォントのローカルパス
FONT_PATH = r"C:\Windows\Fonts\meiryo.ttc"
# フォントサイズ(px)
FONT_SIZE = 36

# 上下左右の余白(px)
PADDING_TOP = 40
PADDING_BOTTOM = 40
PADDING_LEFT = 40
PADDING_RIGHT = 40

# 行間倍率（1.0=詰め、1.3〜1.5おすすめ）
LINE_SPACING_MULT = 1.3
# アライン：'left' | 'center' | 'right'
ALIGN = "center"
# True: テキストに合わせて高さを自動拡張（幅は固定）
AUTO_RESIZE_CANVAS = True
# 自動拡張する場合の上限高さ（Noneで制限なし）
MAX_HEIGHT = None

# 入力テキストファイルのエンコーディング
ENCODING = "utf-8"

# =========================
# ここから下は通常そのままでOK
# =========================

def read_text(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding) as f:
        return f.read()

def load_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"フォントが見つかりません: {font_path}")
    return ImageFont.truetype(font_path, font_size)

def measure_line_height(font: ImageFont.FreeTypeFont) -> int:
    # フォントメトリクスから行の基本高さを取得（日本語も安定）
    ascent, descent = font.getmetrics()
    return ascent + descent

def wrap_text_to_width(text: str,
                       font: ImageFont.FreeTypeFont,
                       max_text_width: int) -> List[str]:
    """
    日本語（スペースが少ない）も考慮した折返し。
    改行（\n）は尊重し、各段落内は文字単位で幅に収まるよう分割。
    """
    # 計測用のダミー描画オブジェクト（textbboxを使うため）
    dummy_img = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy_img)

    def text_width(s: str) -> int:
        if not s:
            return 0
        bbox = draw.textbbox((0, 0), s, font=font)
        return bbox[2] - bbox[0]

    wrapped_lines: List[str] = []
    paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    for para in paragraphs:
        if para == "":
            # 空行はそのまま入れる
            wrapped_lines.append("")
            continue

        line = ""
        for ch in para:
            candidate = line + ch
            if text_width(candidate) <= max_text_width:
                line = candidate
            else:
                # 既に1文字も入っていないのに溢れる場合（極大文字など）は強制改行
                if line == "":
                    wrapped_lines.append(ch)
                    line = ""
                else:
                    wrapped_lines.append(line)
                    line = ch
        # 残りがあれば追加
        if line != "":
            wrapped_lines.append(line)

    return wrapped_lines

def render_text_to_image(lines: List[str],
                         font: ImageFont.FreeTypeFont,
                         image_size: Tuple[int, int],
                         bg_color,
                         text_color,
                         paddings: Tuple[int, int, int, int],
                         align: str = "left",
                         auto_resize: bool = False,
                         max_height: int = None) -> Image.Image:
    """
    折り返し済みの lines を、指定サイズ（必要なら高さ自動調整）で描画してImageを返す。
    """
    pad_left, pad_top, pad_right, pad_bottom = paddings
    base_width, base_height = image_size

    # 行高と行間（ピクセル）を決定
    base_line_height = measure_line_height(font)
    line_gap = int(base_line_height * (LINE_SPACING_MULT - 1.0))
    line_height = base_line_height + line_gap

    # 必要な高さを算出（空行も1行として扱う）
    text_lines_count = len(lines)
    needed_height = pad_top + pad_bottom + (text_lines_count * base_line_height) + (max(0, text_lines_count - 1) * line_gap)

    out_width = base_width
    out_height = base_height

    if auto_resize:
        out_height = max(base_height, needed_height)
        if max_height is not None:
            out_height = min(out_height, max_height)

    # 画像生成（RGBAを許容：背景に透過成分を使える）
    mode = "RGBA" if (isinstance(bg_color, tuple) and len(bg_color) == 4) else "RGB"
    img = Image.new(mode, (out_width, out_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # 描画開始位置
    y = pad_top

    # 各行を描画
    for line in lines:
        # 行の幅を計測
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]

        # 配置に応じてX座標を決める
        x_left = pad_left
        x_right = out_width - pad_right
        drawable_width = x_right - x_left

        if align == "center":
            x = x_left + (drawable_width - line_w) // 2
        elif align == "right":
            x = x_right - line_w
        else:  # 'left'
            x = x_left

        # 描画（アンチエイリアスはPillowが内部的に処理）
        draw.text((x, y), line, font=font, fill=text_color)

        # 次の行へ
        y += line_height

        # 自動拡張なしの場合、下端を超えたら打ち切り
        if not auto_resize and (y + base_line_height + pad_bottom) > out_height:
            break

    return img

def main():
    # 入力読み込み
    text = read_text(INPUT_TXT_PATH, encoding=ENCODING)

    # フォントロード
    font = load_font(FONT_PATH, FONT_SIZE)

    # 折り返し可能なテキスト幅（左右余白を除いた幅）
    drawable_width = IMG_WIDTH - (PADDING_LEFT + PADDING_RIGHT)
    if drawable_width <= 0:
        raise ValueError("画像幅が小さすぎます。左右余白を見直してください。")

    # 折り返し
    lines = wrap_text_to_width(text, font, drawable_width)

    # 画像生成
    img = render_text_to_image(
        lines=lines,
        font=font,
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        bg_color=BACKGROUND_COLOR,
        text_color=TEXT_COLOR,
        paddings=(PADDING_LEFT, PADDING_TOP, PADDING_RIGHT, PADDING_BOTTOM),
        align=ALIGN,
        auto_resize=AUTO_RESIZE_CANVAS,
        max_height=MAX_HEIGHT
    )

    # 保存（PNG）
    img.save(OUTPUT_PNG_PATH, format="PNG")
    print(f"Done: {OUTPUT_PNG_PATH}")

if __name__ == "__main__":
    main()

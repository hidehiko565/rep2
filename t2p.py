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
# アライン： ('left' | 'center' | 'right'), ('top' | 'middle' | 'bottom')
ALIGN = "center"
V_ALIGN = "middle"
# True: テキストに合わせて高さを自動拡張（幅は固定）
AUTO_RESIZE_CANVAS = False
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
    ascent, descent = font.getmetrics()
    return ascent + descent

def wrap_text_to_width(text: str,
                       font: ImageFont.FreeTypeFont,
                       max_text_width: int) -> List[str]:
    """日本語（スペース無し）も考慮した折返し。改行は尊重。"""
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
            wrapped_lines.append("")
            continue

        line = ""
        for ch in para:
            candidate = line + ch
            if text_width(candidate) <= max_text_width:
                line = candidate
            else:
                if line == "":
                    wrapped_lines.append(ch)
                    line = ""
                else:
                    wrapped_lines.append(line)
                    line = ch
        if line != "":
            wrapped_lines.append(line)

    return wrapped_lines

def compute_text_block_height(lines: List[str],
                              font: ImageFont.FreeTypeFont,
                              line_spacing_mult: float) -> Tuple[int, int, int]:
    base_line_height = measure_line_height(font)
    line_gap = int(base_line_height * (line_spacing_mult - 1.0))
    line_height = base_line_height + line_gap

    n = len(lines)
    # 空行も1行として扱う（描画時も同様）
    block_height = (n * base_line_height) + (max(0, n - 1) * line_gap)
    return block_height, base_line_height, line_gap

def render_text_to_image(lines: List[str],
                         font: ImageFont.FreeTypeFont,
                         image_size: Tuple[int, int],
                         bg_color,
                         text_color,
                         paddings: Tuple[int, int, int, int],
                         align: str = "left",
                         v_align: str = "top",
                         auto_resize: bool = False,
                         max_height: int = None) -> Image.Image:
    """
    折り返し済みの lines を、指定サイズ（必要なら高さ自動調整）で描画してImageを返す。
    """
    pad_left, pad_top, pad_right, pad_bottom = paddings
    base_width, base_height = image_size

    # テキストブロックの高さを算出
    block_h, base_line_height, line_gap = compute_text_block_height(
        lines, font, LINE_SPACING_MULT
    )
    line_height = base_line_height + line_gap

    # 画像高さの決定
    needed_height = pad_top + block_h + pad_bottom
    out_width = base_width
    out_height = base_height
    if auto_resize:
        out_height = max(base_height, needed_height)
        if max_height is not None:
            out_height = min(out_height, max_height)

    # 画像生成
    mode = "RGBA" if (isinstance(bg_color, tuple) and len(bg_color) == 4) else "RGB"
    img = Image.new(mode, (out_width, out_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # 描画領域（余白を除く高さ）
    drawable_top = pad_top
    drawable_bottom = out_height - pad_bottom
    drawable_h = max(0, drawable_bottom - drawable_top)

    # 縦アライン: テキストブロックの描画開始Yを決める
    if v_align not in ("top", "middle", "bottom"):
        v_align = "top"

    if auto_resize:
        # 自動拡張時は論理上 top と同様（空白が出ないようキャンバスが伸びる）
        y = drawable_top
    else:
        if v_align == "top":
            y = drawable_top
        elif v_align == "middle":
            y = drawable_top + max(0, (drawable_h - block_h) // 2)
        else:  # 'bottom'
            y = drawable_bottom - block_h

    # 各行を描画
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]

        x_left = pad_left
        x_right = out_width - pad_right
        drawable_width = x_right - x_left

        if align == "center":
            x = x_left + (drawable_width - line_w) // 2
        elif align == "right":
            x = x_right - line_w
        else:
            x = x_left

        # 領域をはみ出す場合の打ち切り
        if not auto_resize and (y + base_line_height) > drawable_bottom:
            break

        draw.text((x, y), line, font=font, fill=text_color)
        y += line_height

    return img

def main():
    text = read_text(INPUT_TXT_PATH, encoding=ENCODING)
    font = load_font(FONT_PATH, FONT_SIZE)

    drawable_width = IMG_WIDTH - (PADDING_LEFT + PADDING_RIGHT)
    if drawable_width <= 0:
        raise ValueError("画像幅が小さすぎます。左右余白を見直してください。")

    lines = wrap_text_to_width(text, font, drawable_width)

    img = render_text_to_image(
        lines=lines,
        font=font,
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        bg_color=BACKGROUND_COLOR,
        text_color=TEXT_COLOR,
        paddings=(PADDING_LEFT, PADDING_TOP, PADDING_RIGHT, PADDING_BOTTOM),
        align=ALIGN,
        v_align=V_ALIGN,
        auto_resize=AUTO_RESIZE_CANVAS,
        max_height=MAX_HEIGHT
    )

    img.save(OUTPUT_PNG_PATH, format="PNG")
    print(f"Done: {OUTPUT_PNG_PATH}")

if __name__ == "__main__":
    main()

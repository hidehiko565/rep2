# -*- coding: utf-8 -*-
"""
テキスト → PNG（縦組み / 横組み対応、縁取り（アウトライン）対応、透過PNG対応）
- 完全ローカル処理（クラウド不要）
- 縦組み：右→左に列、上→下に文字（ASCII/半角は90度回転オプションあり）
- 横組み：日本語の自動折返し対応
- 左右/上下アライン、余白、行間、フォント、縁取り、透明背景など指定可
"""

from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
import os
import math

# =========================
# 設定（ここを編集）
# =========================
INPUT_TXT_PATH = "input.txt"             # 読み込むテキストファイル
OUTPUT_PNG_PATH = "output.png"           # 出力PNG
WRITING_MODE = "vertical"                # "vertical"（デフォルト） | "horizontal"

IMG_WIDTH = 1024                         # 画像の幅(px)
IMG_HEIGHT = 1024                        # 画像の高さ(px)  ※縦組みでは固定推奨
BACKGROUND_COLOR = (255, 255, 255, 0)    # 透過背景（A=0）をデフォに
TEXT_COLOR = (255, 128, 0, 255)          # 黄色 (R,G,B,A) 例: #FFFF00

# 縦書きに向いたフォントを指定（環境に合わせて変更）
# 例: Windows: C:\Windows\Fonts\YuMincho.ttc / YuGothM.ttc / meiryo.ttc
#     macOS: /System/Library/Fonts/ヒラギノ明朝 ProN.ttc
#     Linux: /usr/share/fonts/truetype/noto/NotoSerifCJK-Regular.ttc
FONT_PATH = r"C:\Windows\Fonts\YuMincho.ttc"
FONT_PATH = r"C:\Windows\Fonts\meiryo.ttc"
FONT_SIZE = 72

PADDING_LEFT = 100
PADDING_TOP = 10
PADDING_RIGHT = 40
PADDING_BOTTOM = 10

# 行間倍率（縦/横共通）
LINE_SPACING_MULT = 1.2

# アライン
# 'left'|'center'|'right'  ← 縦組みでは「ブロックの左右位置」
# 'top'|'middle'|'bottom'  ※縦組みで複数列になる場合は各列Top固定
ALIGN = "left"
V_ALIGN = "middle"

# 自動拡張（縦組みでは固定推奨）
AUTO_RESIZE_CANVAS = False
MAX_HEIGHT = None

ENCODING = "utf-8"

# --- 縦組み用の追加オプション ---
VERTICAL_FLOW = "rtl"                    # 'rtl'（右→左）固定推奨
COL_SPACING = 10                         # 列と列の間隔（px）
ROTATE_ASCII = True                      # ASCII/半角を 90°回転（縦書きで自然に）
ROTATE_CLOCKWISE = True                  # True=時計回り（-90°）、False=反時計回り（+90°）

# === アウトライン（縁取り）設定（ご指定） ===
ENABLE_STROKE = True
STROKE_WIDTH = 8
STROKE_FILL = (255, 255, 255, 255)             # 縁取りの色
# =========================


def read_text(path: str, encoding: str) -> str:
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def load_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"フォントが見つかりません: {font_path}")
    return ImageFont.truetype(font_path, font_size)


def measure_line_height(font: ImageFont.FreeTypeFont) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent


# =========================
# 横組み（従来版：参考/保持）
# =========================
def wrap_text_to_width(text: str,
                       font: ImageFont.FreeTypeFont,
                       max_text_width: int) -> List[str]:
    """横組み用の折返し。改行は尊重。縁取り幅も考慮して計測。"""
    dummy_img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy_img)

    stroke_w = STROKE_WIDTH if ENABLE_STROKE and STROKE_WIDTH > 0 else 0

    def text_width(s: str) -> int:
        if not s:
            return 0
        bbox = draw.textbbox((0, 0), s, font=font, stroke_width=stroke_w)
        return bbox[2] - bbox[0]

    wrapped_lines: List[str] = []
    paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    for para in paragraphs:
        if para == "":
            wrapped_lines.append("")
            continue

        line = ""
        for ch in para:
            cand = line + ch
            if text_width(cand) <= max_text_width:
                line = cand
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
    n = len(lines)
    block_height = (n * base_line_height) + (max(0, n - 1) * line_gap)
    return block_height, base_line_height, line_gap


def render_text_to_image_horizontal(lines: List[str],
                                    font: ImageFont.FreeTypeFont,
                                    image_size: Tuple[int, int],
                                    bg_color,
                                    text_color,
                                    paddings: Tuple[int, int, int, int],
                                    align: str = "left",
                                    v_align: str = "top",
                                    auto_resize: bool = False,
                                    max_height: int = None) -> Image.Image:
    pad_left, pad_top, pad_right, pad_bottom = paddings
    base_width, base_height = image_size

    block_h, base_line_height, line_gap = compute_text_block_height(lines, font, LINE_SPACING_MULT)
    line_height = base_line_height + line_gap

    # キャンバス高さ決定
    needed_height = pad_top + block_h + pad_bottom
    out_width = base_width
    out_height = base_height
    if auto_resize:
        out_height = max(base_height, needed_height)
        if max_height is not None:
            out_height = min(out_height, max_height)

    img = Image.new("RGBA", (out_width, out_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # 描画領域
    drawable_left = pad_left
    drawable_right = out_width - pad_right
    drawable_width = max(0, drawable_right - drawable_left)
    drawable_top = pad_top
    drawable_bottom = out_height - pad_bottom
    drawable_h = max(0, drawable_bottom - drawable_top)

    # 縦アライン
    if auto_resize:
        y = drawable_top
    else:
        if v_align == "middle":
            y = drawable_top + max(0, (drawable_h - block_h) // 2)
        elif v_align == "bottom":
            y = drawable_bottom - block_h
        else:
            y = drawable_top

    stroke_w = STROKE_WIDTH if ENABLE_STROKE and STROKE_WIDTH > 0 else 0
    stroke_args = {"stroke_width": stroke_w, "stroke_fill": STROKE_FILL} if stroke_w > 0 else {}

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_w)
        line_w = bbox[2] - bbox[0]

        if align == "center":
            x = drawable_left + (drawable_width - line_w) // 2
        elif align == "right":
            x = drawable_right - line_w
        else:
            x = drawable_left

        # はみ出しチェック（必要なら）
        if not auto_resize and (y + (bbox[3] - bbox[1])) > drawable_bottom:
            break

        # 描画（縁取り付き）
        draw.text((x, y), line, font=font, fill=text_color, **stroke_args)
        y += line_height

    return img


# =========================
# 縦組み
# =========================
def is_halfwidth_or_ascii(ch: str) -> bool:
    code = ord(ch)
    return (code < 128) or (0xFF61 <= code <= 0xFF9F)  # ASCII または 半角ｶﾅ


def render_text_to_image_vertical(text: str,
                                  font: ImageFont.FreeTypeFont,
                                  image_size: Tuple[int, int],
                                  bg_color,
                                  text_color,
                                  paddings: Tuple[int, int, int, int],
                                  align: str = "right",
                                  v_align: str = "top",
                                  flow: str = "rtl",
                                  col_spacing: int = 8,
                                  rotate_ascii: bool = True,
                                  rotate_clockwise: bool = True,
                                  auto_resize: bool = False,
                                  max_height: int = None) -> Image.Image:
    """
    縦組み：上→下、列は右→左（flow='rtl'）
    - ALIGN は「ブロック（列集合）の左右位置」（right/center/left）
    - V_ALIGN は「1列に収まる場合」に有効（複数列では各列Top固定）
    """
    if flow != "rtl":
        raise NotImplementedError("縦組みは 'rtl'（右→左）に対応")

    pad_left, pad_top, pad_right, pad_bottom = paddings
    base_width, base_height = image_size

    # 行送り（縦方向の1ステップ）
    base_line_height = measure_line_height(font)
    line_gap = int(base_line_height * (LINE_SPACING_MULT - 1.0))
    line_advance = base_line_height + line_gap

    # キャンバス高さ決定（縦は基本固定を推奨）
    needed_height = pad_top + base_line_height + pad_bottom  # 最低限
    out_width = base_width
    out_height = base_height
    if auto_resize:
        out_height = max(base_height, needed_height)
        if max_height is not None:
            out_height = min(out_height, max_height)

    # 画像生成（透過対応）
    img = Image.new("RGBA", (out_width, out_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # 描画領域
    drawable_left = pad_left
    drawable_right = out_width - pad_right
    drawable_width = max(0, drawable_right - drawable_left)
    drawable_top = pad_top
    drawable_bottom = out_height - pad_bottom
    drawable_h = max(0, drawable_bottom - drawable_top)

    if drawable_h <= 0 or drawable_width <= 0:
        return img

    # ストローク設定
    stroke_w = STROKE_WIDTH if ENABLE_STROKE and STROKE_WIDTH > 0 else 0
    stroke_args = {"stroke_width": stroke_w, "stroke_fill": STROKE_FILL} if stroke_w > 0 else {}

    # 列幅（最大文字幅で決める。縁取り込みで計測）
    dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(dummy)

    def char_bbox(ch: str):
        return d2.textbbox((0, 0), ch, font=font, stroke_width=stroke_w)

    max_char_w = 0
    for ch in text.replace("\r\n", "\n").replace("\r", "\n"):
        if ch == "\n":
            continue
        bbox = char_bbox(ch)
        w = bbox[2] - bbox[0]
        if w > max_char_w:
            max_char_w = w
    if max_char_w == 0:
        max_char_w = base_line_height  # フォールバック

    col_width = max_char_w
    col_advance = col_width + col_spacing

    # 1列に入る行数
    rows_per_col = max(1, drawable_h // line_advance)

    # 総行数（改行も1行とカウント）
    total_rows = 0
    for ch in text.replace("\r\n", "\n").replace("\r", "\n"):
        total_rows += 1  # 文字でも改行でも1行進める

    # 必要列数
    needed_cols = max(1, math.ceil(total_rows / rows_per_col))

    # 実際に置ける最大列数
    max_cols = max(1, drawable_width // col_advance)
    use_cols = min(needed_cols, max_cols)

    # ブロック幅（列集合の幅）
    block_w = (use_cols * col_advance) - col_spacing

    # ブロック横位置（アライン）
    if align == "center":
        block_left = drawable_left + (drawable_width - block_w) // 2
    elif align == "left":
        block_left = drawable_left
    else:  # 'right'
        block_left = drawable_right - block_w

    # 1列に収まる場合のみ V_ALIGN で縦位置調整（複数列はTop固定）
    if needed_cols == 1:
        content_h = min(total_rows, rows_per_col) * line_advance - line_gap
        if v_align == "middle":
            start_y = drawable_top + max(0, (drawable_h - content_h) // 2)
        elif v_align == "bottom":
            start_y = drawable_bottom - content_h
        else:
            start_y = drawable_top
    else:
        start_y = drawable_top

    # 右端の列（col_idx=0）の起点X（flow=rtl）
    first_col_left = block_left + (use_cols - 1) * col_advance

    # 実描画
    col_idx = 0
    y = start_y
    x_col_left = first_col_left

    def paste_rotated_char(ch_img: Image.Image, x: int, y: int):
        # 透明を維持してペースト
        img.paste(ch_img, (x, y), ch_img)

    for ch in text.replace("\r\n", "\n").replace("\r", "\n"):
        if ch == "\n":
            # 空行として1ステップ進める
            y += line_advance
        else:
            # 1文字の描画用メトリクス（縁取り込み）
            bbox = d2.textbbox((0, 0), ch, font=font, stroke_width=stroke_w)
            ch_w, ch_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # 列内センタリング
            draw_x = x_col_left + (col_width - ch_w) // 2
            draw_y = y

            if rotate_ascii and is_halfwidth_or_ascii(ch):
                # 回転描画：縁取りが切れないようにパディングを確保
                pad = max(1, stroke_w + 2)
                tmp = Image.new("RGBA", (ch_w + 2 * pad, ch_h + 2 * pad), (0, 0, 0, 0))
                tdraw = ImageDraw.Draw(tmp)
                tdraw.text((pad, pad), ch, font=font, fill=text_color, **stroke_args)
                angle = -90 if rotate_clockwise else 90
                rot = tmp.rotate(angle, expand=True)
                rx, ry = rot.size
                px = x_col_left + (col_width - rx) // 2
                py = y
                paste_rotated_char(rot, px, py)
            else:
                # そのまま描画（縁取り付き）
                draw.text((draw_x, draw_y), ch, font=font, fill=text_color, **stroke_args)

            # 次の行へ
            y += line_advance

        # 行がはみ出したら次の列へ（右→左）
        if (y + base_line_height) > drawable_bottom:
            col_idx += 1
            if col_idx >= use_cols:
                break  # 横方向の残りスペースがないので打ち切り
            x_col_left = first_col_left - col_idx * col_advance
            y = drawable_top  # 複数列時はTop固定

    return img


# =========================
# メイン
# =========================
def main():
    text = read_text(INPUT_TXT_PATH, encoding=ENCODING)
    font = load_font(FONT_PATH, FONT_SIZE)

    if WRITING_MODE.lower() == "vertical":
        img = render_text_to_image_vertical(
            text=text,
            font=font,
            image_size=(IMG_WIDTH, IMG_HEIGHT),
            bg_color=BACKGROUND_COLOR,
            text_color=TEXT_COLOR,
            paddings=(PADDING_LEFT, PADDING_TOP, PADDING_RIGHT, PADDING_BOTTOM),
            align=ALIGN,
            v_align=V_ALIGN,
            flow=VERTICAL_FLOW,
            col_spacing=COL_SPACING,
            rotate_ascii=ROTATE_ASCII,
            rotate_clockwise=ROTATE_CLOCKWISE,
            auto_resize=AUTO_RESIZE_CANVAS,
            max_height=MAX_HEIGHT
        )
    else:
        # 横組み（参考/保持）
        drawable_width = IMG_WIDTH - (PADDING_LEFT + PADDING_RIGHT)
        if drawable_width <= 0:
            raise ValueError("画像幅が小さすぎます。左右余白を見直してください。")
        lines = wrap_text_to_width(text, font, drawable_width)
        img = render_text_to_image_horizontal(
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

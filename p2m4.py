
# -*- coding: utf-8 -*-
"""
PDF をスライドショー形式の MP4 に変換（引数なし・固定設定版 / MoviePy 1.0.3 想定）

■ 特徴
- moviepy.editor を使わず、必要モジュールのみ直接 import（matplotlib を避けて NumPy 2.x の衝突を回避）
- crossfadein を使わず、fadein/fadeout＋時間オーバーラップでクロスフェードを再現
- BGM は不足分を自動ループし、音量は afx.volumex で調整
- PyMuPDF (fitz) 優先、無ければ pdf2image（Poppler が必要）

■ 事前インストール例（Miniconda base環境）
    python -m pip install "moviepy==1.0.3" "pillow<11.0" numpy imageio-ffmpeg proglog
    python -m pip install PyMuPDF     # 推奨（Poppler不要）
    # または
    python -m pip install pdf2image   # WindowsはPopplerを導入してPATHへ

"""

import os
import sys
from typing import List, Tuple
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor

# --- ユーザー設定（ここだけ書き換えればOK） --------------------
# 入力PDF
INPUT_PDF   = "20251008_2.pdf"
# 出力動画
OUTPUT_MP4  = "20251008_2.mp4"
# BGM（未指定 or ファイル無しなら無音）
BGM_PATH    = "Escort.mp3"
# 幅（偶数推奨）
WIDTH       = 1920
# 高さ（偶数推奨）
HEIGHT      = 1080
# PDFレンダリングDPI
DPI         = 200
# 1ページの表示秒数
DURATION    = 5.0
# クロスフェード秒（0で無効、DURATIONより短く）
CROSSFADE   = 1.0
# フレームレート
FPS         = 30
# BGM音量（0.0〜1.0）
BGM_VOLUME  = 0.6
# 余白色（黒）
BGCOLOR     = "#000000"
# "auto" | "pymupdf" | "pdf2image"
BACKEND     = "auto"
# -----------------------------------------------------------------

# moviepy.editor を使わずに必要部分のみ import（matplotlib回避）
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import moviepy.video.fx.all as vfx
import moviepy.audio.fx.all as afx

# poppler/binを環境変数PATHに追加する
poppler_dir = Path(__file__).parent.absolute() / "poppler/Library/bin"
os.environ["PATH"] += os.pathsep + str(poppler_dir)


def render_pdf_pages(pdf_path: str, dpi: int = 200, backend: str = "auto") -> List[Image.Image]:
    """PDF を各ページごとに PIL.Image にレンダリングして返す。backend: "auto"|"pymupdf"|"pdf2image" """
    images: List[Image.Image] = []

    def _render_with_pymupdf() -> List[Image.Image]:
        import fitz  # PyMuPDF
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        out = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                out.append(img)
        return out

    def _render_with_pdf2image() -> List[Image.Image]:
        from pdf2image import convert_from_path
        return [p.convert("RGB") for p in convert_from_path(pdf_path, dpi=dpi)]

    tried = []
    if backend in ("auto", "pymupdf"):
        try:
            images = _render_with_pymupdf()
            return images
        except Exception as e:
            tried.append(f"pymupdf: {e}")

    if backend in ("auto", "pdf2image"):
        try:
            images = _render_with_pdf2image()
            return images
        except Exception as e:
            tried.append(f"pdf2image: {e}")

    msg = (
        "PDF のレンダリングに失敗しました。\n"
        f"  - 試行結果: {tried}\n"
        "対処:\n"
        "  1) `pip install PyMuPDF` を実行してから BACKEND='pymupdf' にする\n"
        "  2) または `pip install pdf2image` と Poppler を導入して BACKEND='pdf2image'\n"
    )
    raise RuntimeError(msg)


def pad_to_resolution(img: Image.Image, target_size: Tuple[int, int], bg_color=(0, 0, 0)) -> Image.Image:
    """アスペクト比を維持してリサイズし、左右または上下に余白を付与。"""
    img = img.convert("RGB")
    W, H = target_size
    iw, ih = img.size
    img_ratio = iw / ih
    tgt_ratio = W / H

    if img_ratio > tgt_ratio:
        new_w = W
        new_h = int(new_w / img_ratio)
    else:
        new_h = H
        new_w = int(new_h * img_ratio)

    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), color=bg_color)
    ox = (W - new_w) // 2
    oy = (H - new_h) // 2
    canvas.paste(resized, (ox, oy))
    return canvas


def crossfade_compose(clips: List[ImageClip], overlap: float, size: Tuple[int, int]) -> CompositeVideoClip:
    """
    crossfadein を使わずにクロスフェードを実現する。
    - 各クリップの末尾を fadeout(overlap)
    - 次クリップの先頭を fadein(overlap)
    - それぞれの start を (i * (dur - overlap)) にずらして重ねる
    """
    if overlap <= 0 or len(clips) == 1:
        # フォールバック：単純結合
        return concatenate_videoclips(clips, method="compose")

    DUR = float(clips[0].duration)
    OVER = float(overlap)
    if OVER >= DUR:
        raise ValueError("CROSSFADE は DURATION より短くしてください。")

    layered = []
    for i, c in enumerate(clips):
        cc = c
        if i > 0:
            cc = cc.fx(vfx.fadein, OVER)
        if i < len(clips) - 1:
            cc = cc.fx(vfx.fadeout, OVER)
        start_t = i * (DUR - OVER)
        cc = cc.set_start(start_t)
        layered.append(cc)

    total_duration = (len(clips) - 1) * (DUR - OVER) + DUR
    return CompositeVideoClip(layered, size=size).set_duration(total_duration)


def main():
    # 入力PDFチェック
    if not os.path.isfile(INPUT_PDF):
        print(f"[ERROR] PDF が見つかりません: {INPUT_PDF}")
        sys.exit(1)

    # 背景色
    try:
        bg_color = ImageColor.getrgb(BGCOLOR)
    except Exception:
        print(f"[WARN] 背景色 {BGCOLOR} を解釈できません。黒(#000000)を使用します。")
        bg_color = (0, 0, 0)

    # 偶数へ丸め込み
    W = WIDTH - (WIDTH % 2)
    H = HEIGHT - (HEIGHT % 2)
    size = (W, H)

    print("[INFO] PDF をレンダリング中 ...")
    pages = render_pdf_pages(INPUT_PDF, dpi=DPI, backend=BACKEND)
    if not pages:
        print("[ERROR] PDFにページがありません。")
        sys.exit(1)
    print(f"[INFO] ページ数: {len(pages)}")

    print("[INFO] 解像度にパディング中 ...")
    slides = [pad_to_resolution(p, size, bg_color) for p in pages]

    print("[INFO] クリップを生成中 ...")
    clips: List[ImageClip] = []
    for im in slides:
        frame = np.array(im)  # RGB
        clip = ImageClip(frame).set_duration(float(DURATION))
        # 必要ならスライド個別にフェード（任意・好みで有効化）
        # fade_len = min(1.0, DURATION / 4)
        # clip = clip.fx(vfx.fadein, fade_len).fx(vfx.fadeout, fade_len)
        clips.append(clip)

    # 映像の合成
    if len(clips) == 1 or float(CROSSFADE) <= 0:
        print("[INFO] クロスフェードなしで結合します。")
        video = concatenate_videoclips(clips, method="compose")
    else:
        print(f"[INFO] クロスフェード {CROSSFADE:.2f} 秒で合成します。")
        video = crossfade_compose(clips, overlap=float(CROSSFADE), size=size)

    # BGM（任意）
    audio_flag = False
    audio_codec = None
    if BGM_PATH and os.path.isfile(BGM_PATH):
        print(f"[INFO] BGM を合成します: {BGM_PATH}")
        bgm = AudioFileClip(BGM_PATH)
        if bgm.duration < video.duration:
            bgm = afx.audio_loop(bgm, duration=video.duration)
        else:
            bgm = bgm.subclip(0, video.duration)
        vol = max(0.0, min(1.0, float(BGM_VOLUME)))
        bgm = bgm.fx(afx.volumex, vol)
        # 仕上げに軽くフェードイン/アウト（任意）
        # bgm = bgm.fx(afx.audio_fadein, 0.5).fx(afx.audio_fadeout, 0.5)
        video = video.set_audio(bgm)
        audio_flag = True
        audio_codec = "aac"
    else:
        print("[INFO] BGM なし（無音で出力します）")

    # 出力
    out_path = OUTPUT_MP4
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"[INFO] 書き出し中 ... → {out_path}")
    video.write_videofile(
        out_path,
        fps=int(FPS),
        codec="libx264",
        audio=audio_flag,
        audio_codec=audio_codec if audio_flag else None,
        preset="medium",
        threads=os.cpu_count() or 4,
        temp_audiofile="~temp-audio.m4a",
        remove_temp=True,
    )
    print("[INFO] 完了しました。")


if __name__ == "__main__":
    main()

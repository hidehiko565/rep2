
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
OUTPUT_MP4  = "20251008_2_1.mp4"
# BGMのファイルと音量（0.0〜1.0）
BGM_PATH    = "Escort.mp3"
BGM_VOLUME  = 0.3
# 事前に作った読み上げ音声と音量（0.0〜1.0）
VOICE_PATH   = "voice.wav"
VOICE_VOLUME = 1.0
# 幅（偶数推奨）
WIDTH       = 1920
# 高さ（偶数推奨）
HEIGHT      = 1080
# PDFレンダリングDPI
DPI         = 200

# 最初のページとそれ以外で秒数を分ける
# 最初のページの表示秒数
FIRST_DURATION  =  5
# 2ページ目以降の表示秒数
OTHER_DURATION  = 10

# クロスフェード秒（0で無効、DURATIONより短く）
CROSSFADE   = 1.0
# フレームレート
FPS         = 30
# 余白色（黒）
BGCOLOR     = "#000000"
# "auto" | "pymupdf" | "pdf2image"
BACKEND     = "auto"
# -----------------------------------------------------------------

# moviepy.editor を使わずに必要部分のみ import（matplotlib回避）
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.audio.AudioClip import CompositeAudioClip
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

def crossfade_compose_var(
    clips: List[ImageClip],
    durations: List[float],
    base_overlap: float,
    size: Tuple[int, int],
) -> CompositeVideoClip:
    """
    可変長クリップ用クロスフェード合成。
    - クリップ i の開始時刻は sum_{k < i} (durations[k] - ov_{k,k+1})
    - ov_{i-1,i} = min(base_overlap, durations[i-1], durations[i]) で安全に制限
    - フェードは各ペアの ov を用いて fadeout/fadein を適用
    """
    n = len(clips)
    if n == 0:
        raise ValueError("clips が空です。")
    if n != len(durations):
        raise ValueError("clips と durations の長さが一致しません。")

    # オーバーラップ秒（ペアごとに算出）
    pair_overlap = [0.0] * (n - 1)
    if base_overlap > 0:
        for i in range(1, n):
            pair_overlap[i - 1] = max(
                0.0, min(base_overlap, float(durations[i - 1]), float(durations[i]))
            )

    # 時間配置：start[i] = Σ_{k < i} (durations[k] - ov_{k,k+1})
    starts = [0.0] * n
    for i in range(1, n):
        starts[i] = starts[i - 1] + float(durations[i - 1]) - pair_overlap[i - 1]

    # フェード適用＆合成レイヤー作成
    layers = []
    for i, c in enumerate(clips):
        cc = c.set_duration(float(durations[i]))
        # 前との重なり（フェードイン）
        if i > 0 and pair_overlap[i - 1] > 0:
            cc = cc.fx(vfx.fadein, pair_overlap[i - 1])
        # 次との重なり（フェードアウト）
        if i < n - 1 and pair_overlap[i] > 0:
            cc = cc.fx(vfx.fadeout, pair_overlap[i])
        cc = cc.set_start(starts[i])
        layers.append(cc)

    total_duration = starts[-1] + float(durations[-1])
    return CompositeVideoClip(layers, size=size).set_duration(total_duration)

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


    # --- 可変秒数リストを作成 ---
    # 例：12ページあるPDFに対してページごと秒数を指定
    #per_page = [6, 4, 4, 8, 3, 4, 4, 5, 5, 3, 4, 4]
    #durations = per_page[:len(pages)]
    durations: List[float] = []
    for idx in range(len(pages)):
        if idx == 0:
            durations.append(float(FIRST_DURATION))
        else:
            durations.append(float(OTHER_DURATION))

    print("[INFO] クリップを生成中 ...")
    clips: List[ImageClip] = []
    for im, dur in zip(slides, durations):
        frame = np.array(im)  # RGB
        clip = ImageClip(frame).set_duration(dur)
        # スライド個別フェード（任意でONにするなら以下2行のコメントを外す）
        # fade_len = min(1.0, dur / 4)
        # clip = clip.fx(vfx.fadein, fade_len).fx(vfx.fadeout, fade_len)
        clips.append(clip)

    # --- 映像の合成 ---
    if len(clips) == 1 or float(CROSSFADE) <= 0:
        print("[INFO] クロスフェードなしで結合します。")
        video = concatenate_videoclips(clips, method="compose")
    else:
        print(f"[INFO] クロスフェード {CROSSFADE:.2f} 秒（ペアごとに安全値へ丸め）で合成します。")
        video = crossfade_compose_var(
            clips=clips,
            durations=durations,
            base_overlap=float(CROSSFADE),
            size=size,
        )

    # 出力
    out_path = OUTPUT_MP4
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"[INFO] 書き出し中 ... → {out_path}")


    # ---- ここからクリーンアップを保証するための try/finally ----
    raw_bgm = None       # BGM の元クリップ
    bgm_clip = None      # ループ/トリム/音量適用済み
    raw_voice = None     # VOICE の元クリップ
    voice_clip = None    # トリム/音量適用済み
    audio_mix = None     # 合成後の CompositeAudioClip

    try:
        audio_flag = False
        audio_codec = None

        # --- BGM セット（存在する場合） ---
        if BGM_PATH and os.path.isfile(BGM_PATH):
            print(f"[INFO] BGM を読み込みます: {BGM_PATH}")
            raw_bgm = AudioFileClip(BGM_PATH)
            # 長さ合わせ（ループ or トリム）
            if raw_bgm.duration < video.duration:
                bgm_clip = afx.audio_loop(raw_bgm, duration=video.duration)
            else:
                bgm_clip = raw_bgm.subclip(0, video.duration)
            # 音量（0〜1にクリップ）
            vol_bgm = max(0.0, min(1.0, float(BGM_VOLUME)))
            bgm_clip = bgm_clip.fx(afx.volumex, vol_bgm)

        # --- VOICE セット（存在する場合） ---
        if VOICE_PATH and os.path.isfile(VOICE_PATH):
            print(f"[INFO] VOICE を読み込みます: {VOICE_PATH}")
            raw_voice  = AudioFileClip(VOICE_PATH)
            # VOICE はループせず、動画尺でトリム
            if raw_voice.duration > video.duration:
                voice_clip = raw_voice.subclip(0, video.duration)
            else:
                voice_clip = raw_voice  # 短ければそのまま（無音部分はそのまま）
            # 音量（0〜1にクリップ）
            vol_voice = max(0.0, min(1.0, float(VOICE_VOLUME)))
            voice_clip = voice_clip.fx(afx.volumex, vol_voice)

        # --- 合成ロジック ---
        if bgm_clip is not None and voice_clip is not None:
            print("[INFO] BGM と VOICE をミックスして合成します。")
            audio_mix = CompositeAudioClip([bgm_clip, voice_clip]).set_duration(video.duration)
            video = video.set_audio(audio_mix)
            audio_flag = True
            audio_codec = "aac"

        elif voice_clip is not None:
            print("[INFO] VOICE のみを音声として使用します。")
            # 念のため動画尺で切り上げ
            video = video.set_audio(voice_clip.set_duration(video.duration))
            audio_flag = True
            audio_codec = "aac"

        elif bgm_clip is not None:
            print("[INFO] BGM のみを音声として使用します。")
            video = video.set_audio(bgm_clip.set_duration(video.duration))
            audio_flag = True
            audio_codec = "aac"

        else:
            print("[INFO] BGM/VOICE なし（無音で出力します）")
            audio_flag = False
            audio_codec = None

        # --- 書き出し ---
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

    finally:
        # クローズ順は「合成 → 子クリップ」を推奨
        for obj in (audio_mix, bgm_clip, raw_bgm, voice_clip, raw_voice):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        try:
            if video is not None:
                video.close()
        except Exception:
            pass
        for c in clips:
            try:
                c.close()
            except Exception:
                pass
        try:
            import gc, time
            del audio_mix, bgm_clip, raw_bgm, voice_clip, raw_voice, video, clips
            gc.collect()
            time.sleep(0.1)
        except Exception:
            pass
    # ---- ここまで ----

if __name__ == "__main__":
    main()

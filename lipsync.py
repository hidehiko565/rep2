# -*- coding: utf-8 -*-
"""
voice1.wav ～ voiceN.wav を順番に処理して、フルフェイス差分（N/A/I/U/E/O）を切替えた
複数のクリップを連結し、互換性の高い H.264/AAC の MP4 を出力します。
（画像は縦・横ともに偶数にしておいた方が安定する）
"""

import os
import glob
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image

import librosa
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.audio.io.AudioFileClip import AudioFileClip
import moviepy.video.fx.all as vfx


# ========================
# 設定
# ========================
ASSETS = "assets"

# ニュートラル（閉口）追加
NEUTRAL_FILE = os.path.join(ASSETS, "boy0_n.png")

# フェイス画像（ファイル名は必要に応じて変更）
VOWEL2FILE: Dict[str, str] = {
    "a": os.path.join(ASSETS, "boy1_a.png"),
    "i": os.path.join(ASSETS, "boy2_i.png"),
    "u": os.path.join(ASSETS, "boy3_u.png"),
    "e": os.path.join(ASSETS, "boy4_e.png"),
    "o": os.path.join(ASSETS, "boy5_o.png"),
}

# 解析/出力パラメータ
OUT = "lipsync_out_faces_all.mp4"
FPS = 30
SR_TARGET = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 40

# スムージング・無音処理
MIN_SPAN_SEC = 0.05
SILENCE_RMS_THRESH = 0.012
SILENCE_HOLD = True  # True の場合、無音時は n を使う（今回の仕様に合わせて常に n を採用）

# H.264 出力設定
VIDEO_BITRATE = "6000k"
AUDIO_BITRATE = "192k"
H264_PRESET = "slow"


# ========================
# 画像ロード & 偶数サイズ化
# ========================
def load_face_images_even(vowel2file: Dict[str, str], neutral_path: str) -> Tuple[Dict[str, np.ndarray], Tuple[int, int]]:
    # キー順に 'n' を先頭に追加
    keys = ["n", "a", "i", "u", "e", "o"]
    path_map = {"n": neutral_path, **vowel2file}

    # 参照サイズを n（neutral）から決定（なければ a でエラーになるため n を必須に）
    first_path = path_map["n"]
    if not os.path.exists(first_path):
        raise FileNotFoundError(f"ニュートラル画像が見つかりません: {first_path}")

    base = Image.open(first_path).convert("RGB")
    w0, h0 = base.size
    # 偶数化（H.264/yuv420p 互換）
    even_w = w0 if w0 % 2 == 0 else w0 - 1
    even_h = h0 if h0 % 2 == 0 else h0 - 1

    imgs = {}
    for k in keys:
        p = path_map[k]
        if not os.path.exists(p):
            raise FileNotFoundError(f"画像が見つかりません: {p}")
        im = Image.open(p).convert("RGB")
        if im.size != (even_w, even_h):
            im = im.resize((even_w, even_h), Image.LANCZOS)
        imgs[k] = np.array(im)
    return imgs, (even_w, even_h)


# ========================
# 母音スコア（簡易ヒューリスティック）
# ========================
def make_band_masks(sr: int, n_mels: int):
    mels_hz = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr / 2)

    def band(low, high):
        return (mels_hz >= low) & (mels_hz < high)

    mask_low   = band(   80,  500)
    mask_mlow  = band(  500, 1200)
    mask_mid   = band( 1200, 2400)
    mask_high  = band( 2400, 4000)
    mask_uhigh = band( 4000, 8000)
    return mask_low, mask_mlow, mask_mid, mask_high, mask_uhigh


def vowel_scores_from_mel(mel_frame_db: np.ndarray, masks):
    mask_low, mask_mlow, mask_mid, mask_high, mask_uhigh = masks

    def mean_or(m):
        return mel_frame_db[m].mean() if np.any(m) else -80.0

    e_low   = mean_or(mask_low)
    e_mlow  = mean_or(mask_mlow)
    e_mid   = mean_or(mask_mid)
    e_high  = mean_or(mask_high)
    e_uhigh = mean_or(mask_uhigh)

    score_a = 1.2 * e_low + 0.6 * e_mlow + 0.2 * e_mid - 0.1 * e_high
    score_o = 1.0 * e_low + 0.8 * e_mlow + 0.3 * e_mid - 0.2 * e_high
    score_e = 0.3 * e_low + 0.8 * e_mid + 0.3 * e_high
    score_i = 0.2 * e_low + 0.5 * e_mid + 1.0 * e_high + 0.4 * e_uhigh
    score_u = 0.1 * e_low + 0.3 * e_mid + 0.9 * e_high + 0.7 * e_uhigh

    return {"a": score_a, "i": score_i, "u": score_u, "e": score_e, "o": score_o}


def smooth_labels_minspan(labels: List[str], min_span_frames: int) -> List[str]:
    if min_span_frames <= 1:
        return labels
    L = labels[:]
    i = 0
    while i < len(L):
        j = i + 1
        while j < len(L) and L[j] == L[i]:
            j += 1
        span = j - i
        if span < min_span_frames:
            if i > 0 and j < len(L):
                L[i:j] = [L[i - 1]] * span
            elif i == 0 and j < len(L):
                L[i:j] = [L[j]] * span
            elif i > 0 and j == len(L):
                L[i:j] = [L[i - 1]] * span
        i = j
    return L


# ========================
# 1つの音声→短い動画クリップを作る
# ========================
def build_clip_for_audio(audio_path: str,
                         imgs: Dict[str, np.ndarray],
                         size_wh: Tuple[int, int]) -> ImageSequenceClip:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

    y, sr = librosa.load(audio_path, sr=SR_TARGET, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel + 1e-10)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP_LENGTH)[0]

    n_video_frames = int(np.ceil(duration * FPS))
    t_idx = np.linspace(0, mel_db.shape[1] - 1, n_video_frames).astype(int)
    mel_for_video = mel_db[:, t_idx]
    rms_for_video = rms[np.minimum(t_idx, len(rms) - 1)]

    masks = make_band_masks(sr, N_MELS)

    # 初期はニュートラル 'n'
    labels: List[str] = ["n"] * n_video_frames

    for t in range(n_video_frames):
        # 無音時は n を維持
        if rms_for_video[t] < SILENCE_RMS_THRESH:
            labels[t] = "n"
        else:
            scores = vowel_scores_from_mel(mel_for_video[:, t], masks)
            labels[t] = max(scores, key=scores.get)

    # スムージング
    min_span_frames = max(1, int(MIN_SPAN_SEC * FPS))
    labels = smooth_labels_minspan(labels, min_span_frames)

    # 画像列を作成（すでに偶数サイズ）
    frames = [imgs[v] for v in labels]  # v ∈ {'n','a','i','u','e','o'}

    # 映像クリップ（無音）
    vclip = ImageSequenceClip(frames, fps=FPS)
    # 音声クリップ（元 WAV）
    aclip = AudioFileClip(audio_path)

    # 合成（※ set_duration で映像＝音声に合わせる）
    vclip = vclip.set_audio(aclip).set_duration(aclip.duration)

    # 念のためサイズ確認（偶数のはずだが念押し）
    w, h = vclip.size
    if (w % 2 != 0) or (h % 2 != 0):
        vclip = vclip.fx(vfx.resize, newsize=(size_wh[0], size_wh[1]))

    return vclip


# ========================
# メイン：voice*.wav を連結
# ========================
def main():
    # 1) フェイス画像読み込み（偶数サイズに揃える）
    imgs, even_size = load_face_images_even(VOWEL2FILE, NEUTRAL_FILE)

    # 2) assets/voice*.wav を昇順に列挙
    audio_files = sorted(glob.glob(os.path.join(ASSETS, "voice*.wav")),
                         key=lambda p: (len(p), p))
    if not audio_files:
        raise FileNotFoundError("assets 内に voice*.wav が見つかりません。")

    # 3) 各音声→短いクリップを生成
    parts = []
    for ap in audio_files:
        print(f"[INFO] build clip from: {os.path.basename(ap)}")
        clip = build_clip_for_audio(ap, imgs, even_size)
        parts.append(clip)

    # 4) 連結
    full = concatenate_videoclips(parts, method="compose")

    # 5) 互換性の高い設定で最終書き出し
    full.write_videofile(
        OUT,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate=AUDIO_BITRATE,
        fps=FPS,
        preset=H264_PRESET,
        bitrate=VIDEO_BITRATE,
        ffmpeg_params=[
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-level", "4.1",
            "-movflags", "+faststart",
        ],
    )

    print(f"\nDone. Output: {OUT}")


if __name__ == "__main__":
    main()

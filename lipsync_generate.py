# -*- coding: utf-8 -*-
"""
ゆっくり風 口パク連番PNG生成テンプレート
Desc :
- 音声レベル（短時間エネルギー）に応じて A/B/C の口画像を切り替え
- フレームレート固定で、透過PNG連番を出力
- スムージング（アタック/リリース）、無音クローズ、しきい値自動調整あり

依存:
  pip install numpy soundfile pillow librosa
  （librosa はしきい値自動化でメルスペクトル統計などに使用、不要なら外せます）

入力:
  input/audio.wav    ... 16kHz/mono の WAV 推奨（他でもOK）
  input/base.png     ... キャラクターベース（口なし）
  mouth/mouth_A.png  ... 閉じ
  mouth/mouth_B.png  ... 半開き
  mouth/mouth_C.png  ... 全開

出力:
  out/frames/frame_000001.png など

"""

"""
フロー
1. 元動画から音声のみを抜き取る
ffmpeg -i input/video.mp4 -vn -ac 1 -ar 16000 -sample_fmt s16 input/audio.wav

2. 音声 → 口パク用のイラスト生成（Python）
python lipsync_generate.py

3. 口パクイラストを元動画に重ねる
ffmpeg -i input/video.mp4 -framerate 30 -i out/frames/frame_%06d.png -filter_complex "[1:v]scale=150:-1[scaled]; [0:v][scaled]overlay=20:main_h-h-20:format=auto" -map 0:a? -c:v libx264 -pix_fmt yuv420p -profile:v high -level 4.0 -crf 18 -preset veryfast -c:a aac -b:a 192k -movflags +faststart -shortest output.mp4


"""




import os
import math
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import soundfile as sf
from PIL import Image
import librosa


# =========================
# 設定
# =========================
@dataclass
class Config:
    audio_path: str = "input/audio.wav"
    base_image_path: str = "input/base.png"
    mouth_A_path: str = "mouth/mouth_A.png"  # 閉
    mouth_B_path: str = "mouth/mouth_B.png"  # 半
    mouth_C_path: str = "mouth/mouth_C.png"  # 開
    out_dir: str = "out/frames"

    # 出力フレーム
    fps: int = 30

    # 口の貼り付け位置（ベース画像の座標系, 左上原点）
    mouth_anchor_xy: Tuple[int, int] = (248, 380)  # (x, y)

    # 音声前処理
    target_sr: int = 16000        # 処理サンプリングレート
    normalize: bool = True        # 正規化
    highpass_hz: float = 50.0     # 50Hz HPF（0 で無効）
    frame_hop_ms: float = 1000.0 / 30.0  # 1フレームの解析窓（≈ 33.33ms）
    energy_smooth_ms: float = 50.0       # エネルギー平滑
    voice_floor_db: float = -60.0        # 無音とみなす下限（dBFS相当の目安）

    # しきい値
    threshold_mode: str = "auto"  # "auto" or "manual"
    manual_thresholds: Tuple[float, float] = (0.2, 0.45)  # (A->B, B->C) in [0..1]

    # ヒステリシス・アタック/リリース（目パチのチラツキ抑制）
    attack_frames: int = 1     # 開く反応を遅らせ（>=1）
    release_frames: int = 2    # 閉じる反応を早める/遅らせる（>=1）
    hold_silence_close: int = 3  # 無音が連続したら強制Aに戻すフレーム数

    # 出力ファイル名
    filename_format: str = "frame_{:06d}.png"

cfg = Config()


# =========================
# ユーティリティ
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_wav_mono(path: str, target_sr: int, normalize=True) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(path)
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    y = y.astype(np.float32)
    if normalize:
        peak = np.max(np.abs(y)) + 1e-9
        y = y / peak
    return y, sr


def highpass_filter(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return y
    # シンプルな1次HPF (RC filter)
    rc = 1.0 / (2 * math.pi * cutoff_hz)
    dt = 1.0 / sr
    alpha = rc / (rc + dt)
    out = np.zeros_like(y, dtype=np.float32)
    prev_out = 0.0
    prev_x = 0.0
    for i, x in enumerate(y):
        out[i] = alpha * (prev_out + x - prev_x)
        prev_out = out[i]
        prev_x = x
    return out


def frame_energy(y: np.ndarray, sr: int, hop_ms: float, smooth_ms: float) -> np.ndarray:
    """
    短時間エネルギー（RMS）をフレーム化 & 平滑化
    返却: [0..1] 正規化されたエネルギー系列
    """
    hop = int(sr * hop_ms / 1000.0)
    win = hop  # 単純化 (フレーム幅 = hop)
    n = len(y)
    # RMS per frame
    energies = []
    for start in range(0, n, hop):
        end = min(start + win, n)
        frame = y[start:end]
        if len(frame) == 0:
            e = 0.0
        else:
            e = float(np.sqrt(np.mean(frame**2)))
        energies.append(e)
    energies = np.array(energies, dtype=np.float32)

    # ノイズフロア調整（-60dB以下はゼロ近似）
    eps = 1e-8
    energies_db = 20.0 * np.log10(np.maximum(energies, eps))
    energies_db = np.maximum(energies_db, cfg.voice_floor_db)
    # 0..1へ線形マップ
    energies_norm = (energies_db - cfg.voice_floor_db) / (0.0 - cfg.voice_floor_db)  # -60dB→0, 0dB→1
    energies_norm = np.clip(energies_norm, 0.0, 1.0)

    # 平滑移動平均
    if smooth_ms > 0:
        k = max(1, int((smooth_ms / hop_ms)))
        if k > 1:
            kernel = np.ones(k, dtype=np.float32) / k
            energies_norm = np.convolve(energies_norm, kernel, mode="same")

    return energies_norm


def auto_thresholds(energies: np.ndarray) -> Tuple[float, float]:
    """
    自動しきい値: エネルギーの分位点から2点を抽出
    例: A->B を 60パーセンタイル、B->C を 85パーセンタイル（経験値）
    """
    if len(energies) == 0:
        return (0.2, 0.5)
    t1 = float(np.quantile(energies, 0.60))
    t2 = float(np.quantile(energies, 0.85))
    t1 = max(0.05, min(t1, 0.95))
    t2 = max(t1 + 0.05, min(t2, 0.98))
    return (t1, t2)


def map_energy_to_mouth_states(energies: np.ndarray,
                               t1: float, t2: float,
                               attack: int, release: int,
                               silence_hold: int) -> List[str]:
    """
    energies in [0..1] を口A/B/Cに割当。ヒステリシスを簡易実装。
    戻り値: ["A"|"B"|"C"] * n_frames
    """
    states = []
    current = "A"
    open_counter = 0
    close_counter = 0
    silence_counter = 0

    for e in energies:
        # しきい値によるターゲット状態
        target = "A"
        if e >= t2:
            target = "C"
        elif e >= t1:
            target = "B"
        else:
            target = "A"

        # 無音ホールド（小さな値が続くならA優先）
        if e < 0.05:
            silence_counter += 1
        else:
            silence_counter = 0
        if silence_counter >= silence_hold:
            target = "A"

        # 簡易アタック/リリース
        if target == current:
            open_counter = 0
            close_counter = 0
        else:
            if target > current:
                # A->B->C の順で大小比較できるように並び順を考える
                # ここでは 'A'<'B'<'C' の辞書順に依存（安全に数値化して比較してもOK）
                open_counter += 1
                close_counter = 0
                if open_counter >= attack:
                    current = target
                    open_counter = 0
            else:
                close_counter += 1
                open_counter = 0
                if close_counter >= release:
                    current = target
                    close_counter = 0

        states.append(current)

    return states


def composite_frame(base: Image.Image,
                    mouthA: Image.Image,
                    mouthB: Image.Image,
                    mouthC: Image.Image,
                    state: str,
                    anchor_xy: Tuple[int, int]) -> Image.Image:
    """
    ベース画像に mouth_* を合成して1枚の透過PNGにする
    """
    canvas = base.copy()
    x, y = anchor_xy
    if state == "A":
        canvas.alpha_composite(mouthA, (x, y))
    elif state == "B":
        canvas.alpha_composite(mouthB, (x, y))
    else:
        canvas.alpha_composite(mouthC, (x, y))
    return canvas


def main():
    ensure_dir(cfg.out_dir)

    # 1) 音声読み込み
    y, sr = load_wav_mono(cfg.audio_path, cfg.target_sr, normalize=cfg.normalize)
    y = highpass_filter(y, sr, cfg.highpass_hz)

    # 2) エネルギー系列（フレーム毎）を算出
    energies = frame_energy(y, sr, hop_ms=cfg.frame_hop_ms, smooth_ms=cfg.energy_smooth_ms)

    # 3) しきい値決定
    if cfg.threshold_mode == "auto":
        t1, t2 = auto_thresholds(energies)
        print(f"[auto thresholds] A->B={t1:.3f}, B->C={t2:.3f}")
    else:
        t1, t2 = cfg.manual_thresholds
        print(f"[manual thresholds] A->B={t1:.3f}, B->C={t2:.3f}")

    # 4) 口状態列を決める
    states = map_energy_to_mouth_states(
        energies, t1, t2,
        attack=cfg.attack_frames,
        release=cfg.release_frames,
        silence_hold=cfg.hold_silence_close
    )

    # 5) 画像読み込み
    base = Image.open(cfg.base_image_path).convert("RGBA")
    mouthA = Image.open(cfg.mouth_A_path).convert("RGBA")
    mouthB = Image.open(cfg.mouth_B_path).convert("RGBA")
    mouthC = Image.open(cfg.mouth_C_path).convert("RGBA")

    # サイズが合わない場合は mouth を合わせる（必要なら）
    # mouthA = mouthA.resize((w, h), resample=Image.LANCZOS)
    # ...

    # 6) 連番出力
    for i, st in enumerate(states, start=1):
        frame = composite_frame(base, mouthA, mouthB, mouthC, st, cfg.mouth_anchor_xy)
        out_path = os.path.join(cfg.out_dir, cfg.filename_format.format(i))
        frame.save(out_path)
        if i % 100 == 0:
            print(f"generated: {out_path}")

    print(f"done. frames: {len(states)} -> {cfg.out_dir}")


if __name__ == "__main__":
    main()

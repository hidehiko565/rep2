# -*- coding: utf-8 -*-
"""
cut_silence_video.py
FFmpegのsilencedetectで無音区間を検出し、有音区間のみを連結して新しいMP4を作成します。
・すべてローカルで処理（クラウド不使用）
・実行時に追加のライブラリをダウンロードしません（Python標準 + ローカルFFmpeg/FFprobe）
"""

import subprocess
import sys
import shutil
import re
from pathlib import Path
from typing import List, Tuple

# ========= ユーザー設定 =========
# 入力動画
INPUT_PATH  = r"gakumon.mp4"
# 出力動画（別名）
OUTPUT_PATH = r"gakumon_b.mp4"

# 有音区間数の最大
# これを越えると concat フィルタが長大になり、処理・安定性の面で不利なのでガード
MAX_SEGMENTS      = 250

# 各種パラメータ。調整して有音区間数が MAX_SEGMENTS を越えないようにする。
# 無音閾値（dB）
NOISE_DB          = -30.0
# 無音最小長（秒）
MIN_SILENCE_SEC   = 0.60

# PRE_ROLL_SEC / POST_ROLL_SEC を増やす → 有音断片を自然につなげやすくなる
PRE_ROLL_SEC  = 0.10
POST_ROLL_SEC = 0.10
# MIN_KEEP_SEC を上げる → 短すぎる断片を捨てる
MIN_KEEP_SEC  = 0.40

VIDEO_CODEC       = "libx264"
CRF               = 18
PRESET            = "medium"
AUDIO_CODEC       = "aac"
AUDIO_BITRATE     = "192k"
# =============================

def check_ffmpeg_available():
    for bin_name in ("ffmpeg", "ffprobe"):
        if shutil.which(bin_name) is None:
            print(f"[ERROR] {bin_name} が見つかりません。PATH を確認してください。")
            sys.exit(1)

def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def get_duration_sec(input_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    res = run_cmd(cmd)
    if res.returncode != 0:
        print("[ERROR] ffprobe で動画長が取得できませんでした。", res.stderr)
        sys.exit(1)
    try:
        return float(res.stdout.strip())
    except ValueError:
        print("[ERROR] 動画の長さを数値に変換できませんでした。出力:", res.stdout)
        sys.exit(1)

def detect_silences(input_path: str, noise_db: float, min_silence: float) -> List[Tuple[float, float]]:
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", input_path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f", "null", "-"
    ]
    res = run_cmd(cmd)
    out = res.stderr
    re_start = re.compile(r"silence_start:\s*([0-9.]+)")
    re_end   = re.compile(r"silence_end:\s*([0-9.]+)")

    starts = [float(m.group(1)) for m in re_start.finditer(out)]
    ends   = [float(m.group(1)) for m in re_end.finditer(out)]

    silences = []
    i = j = 0
    while i < len(starts) and j < len(ends):
        if ends[j] > starts[i]:
            silences.append((starts[i], ends[j]))
            i += 1
            j += 1
        else:
            j += 1
    if i < len(starts):
        silences.append((starts[i], -1.0))
    return silences

def invert_to_keep_intervals(silences: List[Tuple[float, float]], total: float) -> List[Tuple[float, float]]:
    sil = []
    for s, e in silences:
        if e < 0:
            e = total
        sil.append((max(0.0, s), min(total, e)))

    sil.sort(key=lambda x: x[0])
    merged = []
    for s, e in sil:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    sil = [(s, e) for s, e in merged]

    keep = []
    cur = 0.0
    for s, e in sil:
        if s > cur:
            keep.append((cur, s))
        cur = max(cur, e)
    if cur < total:
        keep.append((cur, total))

    rolled = []
    for ks, ke in keep:
        ks = max(0.0, ks - PRE_ROLL_SEC)
        ke = min(total, ke + POST_ROLL_SEC)
        if ke - ks > 1e-6:
            rolled.append((ks, ke))

    if not rolled:
        return []
    rolled.sort(key=lambda x: x[0])
    merged_keep = [list(rolled[0])]
    for s, e in rolled[1:]:
        if s <= merged_keep[-1][1] + 1e-6:
            merged_keep[-1][1] = max(merged_keep[-1][1], e)
        else:
            merged_keep.append([s, e])

    pruned = [(s, e) for s, e in merged_keep if (e - s) >= MIN_KEEP_SEC]
    return pruned

def build_filter_complex(segments: List[Tuple[float, float]]) -> str:
    """
    concat用のfilter_complex文字列を生成（v,aをセグメントごとに交互に並べる）
    """
    parts = []
    pair_labels = []

    for i, (start, end) in enumerate(segments):
        s = f"{start:.6f}"
        e = f"{end:.6f}"

        vlabel = f"v{i}"
        alabel = f"a{i}"

        parts.append(f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[{vlabel}]")
        parts.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[{alabel}]")

        pair_labels.append(f"[{vlabel}]")
        pair_labels.append(f"[{alabel}]")

    n = len(segments)
    parts.append("".join(pair_labels) + f"concat=n={n}:v=1:a=1[v][a]")
    return ";".join(parts)

def main():
    check_ffmpeg_available()

    in_path = Path(INPUT_PATH)
    out_path = Path(OUTPUT_PATH)

    if not in_path.exists():
        print(f"[ERROR] 入力ファイルが見つかりません: {in_path}")
        sys.exit(1)

    print("[INFO] 動画長を取得しています...")
    duration = get_duration_sec(str(in_path))
    print(f"[INFO] 動画長: {duration:.3f} sec")

    print("[INFO] 無音区間を検出しています（silencedetect）...")
    silences = detect_silences(str(in_path), NOISE_DB, MIN_SILENCE_SEC)
    print(f"[INFO] 検出された無音イベント件数: {len(silences)}")

    print("[INFO] 有音区間（保持区間）に変換しています...")
    keep_segments = invert_to_keep_intervals(silences, duration)
    print(f"[INFO] 生成された有音区間の数: {len(keep_segments)}")

    if not keep_segments:
        print("[WARN] 有音区間が見つかりませんでした。入力が全無音の可能性があります。処理を中止します。")
        sys.exit(0)

    if len(keep_segments) > MAX_SEGMENTS:
        print(f"[WARN] 区間数が多すぎます（{len(keep_segments)} > {MAX_SEGMENTS}）。"
              f"パラメータ（MIN_SILENCE_SEC, MIN_KEEP_SEC, PRE/POST_ROLL_SEC）を見直してください。")
        sys.exit(1)

    print("[INFO] filter_complex を構築しています...")
    filter_complex = build_filter_complex(keep_segments)

    cmd = [
        "ffmpeg", "-y", "-hide_banner",
        "-i", str(in_path),
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        "-c:v", VIDEO_CODEC, "-crf", str(CRF), "-preset", PRESET,
        "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE,
        "-movflags", "+faststart",
        str(out_path)
    ]

    print("[INFO] FFmpeg で有音のみを連結した動画を生成しています...")
    res = run_cmd(cmd)
    if res.returncode != 0:
        print("[ERROR] FFmpeg の実行に失敗しました。")
        print("----- stderr -----")
        print(res.stderr)
        sys.exit(1)

    print(f"[DONE] 出力完了: {out_path.resolve()}")

if __name__ == "__main__":
    main()
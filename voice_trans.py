import time
from datetime import datetime
from faster_whisper import WhisperModel

# ====== 設定項目（環境に合わせて変更） ======
# 事前配布したモデルフォルダのローカルパスを指定
MODEL_DIR = r"C:\Users\01012119\Documents\bin\models\faster-whisper-large-v3"  # インストール済み
# 文字起こし対象の音声ファイルパス
AUDIO_FILE = "voice2.mp3"

# 実行デバイスと精度（GPUを使うなら下記を有効化）
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# 文字起こしオプション
BEAM_SIZE = 5
USE_VAD = True           # 無音区間の除外（推奨）
WITHOUT_TIMESTAMPS = False  # タイムスタンプ不要なら True
WORD_TIMESTAMPS = False     # 単語単位のタイムスタンプが必要なら True


def main():
    # ---------- 全体計測開始 ----------
    t_all_start = time.time()
    ts_all_start = datetime.now()

    print("=== faster-whisper 実行開始 ===")
    print(f"開始時刻: {ts_all_start.strftime('%Y-%m-%d %H:%M:%S')}")

    # ---------- モデルロード時間を計測 ----------
    t_load_start = time.time()
    print(f"\n[LOAD] モデル読み込み: {MODEL_DIR}")
    model = WhisperModel(MODEL_DIR, device=DEVICE, compute_type=COMPUTE_TYPE)
    t_load_end = time.time()
    load_sec = t_load_end - t_load_start
    print(f"[LOAD] 完了（{load_sec:.2f} 秒）")

    # ---------- 文字起こし時間を計測 ----------
    t_trans_start = time.time()
    print(f"\n[TRANSCRIBE] 開始: ファイル={AUDIO_FILE}")
    segments, info = model.transcribe(
        AUDIO_FILE,
        beam_size=BEAM_SIZE,
        vad_filter=USE_VAD,
        without_timestamps=WITHOUT_TIMESTAMPS,
        word_timestamps=WORD_TIMESTAMPS,
        # 必要に応じて language="ja", multilingual=True, initial_prompt="..." などを追加
    )
    t_trans_end = time.time()
    trans_sec = t_trans_end - t_trans_start
    print(f"[TRANSCRIBE] 完了（{trans_sec:.2f} 秒）")

    # ---------- 結果の表示（セグメント） ----------
    print("\n=== 文字起こし結果 ===")
    print(f"Detected language: {info.language} (prob={info.language_probability:.3f})")
    # segments はジェネレータなので、反復しながら出力
    seg_count = 0
    for seg in segments:
        seg_count += 1
        if WORD_TIMESTAMPS and seg.words:
            # 単語タイムスタンプ付きの詳細表示（必要時）
            words_text = " ".join([w.word for w in seg.words])
            print(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")
            print(f"  words: {words_text}")
        else:
            print(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")

    print(f"\nセグメント数: {seg_count}")

    # ---------- 総時間の表示 ----------
    t_all_end = time.time()
    ts_all_end = datetime.now()
    all_sec = t_all_end - t_all_start

    print("\n=== 実行時間サマリ ===")
    print(f"終了時刻: {ts_all_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"モデルロード時間:   {load_sec:.2f} 秒")
    print(f"文字起こし時間:     {trans_sec:.2f} 秒")
    print(f"総処理時間（合計）: {all_sec:.2f} 秒")
    print("=== 完了 ===")


if __name__ == "__main__":
    main()

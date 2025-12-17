from faster_whisper import WhisperModel

# テキストに変換する対象の音声ファイル
audio_file = 'voice.mp3'

# 音声認識AIモデルの選択
model_size = 'large-v3'
model = WhisperModel(model_size, device='cpu', compute_type='auto')

# 音声ファイルの解析・変換
segments, info = model.transcribe(audio_file, deam_size=5)

# テキストに変換した結果を表示
for data in segmants:
    print(f"[{data.start:.2f} - {data.end:.2f}]: {data.text}")
    
import json
import requests
import wave

# 基本情報
base_hash = {
    'host' : 'localhost',
    'port' : 50021,
    'in_txt' : "./input.txt",
    'out_wav' : "./audio.wav",
}
# パラメータ
params_dict = {
    'text' : "テストテスト",
    'speaker': 13
}
# 追加のパラメータ
data_dict = {
    'speedScale': 1.2
}

#----------

# wavファイルを出力
def generate_out_wav():
    print("generate_out_wav():\n")
    host = base_hash['host']
    port = base_hash['port']

    # audio_queryを作る
    response1 = requests.post(
        f'http://{host}:{port}/audio_query',
        params=params_dict
    )
    headers = {'Content-Type': 'application/json',}

    # 追加のパラメータはここでセット
    my_audio_query = response1.json()
    for key, value in data_dict.items():
        my_audio_query[key] = value

    # synthesis
    response2 = requests.post(
        f'http://{host}:{port}/synthesis',
        headers=headers,
        params=params_dict,
        data=json.dumps(my_audio_query)
    )

    # ファイルに保存
    wf = wave.open(base_hash['out_wav'], 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(response2.content)
    wf.close()

# txtファイルを読み込む
def read_in_txt():
    print("read_in_txt():\n")
    with open(base_hash['in_txt'], 'r', encoding='utf-8') as f:
        str = f.read()
        params_dict['text'] = str
    print("str\n")

# main()
if __name__ == '__main__':
    read_in_txt()
    generate_out_wav()

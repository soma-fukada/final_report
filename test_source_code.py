import torch
from source_code import Encoder, Decoder, LitAutoEncoder

def test_model_output_shape():
    """モデルの入出力サイズが正しいか検証するテスト"""
    
    # モデルとダミーデータを作る
    encoder = Encoder()
    decoder = Decoder()
    model = LitAutoEncoder(encoder, decoder)
    
    # ダミー入力(バッチサイズ1, 28x28=784次元のランダムなデータ)
    input_data = torch.randn(1, 28 * 28)

    # 実行
    z = encoder(input_data)
    output_data = decoder(z)

    # 検証
    #  (1, 3) の形
    assert z.shape == (1, 3)
    # 最終出力は入力と同じ (1, 784) の形だったら成功
    assert output_data.shape == input_data.shape
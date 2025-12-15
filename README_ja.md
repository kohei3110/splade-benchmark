# SPLADE ベンチマーク (Azure NC24ads_A100_v4)

Azure NC A100 上で `bizreach-inc/light-splade-japanese-14M` を計測するカスタムベンチマークです。バッチサイズとシーケンス長を変えながら、レイテンシ（P50/P95/P99）、スループット（QPS）、スパース性、GPU メトリクスを取得します。すべてのファイルは `bizreach-inc/` 直下に配置されています。

## 前提条件
- Azure NC24ads_A100_v4 (Ubuntu 22.04 または 24.04)
- ドライバー/CUDA インストール用の sudo 権限
- モデル/データセットを取得するためのインターネット接続

## 1) VM 初期セットアップ（root で実行）
```
sudo ./setup_azure_vm.sh
sudo reboot
```

## 2) Python 環境構築
```
sudo apt-get update && sudo apt-get install -y python3-venv python3.12-venv  # venv/ensurepip が無い場合
./setup_python_env.sh  # ~/splade_benchmark_env を作成
source ~/splade_benchmark_env/bin/activate
```

## 3) データ準備・ベンチマーク実行・可視化

### デフォルト（14M モデル）
```
./run_full_benchmark.sh
```

### 別のモデルを指定
```
# 28M モデル
MODEL="bizreach-inc/light-splade-japanese-28M" ./run_full_benchmark.sh

# 56M モデル
MODEL="bizreach-inc/light-splade-japanese-56M" ./run_full_benchmark.sh

# Python スクリプトを直接実行
python benchmark_splade.py --model bizreach-inc/light-splade-japanese-28M
```

### 複数モデルを連続ベンチマーク
```bash
for model in "bizreach-inc/light-splade-japanese-14M" \
             "bizreach-inc/light-splade-japanese-28M" \
             "bizreach-inc/light-splade-japanese-56M"; do
  MODEL="${model}" ./run_full_benchmark.sh
done
```

- データ: `data/`
- メトリクス: `results/metrics/*.json` および `.csv`
- プロット: `results/plots/`
- ログ: `results/logs/`

## 利用可能なモデル
- `bizreach-inc/light-splade-japanese-14M`（デフォルト、13.8M パラメータ）
- `bizreach-inc/light-splade-japanese-28M`（27.5M パラメータ）
- `bizreach-inc/light-splade-japanese-56M`（55.0M パラメータ）

## デフォルト設定
- バッチサイズ: 1, 8, 32, 64, 128, 256
- シーケンス長: 128, 256, 512, 1024
- ウォームアップ: 10 回
- 計測: 100 回
- 精度: FP32
- OOM 対応: CUDA OOM 時にバッチサイズを自動で半分に減らす

## トラブルシュート
- OOM: バッチサイズ自動半減で対応。さらに必要なら `--batch-sizes` を小さくしてください。
- GPU 未検出: `nvidia-smi` が動くか確認し、ドライバーを再インストールしてください。
- ダウンロードが遅い: `HF_HOME` を高速ディスクに設定し、`HF_HUB_ENABLE_HF_TRANSFER=1` を試してください。

# SPLADE ベンチマーク (Azure NC24ads_A100_v4 / Standard_D32s_v6)

Azure NC A100 上で `bizreach-inc/light-splade-japanese-14M` を計測するカスタムベンチマークです。バッチサイズとシーケンス長を変えながら、レイテンシ（P50/P95/P99）、スループット（QPS）、スパース性、GPU メトリクスを取得します。すべてのファイルは `bizreach-inc/` 直下に配置されています。

## 前提条件
- Azure NC24ads_A100_v4（GPU）または Standard_D32s_v6（CPU）
- Ubuntu 22.04 または 24.04
-（GPU の場合）ドライバー/CUDA インストール用の sudo 権限
- モデル/データセットを取得するためのインターネット接続

## 1) VM 初期セットアップ（root で実行）

### GPU VM（A100）
```bash
sudo ./setup_azure_vm.sh --device cuda
sudo reboot
```

### CPU VM（例: Standard_D32s_v6）
```bash
sudo ./setup_azure_vm.sh --device cpu
```

## 2) Python 環境構築
```bash
sudo apt-get update && sudo apt-get install -y python3-venv python3.12-venv  # venv/ensurepip が無い場合
./setup_python_env.sh --device cuda  # ~/splade_benchmark_env を作成
source ~/splade_benchmark_env/bin/activate
```

### CPU-only 環境
```bash
./setup_python_env.sh --device cpu
```

## 3) データ準備・ベンチマーク実行・可視化

### デフォルト（14M モデル）
```
./run_full_benchmark.sh
```

### オンライン検索向け（短文＋長文クエリ、E2E）

- 短文（例）: seq={16,32,64,128}, bs={1,2,4,8,16}
- 長文（例）: seq={256,512,1024}, bs={1,2,4}（代表点: seq=512, bs=1）

GPU（A100）の例:
```bash
PROFILE=online DEVICE=cuda VM_SKU=NC24ads_A100_v4 ./run_full_benchmark.sh
```

CPU（D32s v6）の例:
```bash
PROFILE=online DEVICE=cpu VM_SKU=Standard_D32s_v6 ./run_full_benchmark.sh
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

## CPU/GPU 両対応・公平比較の注意
- `--device cpu|cuda`（または `DEVICE=...`）で実行デバイスを明示します。
- GPU メトリクス（NVML）はベストエフォートです。NVML が使えない場合は GPU メトリクスを省略してベンチマークは継続します。
- データは `data/queries_{seq}.json` を seq_length ごとに利用します。必要なら `python scripts/prepare_data.py --seq-lengths ...` で作成してください。

## トラブルシュート
- OOM: バッチサイズ自動半減で対応。さらに必要なら `--batch-sizes` を小さくしてください。
- GPU 未検出: `nvidia-smi` が動くか確認し、ドライバーを再インストールしてください。
- ダウンロードが遅い: `HF_HOME` を高速ディスクに設定し、`HF_HUB_ENABLE_HF_TRANSFER=1` を試してください。

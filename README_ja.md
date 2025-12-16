# SPLADE ベンチマーク (Azure NC24ads_A100_v4 / Standard_D32s_v6)

Azure NC A100（GPU）および Standard_D32s_v6（CPU）上で `bizreach-inc/light-splade-japanese` モデルを計測するカスタムベンチマークです。バッチサイズとシーケンス長を変えながら、レイテンシ（P50/P95/P99）、スループット（QPS）、スパース性、GPU メトリクスを取得します。

## 前提条件
- Azure NC24ads_A100_v4（GPU）または Standard_D32s_v6（CPU）
- Ubuntu 22.04 または 24.04
- （GPU の場合）ドライバー/CUDA インストール用の sudo 権限
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

### デフォルト（full プロファイル、14M モデル）
```bash
./run_full_benchmark.sh
```

**full** プロファイルの設定:
- バッチサイズ: 1, 8, 32, 64, 128, 256
- シーケンス長: 128, 256, 512, 1024
- ウォームアップ: 10 回
- 計測: 100 回
- パディング: longest（動的）

### オンライン検索向け（短文＋長文クエリ）
```bash
PROFILE=online DEVICE=cuda VM_SKU=NC24ads_A100_v4 ./run_full_benchmark.sh
```

プロファイルの詳細:
- **短文クエリグリッド**: seq={16,32,64,128}, bs={1,2,4,8,16}
  - ウォームアップ: 30 回、計測: 500 回
- **長文クエリグリッド**: seq={256,512,1024}, bs={1,2,4}
  - ウォームアップ: 50 回、計測: 500 回
  - 代表点: seq=512, bs=1
- パディング: max_length（固定）

### CPU ベンチマーク例
```bash
PROFILE=online DEVICE=cpu VM_SKU=Standard_D32s_v6 ./run_full_benchmark.sh
```

### 別のモデルを指定
```bash
# 28M モデル
MODEL="bizreach-inc/light-splade-japanese-28M" ./run_full_benchmark.sh

# 56M モデル
MODEL="bizreach-inc/light-splade-japanese-56M" ./run_full_benchmark.sh

# Python スクリプトを直接実行
python benchmark_splade.py --model bizreach-inc/light-splade-japanese-28M --device cuda --vm-sku NC24ads_A100_v4
```

### 複数モデルを連続ベンチマーク
```bash
for model in "bizreach-inc/light-splade-japanese-14M" \
             "bizreach-inc/light-splade-japanese-28M" \
             "bizreach-inc/light-splade-japanese-56M"; do
  MODEL="${model}" PROFILE=full ./run_full_benchmark.sh
done
```

### 出力先
- データ: [`data/`](data/)
- メトリクス: [`results/metrics/*.json`](results/metrics/) および `*.csv`
- プロット: [`results/plots/`](results/plots/)
- ログ: [`results/logs/`](results/logs/)

## 利用可能なモデル
- [`bizreach-inc/light-splade-japanese-14M`](https://huggingface.co/bizreach-inc/light-splade-japanese-14M)（デフォルト、13.8M パラメータ）
- [`bizreach-inc/light-splade-japanese-28M`](https://huggingface.co/bizreach-inc/light-splade-japanese-28M)（27.5M パラメータ）
- [`bizreach-inc/light-splade-japanese-56M`](https://huggingface.co/bizreach-inc/light-splade-japanese-56M)（55.0M パラメータ）

## 高度なオプション

### CPU スレッドチューニング
```bash
python benchmark_splade.py \
  --model bizreach-inc/light-splade-japanese-14M \
  --device cpu \
  --cpu-threads 16 \
  --interop-threads 1 \
  --vm-sku Standard_D32s_v6
```

### パディングポリシー
- `--padding-policy longest`（デフォルト）: バッチごとに動的パディング
- `--padding-policy max_length`: max_length に固定パディング

### GPU モニタリング
- CUDA 実行時はデフォルトで有効
- `--gpu-monitor false` で無効化
- `--monitor-interval 0.5`（秒）でサンプリング間隔を調整

## 主な機能
- **OOM 対応**: CUDA OOM 時にバッチサイズを自動で半分に減らす
- **GPU モニタリング**: NVML ベースのベストエフォートメトリクス（利用不可時は自動スキップ）
- **シーケンス別データセット**: [`scripts/prepare_data.py`](scripts/prepare_data.py) で生成された [`data/queries_{seq}.json`](data/) を使用
- **包括的メトリクス**: パーセンタイル（P50/P95/P99）、スループット（QPS）、スパース性、GPU 利用率
- **実行メタデータ**: デバイス、VM SKU、バッチサイズ（要求値 vs 実際値）、パディングポリシー、スレッド設定

## プロジェクト構成
```
.
├── benchmark_splade.py          # メインベンチマークスクリプト
├── run_full_benchmark.sh        # エンドツーエンドベンチマークランナー
├── setup_azure_vm.sh            # VM ブートストラップ（ドライバー/CUDA）
├── setup_python_env.sh          # Python 環境セットアップ
├── requirements.txt             # Python 依存関係
├── models/
│   └── splade_model.py          # SPLADE モデルラッパー
├── utils/
│   ├── benchmark_runner.py      # ベンチマークオーケストレーション
│   ├── metrics_collector.py     # 結果収集＆エクスポート
│   ├── gpu_monitor.py           # NVML ベース GPU モニタリング
│   └── logger.py                # ロギングユーティリティ
├── scripts/
│   ├── prepare_data.py          # データセット準備
│   └── visualize_results.py     # プロット生成
└── results/
    ├── metrics/                 # JSON/CSV 出力
    ├── plots/                   # 可視化出力
    └── logs/                    # 実行ログ
```

## トラブルシューティング
- **OOM エラー**: CUDA OOM 時にバッチサイズを自動で半分にします。必要に応じて `--batch-sizes` を手動で減らしてください。
- **GPU 未検出**: `nvidia-smi` が動作するか確認し、[`setup_azure_vm.sh`](setup_azure_vm.sh) でドライバーを再インストールしてください。
- **NVML 利用不可**: GPU モニタリングは優雅に劣化し、GPU メトリクスなしでベンチマークは継続します。
- **ダウンロードが遅い**: `HF_HOME` を高速ディスクに設定し、オプションで `HF_HUB_ENABLE_HF_TRANSFER=1` を試してください。
- **データセットが見つからない**: `python scripts/prepare_data.py --seq-lengths 128 256 512 1024` を実行してテストデータを生成してください。

## 環境変数
- `DEVICE`: `cpu` または `cuda`（デフォルト: `cuda`）
- `PROFILE`: `full` または `online`（デフォルト: `full`）
- `MODEL`: HuggingFace モデル名（デフォルト: `bizreach-inc/light-splade-japanese-14M`）
- `VM_SKU`: 結果メタデータ用のオプション注釈（例: `NC24ads_A100_v4`、`Standard_D32s_v6`）
- `ENV_DIR`: Python 仮想環境パス（デフォルト: `~/splade_benchmark_env`）
- `TOKENIZERS_PARALLELISM`: 安定した GPU 実行のため `false` に設定（[`run_full_benchmark.sh`](run_full_benchmark.sh) で自動設定）

## ライセンス
このベンチマークツールキットは、パフォーマンス評価目的で現状のまま提供されます。

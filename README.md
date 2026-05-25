# 🚽 排尿予測アプリ - 機械学習モデル

Ali et al. (2022) "Mitigating urinary incontinence condition using machine learning"
の手法をベースにした排尿時間予測モデルです。

## 概要

飲料の種類・量・時刻、年齢・BMIなどから、次回の排尿までの時間を予測します。

| 予測クラス | 意味 |
|-----------|------|
| Class 0   | ＜30分 |
| Class 1   | 31〜60分 |
| Class 2   | 61〜90分 |
| Class 3   | ＞90分 |

## セットアップ

```bash
# リポジトリをクローン
git clone <repo-url>
cd my-app

# 仮想環境を作成・有効化
python3 -m venv venv
source venv/bin/activate   # Windowsは: venv\Scripts\activate

# ライブラリをインストール
pip install -r requirements.txt
```

## 使い方

```bash
# Jupyter Notebookを起動（分析・実験）
jupyter notebook

# モデルを学習
python src/train.py

# 予測を実行
python src/predict.py
```

## プロジェクト構成

```
my-app/
├── data/           # データ（CSV等）
├── notebooks/      # Jupyter Notebook（分析・実験）
├── src/            # Pythonスクリプト
│   ├── preprocess.py  # データ前処理
│   ├── train.py       # モデル学習
│   └── predict.py     # 予測
├── models/         # 学習済みモデル（.pkl）
├── requirements.txt
└── README.md
```

## 参考論文

Ali, H., Ahmed, A., Olivos, C., Khamis, K., & Liu, J. (2022).
Mitigating urinary incontinence condition using machine learning.
*BMC Medical Informatics and Decision Making*, 22, 243.
https://doi.org/10.1186/s12911-022-01987-3

"""
データ前処理モジュール
============================================================
論文 Ali et al. (2022) の前処理手順を再現します。

処理の流れ:
  1. データ読み込み
  2. 外れ値の除去
  3. 欠損値処理（KNN補完）
  4. カテゴリ変数のエンコーディング
  5. 数値変数のスケーリング
  6. 学習・テスト分割（80:20）
  7. SMOTE によるオーバーサンプリング（クラス不均衡対策）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Union

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE


# ================================================================
# 設定値
# ================================================================

# 特徴量として使う列（論文 Table 2 と Table 4 を参考）
FEATURE_COLS = [
    # 飲み物系
    'volume',       # 合計飲料量（ml）
    'ndrinks',      # 飲んだ杯数
    'vol_inp',      # 最初の1杯の量（ml）
    'time_inp',     # 最初の排尿〜次の飲水までの時間（分）
    'water',        # 水を飲んだか（0/1）
    'coffee',       # コーヒーを飲んだか（0/1）
    'juice',        # ジュースを飲んだか（0/1）
    'milk',         # 牛乳を飲んだか（0/1）
    'soda',         # 炭酸を飲んだか（0/1）
    'alcohol',      # アルコールを飲んだか（0/1）
    'tea',          # お茶を飲んだか（0/1）
    # 身体情報
    'age',          # 年齢
    'bmi',          # BMI
    'weight',       # 体重
    'height',       # 身長
    # 生活習慣
    'gender_enc',       # 性別（エンコード済み）
    'employment_enc',   # 雇用形態（エンコード済み）
    'level_exercise_enc',  # 運動レベル（エンコード済み）
    'alcoholic',    # 常習的な飲酒（0/1）
    'smoking',      # 喫煙（0/1）
]

TARGET_COL = 'target'

# 外れ値除去の上限パーセンタイル
OUTLIER_PERCENTILE = 95

# 欠損値許容率（これ以上欠損している列は削除）
MISSING_DROP_THRESHOLD = 0.10

# テストデータの割合
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ================================================================
# 前処理関数
# ================================================================

def load_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """CSVデータを読み込む"""
    df = pd.read_csv(data_path)
    print(f"📂 データ読み込み完了: {len(df):,} 件, {df.shape[1]} 列")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    外れ値を除去します（論文の手順に準拠）。
    - volume と ndrinks の 95パーセンタイル超を除去
    - 排尿間隔が 300分超（5時間超）を除去
    - 排尿間隔が 0 のレコードを除去
    """
    before = len(df)
    df = df.copy()

    # volume の上限
    vol_upper = df['volume'].quantile(OUTLIER_PERCENTILE / 100)
    df = df[df['volume'] <= vol_upper]

    # ndrinks の上限
    nd_upper = df['ndrinks'].quantile(OUTLIER_PERCENTILE / 100)
    df = df[df['ndrinks'] <= nd_upper]

    # 排尿間隔の上限（5時間 = 300分）
    if 'inter_release_time' in df.columns:
        df = df[df['inter_release_time'] <= 300]
        df = df[df['inter_release_time'] > 0]

    after = len(df)
    print(f"🗑️  外れ値除去: {before:,} → {after:,} 件 ({before - after} 件削除)")
    return df.reset_index(drop=True)


def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    カテゴリ変数をラベルエンコーディングします。
    エンコーダを辞書で返し、予測時に再利用できるようにします。
    """
    df = df.copy()
    encoders = {}

    cat_cols = {
        'gender':         'gender_enc',
        'employment':     'employment_enc',
        'level_exercise': 'level_exercise_enc',
    }

    for src, dst in cat_cols.items():
        if src in df.columns:
            le = LabelEncoder()
            df[dst] = le.fit_transform(df[src].astype(str))
            encoders[src] = le

    print(f"🔤 カテゴリ変数をエンコード: {list(cat_cols.keys())}")
    return df, encoders


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    欠損値を処理します（論文の手順に準拠）。
    1. 欠損率が高い列（10%超）は削除
    2. 数値列は KNN 補完
    3. カテゴリ列の欠損行は削除
    """
    df = df.copy()

    # 欠損率を確認
    missing_rate = df.isnull().mean()
    high_missing = missing_rate[missing_rate > MISSING_DROP_THRESHOLD].index.tolist()
    if high_missing:
        df = df.drop(columns=high_missing)
        print(f"🗑️  欠損率 {MISSING_DROP_THRESHOLD*100:.0f}%超の列を削除: {high_missing}")

    # 数値列に KNN 補完
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if df[numeric_cols].isnull().any().any():
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        print(f"🩹 KNN補完を適用: {len(numeric_cols)} 列")

    # カテゴリ列の残余欠損行を削除
    before = len(df)
    df = df.dropna()
    after = len(df)
    if before != after:
        print(f"🗑️  欠損行を削除: {before - after} 件")

    return df.reset_index(drop=True)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    特徴量 X と目的変数 y を切り出します。
    FEATURE_COLS に含まれる列のみを使用します。
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"⚠️  以下の列が見つかりません（スキップ）: {missing}")

    X = df[available]
    y = df[TARGET_COL]
    print(f"📊 特徴量: {len(available)} 列, サンプル数: {len(X):,}")
    print(f"   クラス分布: {y.value_counts().sort_index().to_dict()}")
    return X, y


def scale_features(X_train: pd.DataFrame,
                   X_test:  pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    数値特徴量を StandardScaler でスケーリングします。
    ※ スケーラーは学習データにのみ fit し、テストデータには transform のみ適用します。
    """
    scaler      = StandardScaler()
    X_train_sc  = pd.DataFrame(scaler.fit_transform(X_train),
                               columns=X_train.columns, index=X_train.index)
    X_test_sc   = pd.DataFrame(scaler.transform(X_test),
                               columns=X_test.columns, index=X_test.index)
    print(f"⚖️  StandardScaler でスケーリング完了")
    return X_train_sc, X_test_sc, scaler


def apply_smote(X_train: pd.DataFrame,
                y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    SMOTE（Synthetic Minority Oversampling Technique）でクラス不均衡を解消します。
    学習データにのみ適用します（テストデータには適用しない）。
    """
    print(f"📈 SMOTE 適用前: {y_train.value_counts().sort_index().to_dict()}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=TARGET_COL)
    print(f"📈 SMOTE 適用後: {y_res.value_counts().sort_index().to_dict()}")
    return X_res, y_res


# ================================================================
# メイン前処理パイプライン
# ================================================================

def preprocess_pipeline(
    data_path: Union[str, Path],
    test_size: float = TEST_SIZE,
    apply_scaling: bool = True,
    apply_smote_flag: bool = True,
) -> dict:
    """
    前処理パイプラインを一括実行します。

    Returns:
        dict: {
            'X_train', 'X_test', 'y_train', 'y_test',
            'feature_names', 'encoders', 'scaler'
        }
    """
    print("=" * 55)
    print("🔧 データ前処理を開始します")
    print("=" * 55)

    df = load_data(data_path)
    df = remove_outliers(df)
    df, encoders = encode_categoricals(df)
    df = handle_missing_values(df)
    X, y = prepare_features(df)

    # 学習・テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n✂️  学習/テスト分割: train={len(X_train):,}, test={len(X_test):,}")

    scaler = None
    if apply_scaling:
        X_train, X_test, scaler = scale_features(X_train, X_test)

    if apply_smote_flag:
        X_train, y_train = apply_smote(X_train, y_train)

    print("\n✅ 前処理完了！")
    print("=" * 55)

    return {
        'X_train':       X_train,
        'X_test':        X_test,
        'y_train':       y_train,
        'y_test':        y_test,
        'feature_names': list(X_train.columns),
        'encoders':      encoders,
        'scaler':        scaler,
    }


# ================================================================
# 動作確認
# ================================================================

if __name__ == '__main__':
    data_path = Path(__file__).parent.parent / 'data' / 'sample_data.csv'
    if not data_path.exists():
        print("❌ sample_data.csv が見つかりません。")
        print("   先に generate_sample_data.py を実行してください。")
    else:
        result = preprocess_pipeline(data_path)
        print(f"\n学習データ形状: {result['X_train'].shape}")
        print(f"テストデータ形状: {result['X_test'].shape}")
        print(f"使用特徴量: {result['feature_names']}")

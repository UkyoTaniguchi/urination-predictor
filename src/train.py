"""
モデル学習スクリプト
============================================================
論文 Ali et al. (2022) の手順に従い、以下を実行します：

  1. 特徴量選択（chi-square / Random Forest + RFE）
  2. XGBoost モデルの学習（k-fold 交差検証付き）
  3. テストデータで評価（accuracy / precision / recall / F1）
  4. SHAP 値でモデルを解釈
  5. 学習済みモデルを保存

実行方法:
    cd ~/my-app/urination-predictor
    source venv/bin/activate
    python src/train.py
"""

import sys
import warnings
import joblib
from pathlib import Path
from typing import Union, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')     # GUI なし環境でも画像を保存できる設定
import seaborn as sns
import shap

from sklearn.feature_selection import (
    SelectKBest, chi2,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# プロジェクトルートを sys.path に追加（src から import できるようにする）
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import preprocess_pipeline

# ================================================================
# 設定値
# ================================================================

DATA_PATH   = Path(__file__).parent.parent / 'data' / 'sample_data.csv'
MODEL_DIR   = Path(__file__).parent.parent / 'models'
FIGURES_DIR = Path(__file__).parent.parent / 'notebooks' / 'figures'

MODEL_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

CLASS_LABELS = {0: '< 30分', 1: '31〜60分', 2: '61〜90分', 3: '> 90分'}
N_SPLITS = 5        # k-fold の分割数
RANDOM_STATE = 42


# ================================================================
# 1. 特徴量選択
# ================================================================

def select_features_chi2(X_train: pd.DataFrame,
                          y_train: pd.Series,
                          k: int = 9) -> List[str]:
    """
    カイ二乗検定（chi-square）で重要特徴量を選択します。
    ※ 入力値が負にならないよう、最小値シフトを行います。
    """
    X_pos = X_train - X_train.min()   # 負値を 0 にシフト
    selector = SelectKBest(chi2, k=k)
    selector.fit(X_pos, y_train)
    selected = X_train.columns[selector.get_support()].tolist()
    print(f"  chi-square が選択した特徴量 ({k}個): {selected}")
    return selected


def select_features_rf_rfe(X_train: pd.DataFrame,
                            y_train: pd.Series,
                            n_features: int = 9) -> List[str]:
    """
    ランダムフォレスト + RFE（Recursive Feature Elimination）で
    重要特徴量を選択します。
    """
    rf  = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rfe = RFE(estimator=rf, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    selected = X_train.columns[rfe.support_].tolist()
    print(f"  RF + RFE が選択した特徴量 ({n_features}個): {selected}")
    return selected


def run_feature_selection(X_train: pd.DataFrame,
                           y_train: pd.Series) -> Dict[str, List[str]]:
    """
    複数の特徴量選択手法を実行し、結果を返します。
    """
    print("\n🔍 特徴量選択を実行中...")
    results = {}

    # 全特徴量（ベースライン）
    results['all'] = list(X_train.columns)

    # chi-square
    results['chi2'] = select_features_chi2(X_train, y_train, k=9)

    # Random Forest + RFE
    results['rf_rfe'] = select_features_rf_rfe(X_train, y_train, n_features=9)

    # 共通して選ばれた特徴量
    common = list(set(results['chi2']) & set(results['rf_rfe']))
    print(f"\n  ⭐ 両手法で共通して選ばれた特徴量 ({len(common)}個): {common}")
    results['common'] = common

    return results


# ================================================================
# 2. XGBoost モデルの学習・評価
# ================================================================

def make_xgb_model() -> XGBClassifier:
    """XGBoost モデルを生成します（論文に準拠したハイパーパラメータ）"""
    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        verbosity=0,
    )


def train_and_evaluate(X_train: pd.DataFrame,
                        X_test:  pd.DataFrame,
                        y_train: pd.Series,
                        y_test:  pd.Series,
                        feature_names: List[str],
                        label: str) -> dict:
    """
    指定した特徴量で XGBoost を学習・評価します。
    k-fold 交差検証も実行します。
    """
    X_tr = X_train[feature_names]
    X_te = X_test[feature_names]

    model = make_xgb_model()

    # k-fold 交差検証
    kf      = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_acc  = cross_val_score(model, X_tr, y_train, cv=kf, scoring='accuracy')

    # 本番学習
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    # 評価指標
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n  [{label}] "
          f"Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}"
          f"  (CV Accuracy={cv_acc.mean():.3f} ± {cv_acc.std():.3f})")

    return {
        'label':     label,
        'model':     model,
        'features':  feature_names,
        'y_pred':    y_pred,
        'accuracy':  acc,
        'precision': prec,
        'recall':    rec,
        'f1':        f1,
        'cv_mean':   cv_acc.mean(),
        'cv_std':    cv_acc.std(),
    }


# ================================================================
# 3. 結果の可視化
# ================================================================

def plot_confusion_matrix(y_test: pd.Series,
                           y_pred: np.ndarray,
                           title: str) -> None:
    """混同行列を保存します"""
    cm = confusion_matrix(y_test, y_pred)
    labels = [f"Class {k}\n{v}" for k, v in CLASS_LABELS.items()]

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'混同行列: {title}', fontsize=13)
    plt.ylabel('実際のクラス')
    plt.xlabel('予測クラス')
    plt.tight_layout()

    save_path = FIGURES_DIR / f'confusion_matrix_{title.replace(" ", "_")}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 混同行列を保存: {save_path.name}")


def plot_model_comparison(results: List[dict]) -> None:
    """各手法の比較棒グラフを保存します"""
    labels  = [r['label'] for r in results]
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    x = np.arange(len(labels))
    width = 0.20

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [r[metric] for r in results]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=color)

    ax.set_ylabel('スコア')
    ax.set_title('特徴量選択手法ごとの XGBoost 評価結果')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.axhline(y=0.70, color='gray', linestyle='--', alpha=0.7, label='論文の目標値 (0.70)')
    plt.tight_layout()

    save_path = FIGURES_DIR / 'model_comparison.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 比較グラフを保存: {save_path.name}")


def plot_shap_values(model: XGBClassifier,
                     X_test: pd.DataFrame,
                     feature_names: List[str],
                     label: str) -> None:
    """SHAP値のビーズスウォームプロットを保存します"""
    X_te = X_test[feature_names]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te)

    # 多クラス分類では shap_values の形が SHAP/XGBoost バージョンにより異なる
    # パターン1: list of (n_samples, n_features) — 古い API
    # パターン2: ndarray (n_samples, n_features, n_classes) — SHAP 0.40〜0.45 系
    # パターン3: ndarray (n_samples, n_classes, n_features) — 一部新しい API
    shap_arr = np.array(shap_values)

    if shap_arr.ndim == 3:
        # どの軸が n_features (= len(feature_names)) かを確認して平均
        n_feat = len(feature_names)
        if shap_arr.shape[1] == n_feat:
            # (n_samples, n_features, n_classes) → axis=(0,2) で平均
            importance = np.abs(shap_arr).mean(axis=(0, 2))
        else:
            # (n_samples, n_classes, n_features) or (n_classes, n_samples, n_features)
            importance = np.abs(shap_arr).mean(axis=(0, 1))
    elif shap_arr.ndim == 2:
        # (n_samples, n_features) — バイナリ分類
        importance = np.abs(shap_arr).mean(axis=0)
    else:
        # list of 2D arrays (旧 API)
        importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)

    mean_shap = pd.Series(importance, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    mean_shap.plot(kind='barh', color='steelblue')
    plt.gca().invert_yaxis()
    plt.title(f'SHAP 特徴量重要度: {label}', fontsize=13)
    plt.xlabel('平均 |SHAP 値|')
    plt.tight_layout()

    save_path = FIGURES_DIR / f'shap_{label.replace(" ", "_")}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 SHAP グラフを保存: {save_path.name}")


# ================================================================
# 4. メイン実行
# ================================================================

def main():
    print("=" * 55)
    print("🤖 排尿予測モデルの学習を開始します")
    print("=" * 55)

    # ─── 前処理 ────────────────────────────────────────
    if not DATA_PATH.exists():
        print(f"❌ データが見つかりません: {DATA_PATH}")
        print("   先に generate_sample_data.py を実行してください。")
        sys.exit(1)

    prep = preprocess_pipeline(DATA_PATH)
    X_train = prep['X_train']
    X_test  = prep['X_test']
    y_train = prep['y_train']
    y_test  = prep['y_test']

    # ─── 特徴量選択 ────────────────────────────────────
    feature_groups = run_feature_selection(X_train, y_train)

    # ─── 各特徴量グループでモデルを学習・評価 ─────────
    print("\n🏋️  XGBoost モデルを学習・評価中...")
    results = []
    for name, features in feature_groups.items():
        if len(features) == 0:
            continue
        res = train_and_evaluate(X_train, X_test, y_train, y_test, features, name)
        results.append(res)

    # ─── 最良モデルを選択 ──────────────────────────────
    best = max(results, key=lambda r: r['f1'])
    print(f"\n🏆 最良モデル: [{best['label']}]")
    print(f"   Accuracy={best['accuracy']:.3f}  Precision={best['precision']:.3f}"
          f"  Recall={best['recall']:.3f}  F1={best['f1']:.3f}")

    # 詳細レポート
    print("\n📋 詳細レポート（最良モデル）:")
    print(classification_report(
        y_test, best['y_pred'],
        target_names=[CLASS_LABELS[i] for i in range(4)],
        zero_division=0,
    ))

    # ─── 可視化 ────────────────────────────────────────
    print("\n🎨 グラフを生成中...")
    plot_confusion_matrix(y_test, best['y_pred'], best['label'])
    plot_model_comparison(results)
    plot_shap_values(best['model'], X_test, best['features'], best['label'])

    # ─── モデルを保存 ──────────────────────────────────
    model_data = {
        'model':         best['model'],
        'feature_names': best['features'],
        'encoders':      prep['encoders'],
        'scaler':        prep['scaler'],
        'class_labels':  CLASS_LABELS,
        'metrics': {
            'accuracy':  best['accuracy'],
            'precision': best['precision'],
            'recall':    best['recall'],
            'f1':        best['f1'],
        }
    }
    save_path = MODEL_DIR / 'xgb_urination_model.pkl'
    joblib.dump(model_data, save_path)
    print(f"\n💾 モデルを保存: {save_path}")

    print("\n✅ 学習完了！")
    print("=" * 55)


if __name__ == '__main__':
    main()

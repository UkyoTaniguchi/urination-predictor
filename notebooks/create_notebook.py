"""
Jupyter Notebook 自動生成スクリプト
実行すると 01_prototype.ipynb が作られます。
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {'name': 'python', 'version': '3.9.10'}
}

def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)

nb.cells = [

# ── タイトル ─────────────────────────────────────────────────────
md("""# 🚽 排尿予測モデル — プロトタイプ

**論文:** Ali et al. (2022) *Mitigating urinary incontinence condition using machine learning*

このノートブックでは、以下を順を追って体験できます：
1. 📊 サンプルデータの生成・探索
2. 🔧 データ前処理
3. 🔍 特徴量選択
4. 🤖 XGBoost モデルの学習・評価
5. 🔮 SHAP による解釈
6. 💾 モデルの保存と読み込み

---"""),

# ── セル 1: ライブラリ ────────────────────────────────────────────
md("## 1. ライブラリの読み込み"),
code("""\
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '../src')   # src/ のスクリプトを import できるようにする

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

# 日本語フォント設定（Mac）
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

print('✅ 全ライブラリの読み込み完了')"""),

# ── セル 2: データ生成 ────────────────────────────────────────────
md("""## 2. サンプルデータの生成

実際のデータが集まるまでの間、論文の設定を参考に合成データを生成します。"""),
code("""\
from generate_sample_data import generate_dataset

df_raw = generate_dataset()
df_raw.to_csv('../data/sample_data.csv', index=False)

print(f'レコード数: {len(df_raw):,} 件')
print(f'参加者数 : {df_raw["participant_id"].nunique()} 名')
print(f'列数     : {df_raw.shape[1]} 列')
df_raw.head()"""),

# ── セル 3: EDA ────────────────────────────────────────────────
md("""## 3. データ探索（EDA）

### 3-1. 基本統計量"""),
code("""\
df_raw.describe().round(2)"""),

md("### 3-2. 目的変数の分布"),
code("""\
CLASS_LABELS = {0: '< 30分', 1: '31〜60分', 2: '61〜90分', 3: '> 90分'}

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# 件数
counts = df_raw['target'].value_counts().sort_index()
axes[0].bar([CLASS_LABELS[i] for i in counts.index], counts.values, color='steelblue')
axes[0].set_title('クラス別レコード数')
axes[0].set_ylabel('件数')
for i, (xi, yi) in enumerate(zip(axes[0].get_xticks(), counts.values)):
    axes[0].text(xi, yi + 5, str(yi), ha='center', fontsize=10)

# 割合
axes[1].pie(counts.values, labels=[CLASS_LABELS[i] for i in counts.index],
            autopct='%1.1f%%', startangle=90, colors=plt.cm.Set2.colors)
axes[1].set_title('クラス別割合')

plt.suptitle('目的変数（排尿までの時間）の分布', fontsize=13)
plt.tight_layout()
plt.show()"""),

md("### 3-3. 飲み物の種類と排尿時間の関係"),
code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# 飲み物の種類ごとの排尿間隔
drink_time = df_raw.groupby('drink_type')['inter_release_time'].mean().sort_values()
axes[0].barh(drink_time.index, drink_time.values, color='coral')
axes[0].set_title('飲み物の種類別 平均排尿間隔（分）')
axes[0].set_xlabel('平均排尿間隔（分）')
axes[0].axvline(x=60, color='gray', linestyle='--', label='60分ライン')
axes[0].legend()

# 年齢グループ別の平均排尿間隔
age_time = df_raw.groupby('age_group')['inter_release_time'].mean()
order = ['child', 'youth', 'adult', 'senior']
order = [o for o in order if o in age_time.index]
axes[1].bar(order, [age_time[o] for o in order], color='mediumseagreen')
axes[1].set_title('年齢グループ別 平均排尿間隔（分）')
axes[1].set_ylabel('平均排尿間隔（分）')

plt.tight_layout()
plt.show()"""),

md("### 3-4. 特徴量間の相関行列"),
code("""\
numeric_cols = ['age', 'bmi', 'volume', 'ndrinks', 'vol_inp', 'time_inp',
                'water', 'coffee', 'alcohol', 'tea', 'target']
corr = df_raw[numeric_cols].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, center=0, square=True, linewidths=0.5)
plt.title('特徴量間の相関行列', fontsize=13)
plt.tight_layout()
plt.show()"""),

# ── セル 4: 前処理 ────────────────────────────────────────────
md("""## 4. データ前処理

論文の手順に従って前処理を行います：
1. 外れ値除去
2. カテゴリ変数のエンコーディング
3. 欠損値処理（KNN 補完）
4. 学習/テスト分割（80:20）
5. SMOTE によるオーバーサンプリング"""),
code("""\
from preprocess import preprocess_pipeline

prep = preprocess_pipeline('../data/sample_data.csv')

X_train = prep['X_train']
X_test  = prep['X_test']
y_train = prep['y_train']
y_test  = prep['y_test']

print(f'\\n学習データ: {X_train.shape}')
print(f'テストデータ: {X_test.shape}')
print(f'\\n学習データのクラス分布（SMOTE後）:')
print(y_train.value_counts().sort_index())"""),

# ── セル 5: 特徴量選択 ────────────────────────────────────────
md("""## 5. 特徴量選択

chi-square と Random Forest + RFE で重要な特徴量を選びます。"""),
code("""\
# ── chi-square ──────────────────────────────
X_pos = X_train - X_train.min()   # 負値を 0 にシフト（chi2 の前処理）
selector_chi2 = SelectKBest(chi2, k=9)
selector_chi2.fit(X_pos, y_train)
features_chi2 = X_train.columns[selector_chi2.get_support()].tolist()
scores_chi2   = selector_chi2.scores_[selector_chi2.get_support()]
print('chi-square 選択特徴量:')
for f, s in sorted(zip(features_chi2, scores_chi2), key=lambda x: -x[1]):
    print(f'  {f:25}: スコア={s:.1f}')

# ── RF + RFE ────────────────────────────────
rf  = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=9)
rfe.fit(X_train, y_train)
features_rfe = X_train.columns[rfe.support_].tolist()
print(f'\\nRF+RFE 選択特徴量: {features_rfe}')

# 共通特徴量
common = list(set(features_chi2) & set(features_rfe))
print(f'\\n⭐ 共通特徴量: {common}')"""),

md("### 特徴量重要度の可視化"),
code("""\
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X_train, y_train)
importances = pd.Series(rf_full.feature_importances_,
                        index=X_train.columns).sort_values(ascending=True)

plt.figure(figsize=(8, 6))
importances.tail(15).plot(kind='barh', color='steelblue')
plt.title('ランダムフォレスト 特徴量重要度（上位15個）')
plt.xlabel('重要度')
plt.tight_layout()
plt.show()"""),

# ── セル 6: モデル学習 ────────────────────────────────────────
md("""## 6. XGBoost モデルの学習・評価

各特徴量グループで XGBoost を学習し、精度を比較します。"""),
code("""\
RANDOM_STATE = 42

def train_xgb(X_tr, X_te, y_tr, y_te, feature_names, label):
    X_tr_ = X_tr[feature_names]
    X_te_ = X_te[feature_names]

    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=RANDOM_STATE, verbosity=0
    )

    # k-fold 交差検証
    kf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_acc = cross_val_score(model, X_tr_, y_tr, cv=kf, scoring='accuracy')

    model.fit(X_tr_, y_tr)
    y_pred = model.predict(X_te_)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_te, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_te, y_pred, average='weighted', zero_division=0)

    return {'label': label, 'model': model, 'features': feature_names,
            'y_pred': y_pred, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1,
            'cv_mean': cv_acc.mean(), 'cv_std': cv_acc.std()}

# 3つの特徴量グループで比較
feature_groups = {
    'All features':  list(X_train.columns),
    'chi2 (k=9)':    features_chi2,
    'RF+RFE (k=9)':  features_rfe,
}
if common:
    feature_groups['Common'] = common

results = []
print('特徴量グループ別の評価結果:')
print(f'  {"グループ":<20} {"Accuracy":>9} {"Precision":>9} {"Recall":>9} {"F1":>9} {"CV Acc":>12}')
print('  ' + '-'*72)
for name, features in feature_groups.items():
    if not features: continue
    res = train_xgb(X_train, X_test, y_train, y_test, features, name)
    results.append(res)
    print(f'  {name:<20} {res["accuracy"]:>9.3f} {res["precision"]:>9.3f}'
          f' {res["recall"]:>9.3f} {res["f1"]:>9.3f}'
          f' {res["cv_mean"]:>6.3f}±{res["cv_std"]:.3f}')

best = max(results, key=lambda r: r['f1'])
print(f'\\n🏆 最良モデル: [{best["label"]}]  F1={best["f1"]:.3f}')"""),

md("### 結果の比較グラフ"),
code("""\
labels  = [r['label'] for r in results]
metrics = ['accuracy', 'precision', 'recall', 'f1']
colors  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

x = np.arange(len(labels))
width = 0.18

fig, ax = plt.subplots(figsize=(11, 5))
for i, (metric, color) in enumerate(zip(metrics, colors)):
    vals = [r[metric] for r in results]
    ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=color)

ax.set_ylabel('スコア')
ax.set_title('特徴量グループ × 評価指標')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(labels, rotation=10)
ax.set_ylim(0, 1.05)
ax.axhline(y=0.70, color='gray', linestyle='--', alpha=0.7, label='目標値 0.70')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()"""),

md("### 混同行列（最良モデル）"),
code("""\
cm = confusion_matrix(y_test, best['y_pred'])
label_names = [f'Class {k}\\n{v}' for k, v in CLASS_LABELS.items()]

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names)
plt.title(f'混同行列: {best["label"]}', fontsize=13)
plt.ylabel('実際のクラス')
plt.xlabel('予測クラス')
plt.tight_layout()
plt.show()

print('\\n詳細分類レポート:')
print(classification_report(y_test, best['y_pred'],
      target_names=[CLASS_LABELS[i] for i in range(4)], zero_division=0))"""),

# ── セル 7: SHAP ────────────────────────────────────────────
md("""## 7. SHAP 値でモデルを解釈する

**SHAP（SHapley Additive exPlanations）** は、各特徴量が予測にどれだけ貢献しているかをゲーム理論に基づいて数値化します。"""),
code("""\
X_test_best = X_test[best['features']]

explainer   = shap.TreeExplainer(best['model'])
shap_values = explainer.shap_values(X_test_best)

# 全クラスのSHAP絶対値の平均
if isinstance(shap_values, list):
    mean_shap = pd.Series(
        np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0),
        index=best['features']
    ).sort_values(ascending=True)
else:
    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=best['features']
    ).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
mean_shap.plot(kind='barh', color='steelblue')
plt.title(f'SHAP 特徴量重要度（{best["label"]}）')
plt.xlabel('平均 |SHAP 値|')
plt.tight_layout()
plt.show()

print('\\n重要度ランキング（上位）:')
for i, (feat, val) in enumerate(mean_shap.sort_values(ascending=False).head(9).items(), 1):
    print(f'  {i}. {feat:25}: {val:.4f}')"""),

# ── セル 8: モデル保存 ────────────────────────────────────────
md("## 8. モデルの保存と読み込み"),
code("""\
model_data = {
    'model':         best['model'],
    'feature_names': best['features'],
    'encoders':      prep['encoders'],
    'scaler':        prep['scaler'],
    'class_labels':  CLASS_LABELS,
    'metrics': {k: best[k] for k in ['accuracy', 'precision', 'recall', 'f1']}
}

save_path = '../models/xgb_urination_model.pkl'
joblib.dump(model_data, save_path)
print(f'💾 モデルを保存: {save_path}')"""),

# ── セル 9: 予測デモ ──────────────────────────────────────────
md("""## 9. 予測デモ

保存したモデルを使って、新しい入力から排尿時間を予測してみましょう。"""),
code("""\
from predict import load_model, predict

model_data_loaded = load_model()

# 試したい条件を自由に変えてみてください
test_cases = [
    {
        'name':           'コーヒーを大量に飲んだ成人男性',
        'age': 35, 'gender': 'M', 'weight': 75.0, 'height': 1.75, 'bmi': 24.5,
        'employment': 'full_time', 'level_exercise': 'medium',
        'alcoholic': 0, 'smoking': 0,
        'drink_type': 'coffee', 'volume': 500.0, 'ndrinks': 2,
        'vol_inp': 250.0, 'time_inp': 30.0,
    },
    {
        'name':           '水を少し飲んだ高齢女性',
        'age': 70, 'gender': 'F', 'weight': 60.0, 'height': 1.58, 'bmi': 24.0,
        'employment': 'retired', 'level_exercise': 'low',
        'alcoholic': 0, 'smoking': 0,
        'drink_type': 'water', 'volume': 200.0, 'ndrinks': 1,
        'vol_inp': 200.0, 'time_inp': 60.0,
    },
    {
        'name':           '牛乳を飲んだ子供',
        'age': 8, 'gender': 'F', 'weight': 25.0, 'height': 1.28, 'bmi': 15.3,
        'employment': 'student', 'level_exercise': 'high',
        'alcoholic': 0, 'smoking': 0,
        'drink_type': 'milk', 'volume': 200.0, 'ndrinks': 1,
        'vol_inp': 200.0, 'time_inp': 20.0,
    },
]

print('=' * 50)
for case in test_cases:
    name = case.pop('name')
    result = predict(case, model_data_loaded)
    print(f'\\n📋 {name}')
    print(f'  予測: Class {result["predicted_class"]} — {result["predicted_label"]}')
    print(f'  {result["recommendation"]}')
    print(f'  確率: ', end='')
    for label, prob in result['probabilities'].items():
        print(f'{label}={prob:.2%}', end='  ')
    print()
    case['name'] = name  # 元に戻す
print('\\n' + '=' * 50)"""),

# ── セル 10: まとめ ───────────────────────────────────────────
md("""## 10. まとめと次のステップ

### 📊 プロトタイプの結果

| 項目 | 内容 |
|------|------|
| モデル | XGBoost（4クラス分類） |
| 特徴量数 | 9個（chi-square または RF+RFE で選択） |
| 目標精度（論文） | Accuracy 0.70, F1 0.71 |

### 🚀 次のステップ

1. **📱 データ収集アプリ** — 実際のユーザーから飲水・排尿データを記録するフォームを作る
2. **🔄 独自データで再学習** — サンプルデータを実データに置き換えてモデルを再学習
3. **⚙️ ハイパーパラメータ最適化** — Optuna などを使って精度をさらに改善
4. **🌐 API 化** — FastAPI で予測エンドポイントを作り、スマートフォンアプリと連携

---
*Ali et al. (2022). Mitigating urinary incontinence condition using machine learning. BMC Medical Informatics and Decision Making, 22, 243.*"""),

]  # end nb.cells

if __name__ == '__main__':
    out = 'notebooks/01_prototype.ipynb'
    with open(f'../{out}', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'✅ ノートブックを生成: {out}')

"""
予測スクリプト
============================================================
学習済みモデルを使って、新しいデータから排尿時間を予測します。

使い方:
    python src/predict.py              # サンプル入力で予測
    python src/predict.py --json '{…}' # JSON形式で入力

将来的にはこのスクリプトをアプリのバックエンドAPIに組み込みます。
"""

import sys
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union


MODEL_PATH = Path(__file__).parent.parent / 'models' / 'xgb_urination_model.pkl'


# ================================================================
# モデルの読み込み
# ================================================================

def load_model(model_path: Path = MODEL_PATH) -> dict:
    """学習済みモデルデータを読み込みます"""
    if not model_path.exists():
        raise FileNotFoundError(
            f"モデルが見つかりません: {model_path}\n"
            "先に src/train.py を実行してモデルを学習してください。"
        )
    model_data = joblib.load(model_path)
    print(f"✅ モデル読み込み完了: {model_path.name}")
    return model_data


# ================================================================
# 入力データの前処理（1件分）
# ================================================================

def preprocess_input(raw_input: dict, model_data: dict) -> pd.DataFrame:
    """
    ユーザーの入力を受け取り、モデルが期待する形式に変換します。

    raw_input のキー例:
        age, gender, weight, height, bmi,
        employment, level_exercise, alcoholic, smoking,
        drink_type, volume, ndrinks, vol_inp, time_inp
    """
    df = pd.DataFrame([raw_input])

    # BMI が未入力なら計算
    if 'bmi' not in df.columns or pd.isna(df['bmi'].iloc[0]):
        if 'weight' in df.columns and 'height' in df.columns:
            df['bmi'] = df['weight'] / (df['height'] ** 2)

    # 飲み物ダミー変数を生成
    drink_types = ['water', 'coffee', 'juice', 'milk', 'soda', 'alcohol', 'tea']
    for dt in drink_types:
        df[dt] = int(raw_input.get('drink_type', '') == dt)

    # カテゴリ変数をエンコード
    encoders = model_data.get('encoders', {})
    for src, le in encoders.items():
        dst = f'{src}_enc'
        if src in df.columns:
            val = df[src].astype(str).iloc[0]
            # 未知ラベルが来た場合は最頻クラスを使用
            if val in le.classes_:
                df[dst] = le.transform([val])
            else:
                df[dst] = le.transform([le.classes_[0]])
                print(f"⚠️  未知の {src} 値 '{val}' → '{le.classes_[0]}' として処理")

    # モデルが必要とする特徴量だけ抽出
    feature_names = model_data['feature_names']
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0   # 欠損特徴量は 0 で補完

    X = df[feature_names]

    # スケーリング
    scaler = model_data.get('scaler')
    if scaler:
        X = pd.DataFrame(scaler.transform(X), columns=feature_names)

    return X


# ================================================================
# 予測の実行
# ================================================================

def predict(raw_input: dict, model_data: dict = None) -> dict:
    """
    1件分の入力データから排尿クラスを予測します。

    Returns:
        {
            'predicted_class': int,
            'predicted_label': str,   # '< 30分' など
            'probabilities':   dict,  # 各クラスの確率
            'recommendation':  str,   # ユーザー向けメッセージ
        }
    """
    if model_data is None:
        model_data = load_model()

    X = preprocess_input(raw_input, model_data)

    model  = model_data['model']
    labels = model_data['class_labels']

    pred_class = int(model.predict(X)[0])
    probs      = model.predict_proba(X)[0]
    prob_dict  = {labels[i]: round(float(p), 3) for i, p in enumerate(probs)}

    # ユーザー向けメッセージ
    messages = {
        0: "⚠️  まもなくトイレに行く必要があります（30分以内）",
        1: "🟡 30〜60分以内にトイレの準備をしましょう",
        2: "🟢 60〜90分は余裕があります",
        3: "✅ 90分以上余裕があります",
    }

    return {
        'predicted_class': pred_class,
        'predicted_label': labels[pred_class],
        'probabilities':   prob_dict,
        'recommendation':  messages[pred_class],
    }


# ================================================================
# CLI から実行
# ================================================================

# サンプル入力（動作確認用）
SAMPLE_INPUT = {
    'age':            35,
    'gender':         'M',
    'weight':         75.0,
    'height':         1.75,
    'bmi':            24.5,
    'employment':     'full_time',
    'level_exercise': 'medium',
    'alcoholic':      0,
    'smoking':        0,
    'drink_type':     'coffee',
    'volume':         400.0,
    'ndrinks':        2,
    'vol_inp':        200.0,
    'time_inp':       45.0,
}


def main():
    parser = argparse.ArgumentParser(description='排尿時間予測スクリプト')
    parser.add_argument('--json', type=str, default=None,
                        help='入力データをJSON文字列で渡す')
    parser.add_argument('--model', type=str, default=str(MODEL_PATH),
                        help='モデルファイルのパス')
    args = parser.parse_args()

    # モデル読み込み
    model_data = load_model(Path(args.model))

    # 入力データ
    if args.json:
        raw_input = json.loads(args.json)
    else:
        print("\n📝 サンプル入力データを使用します:")
        for k, v in SAMPLE_INPUT.items():
            print(f"   {k:20}: {v}")
        raw_input = SAMPLE_INPUT

    # 予測
    print("\n🔮 予測中...")
    result = predict(raw_input, model_data)

    # 結果を表示
    print("\n" + "=" * 45)
    print("📊 予測結果")
    print("=" * 45)
    print(f"  予測クラス : Class {result['predicted_class']} ({result['predicted_label']})")
    print(f"\n  各クラスの確率:")
    for label, prob in result['probabilities'].items():
        bar = '█' * int(prob * 30)
        print(f"    {label:>10}: {bar:<30} {prob:.1%}")
    print(f"\n  {result['recommendation']}")
    print("=" * 45)

    return result


if __name__ == '__main__':
    main()

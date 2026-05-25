"""
サンプルデータ生成スクリプト
============================================================
論文 Ali et al. (2022) の設定を参考に、合成データを生成します。
実際のデータ収集が完了したら、このスクリプトは不要になります。

生成するデータの特徴:
  - 51名の参加者 × 約17件/人 = 約850件
  - 年齢層: 子供・若者・成人・高齢者
  - 飲み物: 水・コーヒー・ジュース・牛乳・炭酸・アルコール・お茶
  - 目的変数: 次回排尿までの時間（4クラス）
    - Class 0: < 30分
    - Class 1: 31〜60分
    - Class 2: 61〜90分
    - Class 3: > 90分
"""

import numpy as np
import pandas as pd
from pathlib import Path

# 再現性のためシードを固定
np.random.seed(42)

N_PARTICIPANTS = 51         # 参加者数（論文と同じ）
MIN_EVENTS = 14             # 1参加者あたりの最小記録数
MAX_EVENTS = 20             # 1参加者あたりの最大記録数


# ================================================================
# 1. 参加者プロフィールの生成
# ================================================================

def generate_participant(pid: int) -> dict:
    """
    1人分の参加者情報を生成します。
    論文の人口統計（Table 1）を参考に年齢・性別・体格を設定。
    """
    # 年齢グループを論文の比率に近い形で割り当て
    age_group = np.random.choice(
        ['child', 'youth', 'adult', 'senior'],
        p=[0.14, 0.27, 0.55, 0.04]
    )

    if age_group == 'child':
        age    = np.random.randint(5, 12)
        gender = np.random.choice(['M', 'F'], p=[0.3, 0.7])
        weight = max(15.0, np.random.normal(25, 5))
        height = max(1.1,  np.random.normal(1.45, 0.1))

    elif age_group == 'youth':
        age    = np.random.randint(15, 25)
        # 論文では若い女性参加者が少ない
        gender = np.random.choice(['M', 'F'], p=[0.93, 0.07])
        weight = max(40.0, np.random.normal(83, 13))
        height = max(1.5,  np.random.normal(1.77, 0.1))

    elif age_group == 'adult':
        age    = np.random.randint(25, 65)
        gender = np.random.choice(['M', 'F'], p=[0.43, 0.57])
        weight = max(40.0, np.random.normal(77, 14))
        height = max(1.5,  np.random.normal(1.70, 0.08))

    else:  # senior
        age    = np.random.randint(65, 80)
        gender = np.random.choice(['M', 'F'])
        weight = max(40.0, np.random.normal(68, 13))
        height = max(1.4,  np.random.normal(1.62, 0.07))

    bmi = weight / (height ** 2)

    return {
        'participant_id':  pid,
        'age':             int(age),
        'age_group':       age_group,
        'gender':          gender,
        'weight':          round(weight, 1),
        'height':          round(height, 2),
        'bmi':             round(bmi, 1),
        # 生活習慣
        'alcoholic':       int(np.random.choice([0, 1], p=[0.70, 0.30])),
        'smoking':         int(np.random.choice([0, 1], p=[0.80, 0.20])),
        'employment':      np.random.choice(
                               ['unemployed', 'part_time', 'full_time', 'student', 'retired'],
                               p=[0.10, 0.15, 0.50, 0.15, 0.10]
                           ),
        'level_exercise':  np.random.choice(['low', 'medium', 'high'], p=[0.30, 0.50, 0.20]),
    }


# ================================================================
# 2. 飲水イベントの生成（1レコード = 1回の飲水＋その後の排尿）
# ================================================================

# 飲み物ごとの利尿作用（分：マイナスほど排尿が早くなる）
DIURETIC_EFFECT = {
    'water':   -10,
    'coffee':  -25,   # カフェインで利尿促進
    'juice':   -15,
    'milk':     -5,
    'soda':    -15,
    'alcohol': -30,   # アルコールで利尿促進
    'tea':     -20,
}

# 飲み物ごとの標準的な1杯の量（ml）
VOLUME_BASE = {
    'water':   350,
    'coffee':  200,
    'juice':   250,
    'milk':    200,
    'soda':    350,
    'alcohol': 350,
    'tea':     200,
}


def get_drink_probs(participant: dict) -> dict:
    """参加者の属性に応じた飲み物選択確率を返す"""
    if participant['age_group'] == 'child':
        return {'water': 0.50, 'juice': 0.30, 'milk': 0.15, 'soda': 0.05,
                'coffee': 0.00, 'alcohol': 0.00, 'tea': 0.00}
    elif participant['alcoholic']:
        return {'water': 0.28, 'juice': 0.10, 'milk': 0.05, 'soda': 0.10,
                'coffee': 0.17, 'alcohol': 0.25, 'tea': 0.05}
    else:
        return {'water': 0.45, 'juice': 0.15, 'milk': 0.10, 'soda': 0.10,
                'coffee': 0.10, 'alcohol': 0.05, 'tea': 0.05}


def compute_inter_release_time(participant: dict, drink_type: str,
                                total_volume: float) -> float:
    """
    排尿間隔（分）を計算します。
    各要因の影響を加算し、ランダムノイズを付与します。
    """
    base = 90.0  # 基準排尿間隔（分）

    # 飲み物の利尿作用
    t = base + DIURETIC_EFFECT[drink_type]

    # 飲量補正（300ml より多いほど排尿が早まる）
    t -= (total_volume - 300) * 0.05

    # 年齢補正
    age = participant['age']
    if age < 12:        # 子供は膀胱が小さい
        t -= 20
    elif 12 <= age < 18:
        t -= 10
    elif age >= 60:     # 高齢者は膀胱コントロールが低下
        t -= 15

    # BMI補正（肥満気味は膀胱に圧がかかりやすい）
    if participant['bmi'] >= 30:
        t -= 5

    # 運動量補正（運動後は発汗で排尿が遅くなる）
    if participant['level_exercise'] == 'high':
        t += 10
    elif participant['level_exercise'] == 'low':
        t -= 5

    # 喫煙者は膀胱刺激で排尿が早まる
    if participant['smoking']:
        t -= 5

    # ランダムノイズ
    t += np.random.normal(0, 18)

    # 10〜280分の範囲にクリップ
    return float(np.clip(t, 10, 280))


def time_to_class(minutes: float) -> int:
    """排尿間隔（分）を4クラスに変換"""
    if minutes < 30:
        return 0
    elif minutes <= 60:
        return 1
    elif minutes <= 90:
        return 2
    else:
        return 3


def generate_event(participant: dict) -> dict:
    """1回の飲水イベント（レコード）を生成"""
    probs_dict = get_drink_probs(participant)
    drink_types = list(probs_dict.keys())
    drink_type  = np.random.choice(drink_types, p=list(probs_dict.values()))

    # 1杯の量
    volume = max(50.0, np.random.normal(VOLUME_BASE[drink_type], 80))

    # 同一セッションで何杯飲むか
    ndrinks      = int(np.random.choice([1, 2, 3], p=[0.60, 0.30, 0.10]))
    total_volume = volume * ndrinks

    # 最初の飲量
    vol_inp = volume

    # 最初の排尿から次の飲水開始までの時間（分）
    time_inp = float(np.clip(np.random.normal(45, 20), 5, 200))

    # 目的変数
    inter_release_time = compute_inter_release_time(participant, drink_type, total_volume)
    target = time_to_class(inter_release_time)

    return {
        # 飲み物情報
        'drink_type':          drink_type,
        'volume':              round(total_volume, 1),
        'ndrinks':             ndrinks,
        'vol_inp':             round(vol_inp, 1),
        'time_inp':            round(time_inp, 1),
        # 飲み物ダミー変数
        'water':    int(drink_type == 'water'),
        'coffee':   int(drink_type == 'coffee'),
        'juice':    int(drink_type == 'juice'),
        'milk':     int(drink_type == 'milk'),
        'soda':     int(drink_type == 'soda'),
        'alcohol':  int(drink_type == 'alcohol'),
        'tea':      int(drink_type == 'tea'),
        # 目的変数
        'inter_release_time':  round(inter_release_time, 1),
        'target':              target,
    }


# ================================================================
# 3. データセット全体を生成
# ================================================================

def generate_dataset() -> pd.DataFrame:
    records = []
    for pid in range(1, N_PARTICIPANTS + 1):
        participant = generate_participant(pid)
        n_events    = np.random.randint(MIN_EVENTS, MAX_EVENTS + 1)

        for _ in range(n_events):
            event  = generate_event(participant)
            record = {**participant, **event}
            records.append(record)

    return pd.DataFrame(records)


# ================================================================
# 4. メイン実行
# ================================================================

if __name__ == '__main__':
    output_dir  = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'sample_data.csv'

    print("🔄 サンプルデータを生成中...")
    df = generate_dataset()
    df.to_csv(output_path, index=False)

    print(f"✅ 保存完了: {output_path}")
    print(f"   レコード数 : {len(df):,} 件")
    print(f"   参加者数   : {df['participant_id'].nunique()} 名")
    print(f"   特徴量数   : {df.shape[1]} 列")
    print()
    print("📊 目的変数の分布（クラス別件数）:")
    class_labels = {0: '< 30分', 1: '31〜60分', 2: '61〜90分', 3: '> 90分'}
    for cls, cnt in df['target'].value_counts().sort_index().items():
        pct = cnt / len(df) * 100
        print(f"   Class {cls} ({class_labels[cls]:>8}): {cnt:>3} 件 ({pct:.1f}%)")
    print()
    print("👥 参加者の年齢・性別内訳:")
    print(df.drop_duplicates('participant_id')[['age_group', 'gender']]
            .value_counts().sort_index().to_string())

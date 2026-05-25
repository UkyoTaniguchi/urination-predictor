"""
FastAPI バックエンド
============================================================
エンドポイント:
  POST /predict   — フォームデータを受け取り排尿クラスを予測
  POST /record    — 実測データをCSVに保存（データ収集用）
  GET  /health    — ヘルスチェック
"""

import sys
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# src/ のモジュールを参照できるようにパスを通す
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from predict import load_model, predict as ml_predict

# ================================================================
# アプリ初期化
# ================================================================

app = FastAPI(
    title="排尿予測 API",
    description="飲水情報・個人情報から次回排尿時間を予測します",
    version="0.1.0",
)

# CORS 設定（React 開発サーバー localhost:5173 を許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルを起動時に1回だけ読み込む
MODEL_PATH = ROOT / 'models' / 'xgb_urination_model.pkl'
DATA_LOG   = ROOT / 'data' / 'collected_data.csv'

_model_data = None

def get_model():
    global _model_data
    if _model_data is None:
        _model_data = load_model(MODEL_PATH)
    return _model_data


# ================================================================
# リクエスト／レスポンスのスキーマ
# ================================================================

class DrinkInput(BaseModel):
    """フォームから受け取る入力データ"""

    # 飲み物情報
    drink_type: str = Field(..., description="飲み物の種類",
                            pattern="^(water|coffee|juice|milk|soda|alcohol|tea)$")
    volume:     float = Field(..., gt=0, le=3000, description="合計飲料量（ml）")
    ndrinks:    int   = Field(..., ge=1, le=10,   description="杯数")
    vol_inp:    float = Field(..., gt=0, le=3000, description="1杯目の量（ml）")
    time_inp:   float = Field(..., ge=0, le=300,  description="前回排尿からの経過時間（分）")

    # 身体情報
    age:    int   = Field(..., ge=1,  le=120, description="年齢")
    gender: str   = Field(..., description="性別", pattern="^(M|F)$")
    weight: float = Field(..., gt=0,  le=300, description="体重（kg）")
    height: float = Field(..., gt=0,  le=3.0, description="身長（m）")

    # 生活習慣
    employment:     str = Field(..., description="雇用形態",
                                pattern="^(unemployed|part_time|full_time|student|retired)$")
    level_exercise: str = Field(..., description="運動レベル",
                                pattern="^(low|medium|high)$")
    alcoholic: int = Field(..., ge=0, le=1, description="常習的飲酒（0/1）")
    smoking:   int = Field(..., ge=0, le=1, description="喫煙（0/1）")

    # 実測値（データ収集時のみ使用、省略可）
    actual_minutes: Optional[float] = Field(None, ge=0, le=600,
                                            description="実際の排尿間隔（分）—収集用")

    @validator('height')
    def height_in_meters(cls, v):
        # 身長が 3 以下なら m 単位、100 超なら cm として m に変換
        if v > 3:
            return v / 100
        return v

    @property
    def bmi(self) -> float:
        return round(self.weight / (self.height ** 2), 1)


class PredictionResponse(BaseModel):
    """予測結果レスポンス"""
    predicted_class: int
    predicted_label: str
    probabilities:   dict
    recommendation:  str
    bmi:             float


class RecordResponse(BaseModel):
    """データ保存レスポンス"""
    saved: bool
    message: str
    total_records: int


# ================================================================
# エンドポイント
# ================================================================

@app.get("/health")
def health_check():
    model_loaded = MODEL_PATH.exists()
    return {
        "status":       "ok",
        "model_ready":  model_loaded,
        "timestamp":    datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(data: DrinkInput):
    """
    フォームデータを受け取り、排尿クラスを予測します。
    """
    try:
        model_data = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"モデル未ロード: {e}")

    # DrinkInput → dict（ml_predict が期待する形式）
    raw_input = {
        'age':            data.age,
        'gender':         data.gender,
        'weight':         data.weight,
        'height':         data.height,
        'bmi':            data.bmi,
        'employment':     data.employment,
        'level_exercise': data.level_exercise,
        'alcoholic':      data.alcoholic,
        'smoking':        data.smoking,
        'drink_type':     data.drink_type,
        'volume':         data.volume,
        'ndrinks':        data.ndrinks,
        'vol_inp':        data.vol_inp,
        'time_inp':       data.time_inp,
    }

    result = ml_predict(raw_input, model_data)
    return PredictionResponse(
        predicted_class=result['predicted_class'],
        predicted_label=result['predicted_label'],
        probabilities=result['probabilities'],
        recommendation=result['recommendation'],
        bmi=data.bmi,
    )


@app.post("/record", response_model=RecordResponse)
def record_endpoint(data: DrinkInput):
    """
    実測データをCSVに保存します（データ収集用）。
    actual_minutes が設定されている場合のみ保存します。
    """
    if data.actual_minutes is None:
        raise HTTPException(
            status_code=400,
            detail="実測値 (actual_minutes) を入力してください"
        )

    # 保存するデータを準備
    row = {
        'timestamp':        datetime.now().isoformat(),
        'age':              data.age,
        'gender':           data.gender,
        'weight':           data.weight,
        'height':           data.height,
        'bmi':              data.bmi,
        'employment':       data.employment,
        'level_exercise':   data.level_exercise,
        'alcoholic':        data.alcoholic,
        'smoking':          data.smoking,
        'drink_type':       data.drink_type,
        'volume':           data.volume,
        'ndrinks':          data.ndrinks,
        'vol_inp':          data.vol_inp,
        'time_inp':         data.time_inp,
        'actual_minutes':   data.actual_minutes,
        # 4クラスラベルを自動付与
        'target': (
            0 if data.actual_minutes < 30
            else 1 if data.actual_minutes <= 60
            else 2 if data.actual_minutes <= 90
            else 3
        ),
    }

    # CSV に追記
    DATA_LOG.parent.mkdir(exist_ok=True)
    file_exists = DATA_LOG.exists()
    with open(DATA_LOG, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # 総件数を取得
    total = sum(1 for _ in open(DATA_LOG, encoding='utf-8')) - 1  # ヘッダ除く

    return RecordResponse(
        saved=True,
        message=f"データを保存しました（通算 {total} 件目）",
        total_records=total,
    )

import { useState } from 'react'

/**
 * 予測結果の表示 + データ記録フォーム
 */
export default function PredictionResult({ result, onReset }) {
  const [actualMinutes, setActualMinutes] = useState('')
  const [saving, setSaving]               = useState(false)
  const [saveMsg, setSaveMsg]             = useState(null)

  const { predicted_label, recommendation, probabilities, bmi, formData } = result

  // 最大確率のクラスを強調
  const maxProb = Math.max(...Object.values(probabilities))

  // 実測データを保存
  const handleSave = async () => {
    if (!actualMinutes || isNaN(parseFloat(actualMinutes))) {
      setSaveMsg({ type: 'error', text: '実際の時間（分）を入力してください' })
      return
    }
    setSaving(true)
    setSaveMsg(null)

    try {
      const payload = {
        ...formData,
        actual_minutes: parseFloat(actualMinutes),
      }
      const res = await fetch('/api/record', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || '保存に失敗しました')
      }
      const data = await res.json()
      setSaveMsg({ type: 'success', text: data.message })
    } catch (err) {
      setSaveMsg({ type: 'error', text: err.message })
    } finally {
      setSaving(false)
    }
  }

  return (
    <>
      {/* ── 予測結果カード ── */}
      <div className="result-card">
        <div className="result-label">
          <div className="class-badge">🕐 {predicted_label}</div>
          <p className="recommendation">{recommendation}</p>
          {bmi && (
            <p style={{ fontSize: '0.82rem', color: '#718096', marginTop: 8 }}>
              BMI: {bmi} kg/m²
            </p>
          )}
        </div>

        {/* 確率バー */}
        <div className="prob-bars">
          {Object.entries(probabilities).map(([label, prob]) => {
            const isTop = prob === maxProb
            return (
              <div className="prob-row" key={label}>
                <span className="label">{label}</span>
                <div className="prob-bar-wrap">
                  <div
                    className={`prob-bar-fill ${isTop ? 'top' : ''}`}
                    style={{ width: `${(prob * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="pct">{(prob * 100).toFixed(1)}%</span>
              </div>
            )
          })}
        </div>

        {/* ── データ収集セクション ── */}
        <div className="record-section">
          <p className="record-hint">
            📝 実際に排尿するまでの時間を記録してください。<br />
            データが増えるほどモデルの精度が向上します。
          </p>

          {saveMsg && (
            <div className={`alert alert-${saveMsg.type}`}>
              {saveMsg.type === 'success' ? '✅' : '⚠️'} {saveMsg.text}
            </div>
          )}

          <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            <div className="form-group" style={{ flex: 1 }}>
              <label>実際に排尿するまでの時間 <span className="unit">分</span></label>
              <input
                type="number" min="1" max="600" step="1"
                placeholder="例: 52"
                value={actualMinutes}
                onChange={e => setActualMinutes(e.target.value)}
                disabled={saveMsg?.type === 'success'}
              />
            </div>
            <button
              type="button"
              className="btn-secondary"
              style={{ marginTop: 22, whiteSpace: 'nowrap' }}
              onClick={handleSave}
              disabled={saving || saveMsg?.type === 'success'}
            >
              {saving ? '保存中…' : '💾 記録する'}
            </button>
          </div>
        </div>
      </div>

      {/* ── もう一度ボタン ── */}
      <button
        type="button"
        className="btn-submit"
        style={{ background: 'linear-gradient(135deg, #38a169, #276749)' }}
        onClick={onReset}
      >
        ➕　もう一度入力する
      </button>
    </>
  )
}

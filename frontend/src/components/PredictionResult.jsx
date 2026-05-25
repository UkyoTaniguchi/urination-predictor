import { useState } from 'react'

// クラスごとの配色
const CLASS_COLORS = {
  0: { bg: 'bg-red-50',    border: 'border-red-300',   badge: 'bg-red-600',    bar: 'bg-red-500'    },
  1: { bg: 'bg-amber-50',  border: 'border-amber-300', badge: 'bg-amber-500',  bar: 'bg-amber-400'  },
  2: { bg: 'bg-green-50',  border: 'border-green-300', badge: 'bg-green-600',  bar: 'bg-green-500'  },
  3: { bg: 'bg-blue-50',   border: 'border-blue-300',  badge: 'bg-blue-600',   bar: 'bg-blue-500'   },
}

const inputCls =
  'w-full rounded-lg border border-slate-300 px-3 py-2.5 text-sm ' +
  'focus:border-blue-400 focus:ring-2 focus:ring-blue-200 outline-none transition'

export default function PredictionResult({ result, onReset }) {
  const [actualMinutes, setActualMinutes] = useState('')
  const [saving, setSaving]               = useState(false)
  const [saveMsg, setSaveMsg]             = useState(null)

  const { predicted_class, predicted_label, recommendation, probabilities, bmi, formData } = result
  const colors  = CLASS_COLORS[predicted_class] ?? CLASS_COLORS[2]
  const maxProb = Math.max(...Object.values(probabilities))

  const handleSave = async () => {
    const mins = parseFloat(actualMinutes)
    if (isNaN(mins) || mins <= 0) {
      setSaveMsg({ type: 'error', text: '実際の時間（分）を入力してください' })
      return
    }
    setSaving(true)
    setSaveMsg(null)
    try {
      const res = await fetch('/api/record', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...formData, actual_minutes: mins }),
      })
      if (!res.ok) throw new Error((await res.json()).detail || '保存失敗')
      setSaveMsg({ type: 'success', text: (await res.json()).message })
    } catch (err) {
      setSaveMsg({ type: 'error', text: err.message })
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="space-y-5 animate-[fadeIn_0.4s_ease]"
      style={{ ['--tw-animate-duration']: '0.4s' }}>

      {/* ── 予測結果カード ── */}
      <div className={`rounded-2xl border-2 ${colors.bg} ${colors.border} p-6 shadow-sm`}>

        {/* バッジ + メッセージ */}
        <div className="text-center mb-6">
          <span className={`inline-block rounded-full ${colors.badge}
                            px-8 py-2.5 text-xl font-bold text-white mb-3`}>
            🕐 {predicted_label}
          </span>
          <p className="text-sm font-semibold text-slate-700">{recommendation}</p>
          {bmi && (
            <p className="text-xs text-slate-400 mt-1">BMI: {bmi} kg/m²</p>
          )}
        </div>

        {/* 確率バー */}
        <div className="space-y-2.5">
          <p className="text-xs font-semibold text-slate-500 mb-1">各クラスの確率</p>
          {Object.entries(probabilities).map(([label, prob]) => {
            const isTop = prob === maxProb
            return (
              <div key={label} className="flex items-center gap-3">
                <span className="w-20 text-right text-xs text-slate-500 shrink-0">
                  {label}
                </span>
                <div className="flex-1 rounded-full bg-slate-200 h-4 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700
                                ${isTop ? colors.bar : 'bg-slate-400'}`}
                    style={{ width: `${(prob * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className={`w-11 text-xs font-bold shrink-0
                                  ${isTop ? 'text-blue-700' : 'text-slate-500'}`}>
                  {(prob * 100).toFixed(1)}%
                </span>
              </div>
            )
          })}
        </div>

        {/* ── データ記録セクション ── */}
        <div className="mt-6 pt-5 border-t border-dashed border-slate-300">
          <p className="text-xs text-slate-500 mb-4 leading-relaxed">
            📝 実際に排尿するまでの時間を記録してください。<br />
            データが増えるほどモデルの精度が向上します。
          </p>

          {saveMsg && (
            <div className={[
              'rounded-xl border px-4 py-3 text-sm mb-3',
              saveMsg.type === 'success'
                ? 'bg-green-50 border-green-200 text-green-700'
                : 'bg-red-50  border-red-200  text-red-700',
            ].join(' ')}>
              {saveMsg.type === 'success' ? '✅' : '⚠️'} {saveMsg.text}
            </div>
          )}

          <div className="flex gap-3 items-end">
            <div className="flex-1">
              <label className="block text-xs font-semibold text-slate-600 mb-1.5">
                実際に排尿するまでの時間{' '}
                <span className="font-normal text-slate-400">分</span>
              </label>
              <input
                type="number" min="1" max="600" step="1"
                placeholder="例: 52"
                value={actualMinutes}
                onChange={e => setActualMinutes(e.target.value)}
                disabled={saveMsg?.type === 'success'}
                className={inputCls}
              />
            </div>
            <button
              type="button"
              onClick={handleSave}
              disabled={saving || saveMsg?.type === 'success'}
              className="rounded-xl bg-slate-100 px-5 py-2.5 text-sm font-semibold
                         text-slate-700 hover:bg-slate-200 active:scale-[0.98]
                         transition disabled:opacity-50 disabled:cursor-not-allowed
                         whitespace-nowrap">
              {saving ? '保存中…' : '💾 記録する'}
            </button>
          </div>
        </div>
      </div>

      {/* ── もう一度ボタン ── */}
      <button
        type="button"
        onClick={onReset}
        className="w-full rounded-xl bg-gradient-to-r from-emerald-600 to-emerald-800
                   py-3.5 text-base font-bold text-white shadow-md
                   hover:opacity-90 active:scale-[0.99] transition">
        ➕　もう一度入力する
      </button>
    </div>
  )
}

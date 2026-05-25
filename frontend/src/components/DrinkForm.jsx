import { useState, useMemo } from 'react'

// ── 定数 ──────────────────────────────────────────────────────────
const DRINK_OPTIONS = [
  { value: 'water',   emoji: '💧', label: '水' },
  { value: 'coffee',  emoji: '☕', label: 'コーヒー' },
  { value: 'tea',     emoji: '🍵', label: 'お茶' },
  { value: 'juice',   emoji: '🍊', label: 'ジュース' },
  { value: 'milk',    emoji: '🥛', label: '牛乳' },
  { value: 'soda',    emoji: '🥤', label: '炭酸' },
  { value: 'alcohol', emoji: '🍺', label: 'お酒' },
]

const EMPLOYMENT_OPTIONS = [
  { value: 'full_time',  label: '正社員 / フルタイム' },
  { value: 'part_time',  label: 'パートタイム' },
  { value: 'student',    label: '学生' },
  { value: 'unemployed', label: '無職' },
  { value: 'retired',    label: '退職 / 年金' },
]

const EXERCISE_OPTIONS = [
  { value: 'low',    label: 'ほとんどしない' },
  { value: 'medium', label: '週 1〜3 回程度' },
  { value: 'high',   label: '週 4 回以上' },
]

const INITIAL = {
  drink_type: 'water', volume: '', ndrinks: 1,
  vol_inp: '', time_inp: '',
  age: '', gender: 'M', weight: '', height: '',
  employment: 'full_time', level_exercise: 'medium',
  alcoholic: false, smoking: false,
}

// ── 共通スタイル ──────────────────────────────────────────────────
const inputCls =
  'w-full rounded-lg border border-slate-300 px-3 py-2.5 text-sm ' +
  'focus:border-blue-400 focus:ring-2 focus:ring-blue-200 outline-none transition'

const labelCls = 'block text-xs font-semibold text-slate-600 mb-1.5'

// ── コンポーネント ────────────────────────────────────────────────
export default function DrinkForm({ onResult }) {
  const [form, setForm]       = useState(INITIAL)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  // BMI 自動計算
  const bmi = useMemo(() => {
    const w = parseFloat(form.weight)
    let h   = parseFloat(form.height)
    if (!w || !h) return null
    if (h > 3) h /= 100          // cm → m
    const v = w / (h * h)
    return isNaN(v) ? null : v.toFixed(1)
  }, [form.weight, form.height])

  // 飲み物を選んだとき
  const handleDrinkType = (value) =>
    setForm(f => ({ ...f, drink_type: value }))

  // 合計量変更 → 杯数が 1 なら vol_inp も連動
  const handleVolume = (e) => {
    const v = e.target.value
    setForm(f => ({ ...f, volume: v, vol_inp: f.ndrinks === 1 ? v : f.vol_inp }))
  }

  // 杯数変更 → 1杯あたりの量を再計算
  const handleNdrinks = (e) => {
    const n = parseInt(e.target.value, 10)
    const perCup = form.volume ? Math.round(parseFloat(form.volume) / n) : ''
    setForm(f => ({ ...f, ndrinks: n, vol_inp: String(perCup) }))
  }

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target
    setForm(f => ({ ...f, [name]: type === 'checkbox' ? checked : value }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setLoading(true)

    try {
      const payload = {
        drink_type:     form.drink_type,
        volume:         parseFloat(form.volume),
        ndrinks:        parseInt(form.ndrinks, 10),
        vol_inp:        parseFloat(form.vol_inp || form.volume),
        time_inp:       parseFloat(form.time_inp),
        age:            parseInt(form.age, 10),
        gender:         form.gender,
        weight:         parseFloat(form.weight),
        height:         parseFloat(form.height),
        employment:     form.employment,
        level_exercise: form.level_exercise,
        alcoholic:      form.alcoholic ? 1 : 0,
        smoking:        form.smoking   ? 1 : 0,
      }

      // 必須項目チェック
      for (const key of ['volume', 'time_inp', 'age', 'weight', 'height']) {
        if (isNaN(payload[key]) || payload[key] <= 0)
          throw new Error(`「${key}」を正しく入力してください`)
      }

      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      })
      if (!res.ok) throw new Error((await res.json()).detail || '予測に失敗しました')

      onResult({ ...(await res.json()), formData: payload })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5">

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          ⚠️ {error}
        </div>
      )}

      {/* ── 飲んだもの ── */}
      <section className="rounded-2xl bg-white p-6 shadow-sm">
        <h2 className="flex items-center gap-2 text-base font-bold text-blue-700
                       border-b-2 border-blue-100 pb-3 mb-5">
          🥤 飲んだもの
        </h2>

        {/* 飲み物セレクター */}
        <div className="mb-5">
          <p className={labelCls}>飲み物の種類</p>
          <div className="grid grid-cols-4 gap-2.5 sm:grid-cols-7">
            {DRINK_OPTIONS.map(opt => (
              <button
                key={opt.value}
                type="button"
                onClick={() => handleDrinkType(opt.value)}
                className={[
                  'flex flex-col items-center gap-1 rounded-xl border-2 py-2.5 px-1',
                  'text-xs font-medium transition-all duration-150 cursor-pointer',
                  form.drink_type === opt.value
                    ? 'border-blue-500 bg-blue-50 text-blue-700 font-bold'
                    : 'border-slate-200 bg-white text-slate-600 hover:border-blue-300 hover:bg-blue-50',
                ].join(' ')}
              >
                <span className="text-2xl">{opt.emoji}</span>
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        {/* 量・杯数・経過時間 */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className={labelCls}>
              合計飲料量 <span className="font-normal text-slate-400">ml</span>
            </label>
            <input type="number" min="1" max="3000" step="10"
              placeholder="例: 350"
              value={form.volume} onChange={handleVolume}
              className={inputCls} />
          </div>

          <div>
            <label className={labelCls}>杯数</label>
            <select name="ndrinks" value={form.ndrinks}
              onChange={handleNdrinks} className={inputCls}>
              {[1,2,3,4,5].map(n => (
                <option key={n} value={n}>{n} 杯</option>
              ))}
            </select>
          </div>

          <div>
            <label className={labelCls}>
              1 杯目の量 <span className="font-normal text-slate-400">ml</span>
            </label>
            <input type="number" name="vol_inp" min="1" max="1000" step="10"
              placeholder="例: 350"
              value={form.vol_inp} onChange={handleChange}
              className={inputCls} />
          </div>

          <div>
            <label className={labelCls}>
              前回排尿からの経過時間 <span className="font-normal text-slate-400">分</span>
            </label>
            <input type="number" name="time_inp" min="0" max="300" step="1"
              placeholder="例: 45"
              value={form.time_inp} onChange={handleChange}
              className={inputCls} />
          </div>
        </div>
      </section>

      {/* ── 身体情報 ── */}
      <section className="rounded-2xl bg-white p-6 shadow-sm">
        <h2 className="flex items-center gap-2 text-base font-bold text-blue-700
                       border-b-2 border-blue-100 pb-3 mb-5">
          👤 身体情報
        </h2>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className={labelCls}>
              年齢 <span className="font-normal text-slate-400">歳</span>
            </label>
            <input type="number" name="age" min="1" max="120"
              placeholder="例: 35"
              value={form.age} onChange={handleChange}
              className={inputCls} />
          </div>

          <div>
            <label className={labelCls}>性別</label>
            <select name="gender" value={form.gender}
              onChange={handleChange} className={inputCls}>
              <option value="M">男性</option>
              <option value="F">女性</option>
            </select>
          </div>

          <div>
            <label className={labelCls}>
              体重 <span className="font-normal text-slate-400">kg</span>
            </label>
            <input type="number" name="weight" min="1" max="300" step="0.1"
              placeholder="例: 65.0"
              value={form.weight} onChange={handleChange}
              className={inputCls} />
          </div>

          <div>
            <label className={labelCls}>
              身長 <span className="font-normal text-slate-400">cm または m</span>
            </label>
            <input type="number" name="height" min="50" max="250" step="0.1"
              placeholder="例: 170"
              value={form.height} onChange={handleChange}
              className={inputCls} />
          </div>

          {/* BMI 自動計算 */}
          <div className="col-span-2">
            <label className={labelCls}>
              BMI <span className="font-normal text-slate-400">自動計算</span>
            </label>
            <div className="rounded-lg border border-blue-200 bg-blue-50
                            px-3 py-2.5 text-sm font-semibold text-blue-700">
              {bmi ? `${bmi} kg/m²` : '— （体重・身長を入力してください）'}
            </div>
          </div>
        </div>
      </section>

      {/* ── 生活習慣 ── */}
      <section className="rounded-2xl bg-white p-6 shadow-sm">
        <h2 className="flex items-center gap-2 text-base font-bold text-blue-700
                       border-b-2 border-blue-100 pb-3 mb-5">
          🏃 生活習慣
        </h2>

        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className={labelCls}>雇用形態</label>
            <select name="employment" value={form.employment}
              onChange={handleChange} className={inputCls}>
              {EMPLOYMENT_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>

          <div>
            <label className={labelCls}>運動習慣</label>
            <select name="level_exercise" value={form.level_exercise}
              onChange={handleChange} className={inputCls}>
              {EXERCISE_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
        </div>

        {/* チェックボックス */}
        <div>
          <label className={labelCls}>その他</label>
          <div className="flex gap-6">
            {[
              { name: 'alcoholic', label: '習慣的にお酒を飲む' },
              { name: 'smoking',   label: '喫煙者' },
            ].map(({ name, label }) => (
              <label key={name}
                className="flex cursor-pointer items-center gap-2 text-sm text-slate-700">
                <input type="checkbox" name={name}
                  checked={form[name]} onChange={handleChange}
                  className="h-4 w-4 cursor-pointer accent-blue-600" />
                {label}
              </label>
            ))}
          </div>
        </div>
      </section>

      {/* ── 送信ボタン ── */}
      <button type="submit" disabled={loading}
        className="w-full rounded-xl bg-gradient-to-r from-blue-600 to-blue-800
                   py-3.5 text-base font-bold text-white shadow-md
                   hover:opacity-90 active:scale-[0.99] transition
                   disabled:opacity-50 disabled:cursor-not-allowed">
        {loading
          ? <span className="flex items-center justify-center gap-2">
              <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10"
                  stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor"
                  d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              予測中…
            </span>
          : '🔮　排尿時間を予測する'}
      </button>
    </form>
  )
}

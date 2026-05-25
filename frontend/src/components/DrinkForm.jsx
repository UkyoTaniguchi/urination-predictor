import { useState, useMemo } from 'react'

// 飲み物の選択肢（絵文字 + ラベル + value）
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
  { value: 'full_time',  label: '正社員/フルタイム' },
  { value: 'part_time',  label: 'パートタイム' },
  { value: 'student',    label: '学生' },
  { value: 'unemployed', label: '無職' },
  { value: 'retired',    label: '退職/年金' },
]

const EXERCISE_OPTIONS = [
  { value: 'low',    label: 'ほとんどしない' },
  { value: 'medium', label: '週1〜3回程度' },
  { value: 'high',   label: '週4回以上' },
]

/** 初期値 */
const INITIAL_FORM = {
  drink_type:     'water',
  volume:         '',
  ndrinks:        1,
  vol_inp:        '',
  time_inp:       '',
  age:            '',
  gender:         'M',
  weight:         '',
  height:         '',
  employment:     'full_time',
  level_exercise: 'medium',
  alcoholic:      false,
  smoking:        false,
}

export default function DrinkForm({ onResult }) {
  const [form, setForm]       = useState(INITIAL_FORM)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  // BMI を自動計算
  const bmi = useMemo(() => {
    const w = parseFloat(form.weight)
    let h = parseFloat(form.height)
    if (!w || !h || h <= 0) return null
    if (h > 3) h = h / 100   // cm → m
    const val = w / (h * h)
    return isNaN(val) ? null : val.toFixed(1)
  }, [form.weight, form.height])

  // 飲み物を選ぶと 1杯目の量を合計量と同じにセット
  const handleDrinkType = (value) => {
    setForm(f => ({
      ...f,
      drink_type: value,
      vol_inp: f.volume || f.vol_inp,
    }))
  }

  // 合計量が変わったとき、杯数が1なら vol_inp も連動
  const handleVolume = (e) => {
    const v = e.target.value
    setForm(f => ({
      ...f,
      volume: v,
      vol_inp: f.ndrinks === 1 ? v : f.vol_inp,
    }))
  }

  // 杯数が変わったとき vol_inp を再計算
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

      // バリデーション
      const required = ['volume', 'time_inp', 'age', 'weight', 'height']
      for (const key of required) {
        if (isNaN(payload[key]) || payload[key] <= 0) {
          throw new Error(`「${key}」の値を正しく入力してください`)
        }
      }

      const res = await fetch('/api/predict', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || '予測に失敗しました')
      }

      const data = await res.json()
      onResult({ ...data, formData: payload })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      {error && <div className="alert alert-error">⚠️ {error}</div>}

      {/* ── 飲み物 ── */}
      <div className="card">
        <div className="card-title">🥤 飲んだもの</div>

        {/* 飲み物の種類 */}
        <div className="form-group" style={{ marginBottom: 18 }}>
          <label>飲み物の種類</label>
          <div className="drink-grid">
            {DRINK_OPTIONS.map(opt => (
              <button
                key={opt.value}
                type="button"
                className={`drink-btn ${form.drink_type === opt.value ? 'active' : ''}`}
                onClick={() => handleDrinkType(opt.value)}
              >
                <span className="emoji">{opt.emoji}</span>
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        <div className="form-grid">
          {/* 合計量 */}
          <div className="form-group">
            <label>合計飲料量 <span className="unit">ml</span></label>
            <input
              type="number" name="volume" min="1" max="3000" step="10"
              placeholder="例: 350"
              value={form.volume}
              onChange={handleVolume}
            />
          </div>

          {/* 杯数 */}
          <div className="form-group">
            <label>杯数</label>
            <select name="ndrinks" value={form.ndrinks} onChange={handleNdrinks}>
              {[1,2,3,4,5].map(n => (
                <option key={n} value={n}>{n}杯</option>
              ))}
            </select>
          </div>

          {/* 1杯目の量 */}
          <div className="form-group">
            <label>1杯目の量 <span className="unit">ml</span></label>
            <input
              type="number" name="vol_inp" min="1" max="1000" step="10"
              placeholder="例: 350"
              value={form.vol_inp}
              onChange={handleChange}
            />
          </div>

          {/* 前回排尿からの経過時間 */}
          <div className="form-group">
            <label>前回排尿からの経過時間 <span className="unit">分</span></label>
            <input
              type="number" name="time_inp" min="0" max="300" step="1"
              placeholder="例: 45"
              value={form.time_inp}
              onChange={handleChange}
            />
          </div>
        </div>
      </div>

      {/* ── 身体情報 ── */}
      <div className="card">
        <div className="card-title">👤 身体情報</div>
        <div className="form-grid">
          {/* 年齢 */}
          <div className="form-group">
            <label>年齢 <span className="unit">歳</span></label>
            <input
              type="number" name="age" min="1" max="120"
              placeholder="例: 35"
              value={form.age}
              onChange={handleChange}
            />
          </div>

          {/* 性別 */}
          <div className="form-group">
            <label>性別</label>
            <select name="gender" value={form.gender} onChange={handleChange}>
              <option value="M">男性</option>
              <option value="F">女性</option>
            </select>
          </div>

          {/* 体重 */}
          <div className="form-group">
            <label>体重 <span className="unit">kg</span></label>
            <input
              type="number" name="weight" min="1" max="300" step="0.1"
              placeholder="例: 65.0"
              value={form.weight}
              onChange={handleChange}
            />
          </div>

          {/* 身長 */}
          <div className="form-group">
            <label>身長 <span className="unit">cm または m</span></label>
            <input
              type="number" name="height" min="50" max="250" step="0.1"
              placeholder="例: 170"
              value={form.height}
              onChange={handleChange}
            />
          </div>

          {/* BMI（自動計算） */}
          <div className="form-group">
            <label>BMI <span className="unit">自動計算</span></label>
            <div className="bmi-display">
              {bmi ? `${bmi} kg/m²` : '— （体重・身長を入力してください）'}
            </div>
          </div>
        </div>
      </div>

      {/* ── 生活習慣 ── */}
      <div className="card">
        <div className="card-title">🏃 生活習慣</div>
        <div className="form-grid">
          {/* 雇用形態 */}
          <div className="form-group">
            <label>雇用形態</label>
            <select name="employment" value={form.employment} onChange={handleChange}>
              {EMPLOYMENT_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>

          {/* 運動レベル */}
          <div className="form-group">
            <label>運動習慣</label>
            <select name="level_exercise" value={form.level_exercise} onChange={handleChange}>
              {EXERCISE_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
        </div>

        {/* チェックボックス */}
        <div className="form-group" style={{ marginTop: 16 }}>
          <label>その他</label>
          <div className="checkbox-group">
            <label className="checkbox-item">
              <input
                type="checkbox" name="alcoholic"
                checked={form.alcoholic}
                onChange={handleChange}
              />
              習慣的にお酒を飲む
            </label>
            <label className="checkbox-item">
              <input
                type="checkbox" name="smoking"
                checked={form.smoking}
                onChange={handleChange}
              />
              喫煙者
            </label>
          </div>
        </div>
      </div>

      {/* ── 送信ボタン ── */}
      <button type="submit" className="btn-submit" disabled={loading}>
        {loading && <span className="spinner" />}
        {loading ? '予測中...' : '🔮　排尿時間を予測する'}
      </button>
    </form>
  )
}

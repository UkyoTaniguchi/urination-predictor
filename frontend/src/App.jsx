import { useState } from 'react'
import './index.css'
import DrinkForm from './components/DrinkForm'
import PredictionResult from './components/PredictionResult'

export default function App() {
  const [result, setResult] = useState(null)

  return (
    <div className="min-h-screen bg-slate-100 py-8 px-4">
      <div className="max-w-2xl mx-auto">

        {/* ヘッダー */}
        <header className="text-center mb-8">
          <h1 className="text-3xl font-bold text-blue-700 mb-2">🚽 排尿予測アプリ</h1>
          <p className="text-slate-500 text-sm">
            飲んだもの・身体情報を入力すると、次回トイレまでの時間を予測します
          </p>
        </header>

        {result ? (
          <PredictionResult result={result} onReset={() => setResult(null)} />
        ) : (
          <DrinkForm onResult={setResult} />
        )}

      </div>
    </div>
  )
}

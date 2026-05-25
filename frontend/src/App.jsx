import { useState } from 'react'
import './index.css'
import DrinkForm from './components/DrinkForm'
import PredictionResult from './components/PredictionResult'

export default function App() {
  const [result, setResult] = useState(null)

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🚽 排尿予測アプリ</h1>
        <p>飲んだもの・身体情報を入力すると、次回トイレに行くまでの時間を予測します</p>
      </header>

      {result ? (
        <PredictionResult
          result={result}
          onReset={() => setResult(null)}
        />
      ) : (
        <DrinkForm onResult={setResult} />
      )}
    </div>
  )
}

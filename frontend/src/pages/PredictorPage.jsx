import { useState } from 'react';

export default function PredictorPage() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/predict');
      if (!res.ok) throw new Error(`API Error: ${res.status}`);
      const data = await res.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center">
      <div className="text-center mb-10">
        <h1 className="text-5xl font-black mb-3 bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
          MLOps Bitcoin On-Chain Predictor
        </h1>
      </div>

      <button 
        onClick={handlePredict}
        disabled={loading}
        className="px-8 py-4 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-800 rounded-xl font-bold text-lg transition-all mb-10 shadow-[0_0_20px_rgba(37,99,235,0.2)] hover:shadow-[0_0_30px_rgba(37,99,235,0.4)] disabled:shadow-none"
      >
        {loading ? "EXECUTING INFERENCE..." : "GET TODAY'S SIGNAL"}
      </button>

      {error && <div className="text-red-400 mb-8 font-mono bg-red-950/30 p-4 rounded border border-red-900/50">[ERROR] {error}</div>}

      {prediction && (
        <div className="w-full max-w-md bg-gray-900 border border-gray-800 rounded-2xl p-8 shadow-2xl relative overflow-hidden">
          <div className={`absolute -top-24 -right-24 w-48 h-48 rounded-full blur-[80px] opacity-20 pointer-events-none ${prediction.prediction === 1 ? 'bg-green-500' : 'bg-red-500'}`}></div>
          
          <div className="text-center mb-8 relative z-10">
            <div className={`text-6xl font-black tracking-tighter ${prediction.prediction === 1 ? 'text-green-400' : 'text-red-400'}`}>
              {prediction.prediction === 1 ? "BUY" : "DON'T BUY"}
            </div>
          </div>
          <div className="flex justify-between items-center p-3 bg-gray-950/50 rounded-lg relative z-10">
            <span className="text-gray-400 text-sm">Model Confidence</span>
            <span className="font-mono font-bold">{Number(prediction.probability).toFixed(2)}%</span>
          </div>
        </div>
      )}
    </div>
  );
}
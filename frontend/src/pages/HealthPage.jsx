import { useState, useEffect } from 'react';

export default function HealthPage() {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const checkHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
      const data = await res.json();
      setHealth(data);
    } catch (err) {
      setError(err.message);
      setHealth({ status: "offline", model_loaded: false });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  return (
    <div className="w-full max-w-2xl mx-auto bg-gray-900 border border-gray-800 rounded-xl p-8 shadow-xl">
      <div className="space-y-4">

        <div className="flex justify-between items-center p-4 bg-gray-950/50 rounded-lg border border-gray-800/50">
          <span className="text-gray-400 font-medium">API Server Connection</span>
          <div className="flex items-center space-x-2">
            <div className={`w-2.5 h-2.5 rounded-full ${health?.status === 'online' ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.8)]' : 'bg-red-500'}`}></div>
            <span className="font-mono font-bold text-gray-200">
              {health?.status === 'online' ? 'CONNECTED' : 'DISCONNECTED'}
            </span>
          </div>
        </div>

        <div className="flex justify-between items-center p-4 bg-gray-950/50 rounded-lg border border-gray-800/50">
          <span className="text-gray-400 font-medium">Model</span>
          <div className="flex items-center space-x-2">
            <div className={`w-2.5 h-2.5 rounded-full ${health?.model_loaded ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.8)]' : 'bg-red-500'}`}></div>
            <span className="font-mono font-bold text-gray-200">
              {health?.model_loaded ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>
        </div>

        {error && (
          <div className="mt-4 p-3 bg-red-950/30 border border-red-900/50 rounded text-red-400 font-mono text-sm text-center">
            System Error: {error}
          </div>
        )}
      </div>
    </div>
  );
}
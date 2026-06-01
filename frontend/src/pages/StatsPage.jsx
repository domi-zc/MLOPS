import { useState, useEffect } from 'react';

export default function StatsPage() {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch('/api/stats');
        if (!res.ok) throw new Error('Failed to load stats');
        const data = await res.json();
        setStats(data);
      } catch (err) {
        setError(err.message);
      }
    };
    fetchStats();
  }, []);

  return (
    <div className="w-full max-w-2xl mx-auto bg-gray-900 border border-gray-800 rounded-xl p-8 shadow-xl relative overflow-hidden">
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-600 to-cyan-400"></div>

      <h2 className="text-2xl font-bold text-white mb-6 border-b border-gray-800 pb-4">Metrics</h2>
            
      {error ? (
        <div className="text-red-400 font-mono">API Offline: {error}</div>
      ) : !stats ? (
        <div className="text-gray-400 animate-pulse font-mono">Loading telemetry...</div>
      ) : (
        <div className="space-y-4">
          
          <div className="flex justify-between items-center border-b border-gray-800 pb-3">
            <span className="text-gray-400 text-sm">Model Name</span>
            <span className="font-mono text-blue-400 font-bold">{stats.metrics?.name || "N/A"}</span>
          </div>

          <div className="flex justify-between items-center border-b border-gray-800 pb-3">
            <span className="text-gray-400 text-sm">Model Version</span>
            <span className="font-mono text-blue-400 font-bold">{stats.metrics?.version || "N/A"}</span>
          </div>

          <div className="flex justify-between items-center border-b border-gray-800 pb-3">
            <span className="text-gray-400 text-sm">Created At</span>
            <span className="font-mono text-blue-400 font-bold">
              {stats.metrics?.created_at ? new Date(stats.metrics.created_at).toLocaleString() : "N/A"}
            </span>
          </div>

          <div className="flex justify-between items-center border-b border-gray-800 pb-3">
            <span className="text-gray-400 text-sm">Threshold</span>
            <span className="font-mono text-blue-400 font-bold">{Number(stats.threshold).toFixed(2)}</span>
          </div>

          <div className="flex justify-between items-center border-b border-gray-800 pb-3">
            <span className="text-gray-400 text-sm">Accuracy</span>
            <span className="font-mono text-blue-400 font-bold">{stats.metrics?.accuracy?.toFixed(2) || "N/A"}</span>
          </div>

          <div className="flex justify-between items-center border-b border-gray-800 pb-3">
            <span className="text-gray-400 text-sm">Precision</span>
            <span className="font-mono text-blue-400 font-bold">{stats.metrics?.precision?.toFixed(2) || "N/A"}</span>
          </div>
          
          <div className="flex justify-between items-center border-b border-gray-800 pb-3">
            <span className="text-gray-400 text-sm">Recall</span>
            <span className="font-mono text-blue-400 font-bold">{stats.metrics?.recall?.toFixed(2) || "N/A"}</span>
          </div>
          
          <div className="flex justify-between items-center border-b border-gray-800 pb-3">
            <span className="text-gray-400 text-sm">F1 Score</span>
            <span className="font-mono text-blue-400 font-bold">{stats.metrics?.f1?.toFixed(2) || "N/A"}</span>
          </div>
        </div>
      )}
    </div>
  );
}
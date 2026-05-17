import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import PredictorPage from './pages/PredictorPage';
import StatsPage from './pages/StatsPage';
import HealthPage from './pages/HealthPage';

export default function App() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 font-sans selection:bg-blue-500/30">
      <Navbar />
      
      <div className="container mx-auto px-4 pb-20">
        <Routes>
          <Route path="/" element={<PredictorPage />} />
          <Route path="/stats" element={<StatsPage />} />
          <Route path="/health" element={<HealthPage />} />
        </Routes>
      </div>
    </div>
  );
}
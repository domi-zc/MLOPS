import { Link, useLocation } from 'react-router-dom';

export default function Navbar() {
  const location = useLocation();

  const getLinkClass = (path) => {
    return `font-bold transition-colors ${location.pathname === path ? 'text-blue-400' : 'text-gray-400 hover:text-white'}`;
  };

  return (
    <nav className="w-full bg-gray-900 border-b border-gray-800 px-8 py-4 mb-12 flex justify-center space-x-8 shadow-md">
      <Link to="/" className={getLinkClass('/')}>
        Prediction
      </Link>
      <Link to="/stats" className={getLinkClass('/stats')}>
        Model Information
      </Link>
      <Link to="/health" className={getLinkClass('/health')}>
        Status
      </Link>
    </nav>
  );
}
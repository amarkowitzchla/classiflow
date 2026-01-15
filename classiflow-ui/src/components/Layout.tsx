import { Link, Outlet, useLocation } from 'react-router-dom';
import { LayoutDashboard, Activity } from 'lucide-react';
import { clsx } from 'clsx';
import { useHealth } from '../hooks/useApi';

export function Layout() {
  const location = useLocation();
  const { data: health } = useHealth();

  const navItems = [
    { to: '/', icon: LayoutDashboard, label: 'Projects' },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <Link to="/" className="flex items-center space-x-2">
                <Activity className="w-6 h-6 text-blue-600" />
                <span className="font-semibold text-lg">Classiflow</span>
              </Link>
              <nav className="hidden md:flex items-center space-x-1">
                {navItems.map(({ to, icon: Icon, label }) => (
                  <Link
                    key={to}
                    to={to}
                    className={clsx(
                      'flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium',
                      location.pathname === to
                        ? 'bg-blue-100 text-blue-700'
                        : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                    )}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{label}</span>
                  </Link>
                ))}
              </nav>
            </div>
            <div className="flex items-center space-x-4">
              {health && (
                <span className="text-xs text-gray-400">
                  v{health.version} | {health.project_count} projects
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>
    </div>
  );
}

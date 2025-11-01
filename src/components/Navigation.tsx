import { NavLink } from "react-router-dom";
import { Heart, Home, Calendar, Activity, Apple, User } from "lucide-react";

export const Navigation = () => {
  const navItems = [
    { name: "Dashboard", path: "/", icon: Home },
    { name: "Period Tracker", path: "/period-tracker", icon: Calendar },
    { name: "PCOS Assessment", path: "/pcos-assessment", icon: Activity },
    { name: "Nutrition", path: "/nutrition", icon: Apple },
    { name: "Profile", path: "/profile", icon: User },
  ];

  return (
    <nav className="bg-card/50 backdrop-blur-sm border-b border-border sticky top-0 z-50 shadow-[var(--shadow-soft)]">
      <div className="w-full px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Logo */}
          <NavLink to="/" className="flex items-center gap-3 group">
            <div className="group-hover:scale-105 transition-transform">
              <img src="/menstrual-cycle.png" alt="RituCare Logo" className="w-12 h-12" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
                RituCare
              </h1>
            </div>
          </NavLink>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center gap-1">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === "/"}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-5 py-3 rounded-lg transition-[var(--transition-smooth)] ${
                    isActive
                      ? "bg-gradient-to-br from-primary to-primary/80 text-white shadow-[var(--shadow-card)]"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted"
                  }`
                }
              >
                <item.icon className="w-5 h-5" />
                <span className="text-base font-medium">{item.name}</span>
              </NavLink>
            ))}
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button className="p-2 rounded-lg hover:bg-muted">
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16M4 18h16"
                />
              </svg>
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        <div className="md:hidden pb-4">
          <div className="flex flex-col gap-1">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === "/"}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-5 py-3 rounded-lg transition-[var(--transition-smooth)] ${
                    isActive
                      ? "bg-gradient-to-br from-primary to-primary/80 text-white"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted"
                  }`
                }
              >
                <item.icon className="w-5 h-5" />
                <span className="text-base font-medium">{item.name}</span>
              </NavLink>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

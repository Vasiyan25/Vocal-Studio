import { LayoutDashboard, Scissors, MessageSquareText, Repeat2, AudioWaveform } from "lucide-react";

export const NAV = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "separation", label: "Vocal Separation", icon: Scissors },
  { id: "tts", label: "Text-to-Speech", icon: MessageSquareText },
  { id: "convert", label: "Voice Conversion", icon: Repeat2 },
];

export default function Sidebar({ page, onNavigate }) {
  return (
    <nav className="sidebar" aria-label="Main">
      <div className="brand"><AudioWaveform size={22} aria-hidden="true" /> Vocal Studio</div>
      <ul>
        {NAV.map(({ id, label, icon: Icon }) => (
          <li key={id}>
            <button className={page === id ? "active" : ""} aria-current={page === id ? "page" : undefined} onClick={() => onNavigate(id)}>
              <Icon size={18} aria-hidden="true" /> <span>{label}</span>
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}

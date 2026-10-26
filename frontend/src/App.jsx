import { useEffect, useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import FeaturePage from "./pages/FeaturePage.jsx";
import { api } from "./services/api.js";
import { TaskProvider, useTasks } from "./state/TaskContext.jsx";

const FEATURE_PAGES = { tts: "tts", separate: "separation", convert: "convert" };

function TaskIndicator({ onNavigate }) {
  const { tasks, storageWarning } = useTasks();
  const visibleTasks = [...tasks]
    .sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt))
    .slice(0, 3);

  if (!visibleTasks.length && !storageWarning) return null;
  return (
    <>
      {storageWarning && <p className="storage-warning" role="alert">{storageWarning}</p>}
      {!!visibleTasks.length && (
        <section className="task-indicator" aria-label="Recent tasks">
          <strong>Recent tasks</strong>
          <div>
            {visibleTasks.map((task) => (
              <button key={task.taskId} type="button" onClick={() => onNavigate(FEATURE_PAGES[task.feature])}>
                <span>{task.featureName}</span>
                <span className={`task-status task-${task.status}`} title={task.pollError || task.message}>
                  {task.pollError ? `${task.status} · reconnecting` : task.status}
                </span>
                {task.progress != null && <span>{task.progress}%</span>}
              </button>
            ))}
          </div>
        </section>
      )}
    </>
  );
}

function AppContent() {
  const [page, setPage] = useState("dashboard");
  const [health, setHealth] = useState(null);
  const [models, setModels] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([api.health(), api.models()])
      .then(([h, m]) => { setHealth(h); setModels(m); })
      .catch((e) => setError(e.message));
  }, []);

  const pages = {
    dashboard: <Dashboard health={health} models={models} error={error} onNavigate={setPage} />,
    separation: <FeaturePage key="separation" mode="separate" modelId="demucs" models={models} title="Vocal Separation"
      intro="Split a song into vocals and instrumental." />,
    tts: <FeaturePage key="tts" mode="tts" modelId="tts" models={models} title="Text-to-Speech"
      intro="Turn written text into spoken audio." />,
    convert: <FeaturePage key="convert" mode="convert" modelId="seed-vc" models={models} title="Voice Conversion"
      intro="Convert source speech to resemble a target speaker using a short reference recording. Use only voices you are authorized to use." />,
  };

  return (
    <div className="app">
      <Sidebar page={page} onNavigate={setPage} />
      <main>
        <TaskIndicator onNavigate={setPage} />
        {pages[page]}
      </main>
    </div>
  );
}

export default function App() {
  return <TaskProvider><AppContent /></TaskProvider>;
}

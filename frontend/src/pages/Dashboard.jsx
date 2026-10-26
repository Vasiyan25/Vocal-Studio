import ModelBadge from "../components/ModelBadge.jsx";
import { useTasks } from "../state/TaskContext.jsx";

export default function Dashboard({ health, models, error, onNavigate }) {
  const { tasks } = useTasks();
  return (
    <>
      <h1>Dashboard</h1>
      {error && <p className="error" role="alert">{error}</p>}
      <section className="card">
        <h2>Backend</h2>
        <p>{health ? "Connected. Uploads are validated and stored privately." : error ? "Offline." : "Checking…"}</p>
        {health && (
          <p className="muted">
            Upload limit {health.max_upload_mb} MB. FFmpeg audio check {health.ffprobe ? "is on" : "is off (install FFmpeg for deeper validation)"}.
          </p>
        )}
      </section>
      <section className="card">
        <h2>Model status</h2>
        <ul className="models">{models.map((m) => <ModelBadge key={m.id} model={m} />)}</ul>
        <p className="muted">Features turn on only when their model is installed and verified.</p>
      </section>
      <section className="card">
        <h2>Recent tasks</h2>
        {tasks.length ? (
          <ul className="task-history">
            {tasks.slice(0, 8).map((task) => (
              <li key={task.taskId}>
                <button type="button" onClick={() => onNavigate(
                  task.feature === "separate" ? "separation" : task.feature === "convert" ? "convert" : "tts",
                )}>
                  <strong>{task.featureName}</strong>
                  <span className={`task-status task-${task.status}`}>{task.status}</span>
                  <span>{new Date(task.createdAt).toLocaleString()}</span>
                  <span>{task.status === "completed" && task.outputs.length
                    ? "Result available"
                    : task.pollError || task.error || task.message}</span>
                </button>
              </li>
            ))}
          </ul>
        ) : <p className="muted">Tasks you run will remain available here during this browser session.</p>}
      </section>
      <section className="card">
        <div className="dashboard-demo">
          <div>
            <p className="eyebrow">LISTEN FIRST</p>
            <h2>Explore the Demo Gallery</h2>
            <p className="muted">Hear real samples when available, or try live text-to-speech with the connected backend.</p>
          </div>
          <button className="primary" onClick={() => onNavigate("demos")}>Try a Demo</button>
        </div>
      </section>
      <section className="card">
        <h2>Start here</h2>
        <p>Upload a track on the Vocal Separation page to try the live upload and separation workflow.</p>
        <button className="primary" onClick={() => onNavigate("separation")}>Open Vocal Separation</button>
      </section>
    </>
  );
}

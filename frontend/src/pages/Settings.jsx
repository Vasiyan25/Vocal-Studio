import ModelBadge from "../components/ModelBadge.jsx";

export default function Settings({ models, health }) {
  return (
    <>
      <h1>Settings and model status</h1>
      <section className="card">
        <h2>Models</h2>
        <ul className="models">{models.map((m) => <ModelBadge key={m.id} model={m} />)}</ul>
      </section>
      <section className="card">
        <h2>Backend</h2>
        <p className="muted">API address: {import.meta.env.VITE_API_URL || "http://localhost:8000"}</p>
        <p className="muted">{health ? "Connected" : "Not connected"}</p>
      </section>
    </>
  );
}

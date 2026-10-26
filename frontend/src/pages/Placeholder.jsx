export default function Placeholder({ title, text }) {
  return (
    <>
      <h1>{title}</h1>
      <section className="card"><p>{text}</p></section>
    </>
  );
}

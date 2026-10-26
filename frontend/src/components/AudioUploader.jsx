import { useRef, useState } from "react";
import { UploadCloud } from "lucide-react";
import { api } from "../services/api.js";

const ALLOWED = [".wav", ".mp3", ".flac", ".ogg", ".m4a"];
const MAX_MB = 50;

export default function AudioUploader({
  onUploaded,
  label = "Audio file",
  hint = `WAV, MP3, FLAC, OGG or M4A, up to ${MAX_MB} MB`,
  maxMb = MAX_MB,
}) {
  const input = useRef(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function handle(file) {
    if (!file) return;
    setError("");
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!ALLOWED.includes(ext)) return setError(`Choose a ${ALLOWED.join(", ")} file.`);
    if (file.size > maxMb * 1024 * 1024) return setError(`File is larger than ${maxMb} MB.`);
    setBusy(true);
    try {
      onUploaded(await api.upload(file));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  }

  return (
    <div>
      {label && <p className="uploader-label">{label}</p>}
      <button
        type="button"
        className="dropzone"
        disabled={busy}
        onClick={() => input.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); handle(e.dataTransfer.files[0]); }}
      >
        <UploadCloud size={28} aria-hidden="true" />
        <span>{busy ? "Uploading and checking the audio…" : "Choose an audio file or drop it here"}</span>
        <small>{hint}</small>
      </button>
      <input ref={input} type="file" hidden accept={ALLOWED.join(",")} onChange={(e) => handle(e.target.files[0])} />
      {error && <p className="error" role="alert">{error}</p>}
    </div>
  );
}

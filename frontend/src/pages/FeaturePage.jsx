import { useEffect, useRef, useState } from "react";
import AudioUploader from "../components/AudioUploader.jsx";
import AudioPlayer from "../components/AudioPlayer.jsx";
import ModelBadge from "../components/ModelBadge.jsx";
import JobResult from "../components/JobResult.jsx";
import { api } from "../services/api.js";
import { useTasks } from "../state/TaskContext.jsx";

/** One page component for the three AI features. Each calls the real endpoint. */
function normalizeVoices(raw) {
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((entry) => {
    if (typeof entry === "string") {
      return [{ id: entry, label: entry }];
    }
    if (entry && typeof entry === "object") {
      const id = typeof entry.id === "string" ? entry.id : typeof entry.name === "string" ? entry.name : null;
      if (!id) return [];
      return [{ id, label: typeof entry.name === "string" ? entry.name : id }];
    }
    return [];
  });
}

function UploadedFileCard({ file, label }) {
  if (!file) return null;
  return (
    <div className="file">
      <p><strong>{file.filename}</strong> <span className="muted">
        {file.duration_seconds ? `${file.duration_seconds.toFixed(1)} s, ` : ""}{(file.size_bytes / 1048576).toFixed(2)} MB</span></p>
      <AudioPlayer src={api.streamUrl(file.file_id)} label={label} />
      <p className="muted">Uploads are kept temporarily by the backend. If this file has expired, upload it again.</p>
    </div>
  );
}

export default function FeaturePage({ title, intro, modelId, models, mode }) {
  const feature = mode === "separate" ? "separate" : mode;
  const { tasks, drafts, createTask, updateTask, updateDraft } = useTasks();
  const draft = drafts[feature] || {};
  const file = draft.file || null;
  const referenceFile = draft.referenceFile || null;
  const text = draft.text || "";
  const voice = draft.voice || "";
  const [voices, setVoices] = useState([]);
  const [voicesError, setVoicesError] = useState("");
  const ttsModelId = draft.ttsModelId || "";
  const conversionMethod = draft.conversionMethod || "seed-vc";
  const [rvcVoices, setRvcVoices] = useState([]);
  const [rvcVoicesError, setRvcVoicesError] = useState("");
  const submitting = useRef(false);
  const availableTtsModel = models.find((item) => item.id === "kokoro" && item.available)
    || models.find((item) => item.id === "tts" && item.available);
  const activeTtsModelId = ttsModelId || availableTtsModel?.id || "kokoro";
  const activeModelId = mode === "tts"
    ? activeTtsModelId
    : mode === "convert" && conversionMethod === "rvc" ? "rvc" : modelId;
  const model = models.find((m) => m.id === activeModelId);
  const task = tasks.find((item) => item.feature === feature);
  const busy = task && ["submitting", "queued", "running", "recovering"].includes(task.status);
  const voiceOptions = normalizeVoices(voices);

  useEffect(() => {
    if (mode === "tts") {
      setVoicesError("");
      api.ttsVoices(activeTtsModelId)
        .then((list) => {
          const normalized = normalizeVoices(list);
          setVoices(normalized);
          if (!normalized.some((entry) => entry.id === voice)) {
            updateDraft(feature, { voice: normalized[0]?.id || "" });
          }
        })
        .catch((e) => setVoicesError(e.message));
    } else if (mode === "convert") {
      api.voices()
        .then((list) => setRvcVoices(normalizeVoices(list)))
        .catch((e) => setRvcVoicesError(e.message));
    }
  }, [mode, feature, updateDraft, voice, activeTtsModelId]);

  async function run() {
    if (busy || submitting.current || !ready) return;
    submitting.current = true;
    const record = createTask(feature, {
      text: mode === "tts" ? text : "",
      voice: mode === "tts" ? voice : "",
      file: needsFile ? file : null,
      referenceFile: mode === "convert" ? referenceFile : null,
      conversionMethod: mode === "convert" ? conversionMethod : "",
    });
    try {
      const started =
        mode === "separate" ? await api.separate(file.file_id)
        : mode === "tts" ? await api.tts(text, voice, activeTtsModelId)
        : conversionMethod === "rvc"
          ? await api.convertRvc(file.file_id, voice)
          : await api.convert(file.file_id, referenceFile.file_id);
      updateTask(record.taskId, {
        job_id: started.job_id,
        status: started.status,
        message: started.message,
        outputs: started.outputs || [],
        error: "",
      });
      submitting.current = false;
    } catch (e) {
      updateTask(record.taskId, { status: "failed", message: "Task could not be started.", error: e.message });
      submitting.current = false;
    }
  }

  const needsFile = mode !== "tts";
  const ttsModels = models.filter((item) => item.id === "kokoro" || item.id === "tts");
  const ready = mode === "tts"
    ? !!model?.available && text.trim().length > 0 && !!voice && voiceOptions.some((entry) => entry.id === voice)
    : mode === "convert"
      ? conversionMethod === "rvc"
        ? !!file && !!voice && rvcVoices.some((entry) => entry.id === voice) && !!model?.available
        : !!file && !!referenceFile && file.file_id !== referenceFile.file_id
      : !!file;
  const job = task?.job_id ? task : null;

  return (
    <>
      <h1>{title}</h1>
      <p className="muted">{intro}</p>
      {mode === "separate" && model && <ul className="models card"><ModelBadge model={model} /></ul>}
      <section className="card">
        {mode === "convert" && (
          <>
            <div className="upload-block">
              <AudioUploader
                label="1. Source audio"
                hint="Recording to convert (same words will be kept). WAV/MP3/FLAC/OGG/M4A, up to 50 MB."
                onUploaded={(uploaded) => updateDraft(feature, { file: uploaded })}
              />
              <UploadedFileCard file={file} label="Source audio" />
            </div>
            <label className="field">Target voice method
              <select value={conversionMethod} onChange={(event) => updateDraft(feature, { conversionMethod: event.target.value })}>
                <option value="seed-vc">Seed-VC reference audio</option>
                {rvcVoices.length > 0 && models.some((item) => item.id === "rvc" && item.available)
                  && <option value="rvc">Installed RVC voice model</option>}
              </select>
            </label>
            {conversionMethod === "seed-vc" ? (
              <>
                <div className="upload-block">
                  <AudioUploader
                    label="2. Target voice sample"
                    hint="Short clean clip of the desired speaker (about 3–30 seconds). WAV/MP3/FLAC/OGG/M4A, up to 20 MB."
                    maxMb={20}
                    onUploaded={(uploaded) => updateDraft(feature, { referenceFile: uploaded })}
                  />
                  <UploadedFileCard file={referenceFile} label="Target voice sample" />
                </div>
                {file && referenceFile && file.file_id === referenceFile.file_id && (
                  <p className="error" role="alert">Source and target must be two different uploads.</p>
                )}
              </>
            ) : (
              <label className="field">Installed RVC voice
                <select value={voice} onChange={(event) => updateDraft(feature, { voice: event.target.value })}>
                  <option value="">Choose an installed voice</option>
                  {rvcVoices.map(({ id, label }) => <option key={id} value={id}>{label}</option>)}
                </select>
              </label>
            )}
            {conversionMethod === "seed-vc" && model && !model.available
              && <p className="error" role="alert">Seed-VC is unavailable: {model.detail}</p>}
            {rvcVoicesError && <p className="error" role="alert">{rvcVoicesError}</p>}
          </>
        )}
        {needsFile && mode !== "convert" && (
          <>
            <AudioUploader onUploaded={(uploaded) => updateDraft(feature, { file: uploaded })} />
            <UploadedFileCard file={file} label="Your upload" />
          </>
        )}
        {mode === "tts" && (
          <>
            <label className="field">Speech model
              <select value={activeTtsModelId} onChange={(event) => {
                updateDraft(feature, { ttsModelId: event.target.value, voice: "" });
              }}>
                {ttsModels.map((item) => (
                  <option key={item.id} value={item.id} disabled={!item.available}>
                    {item.name}{item.available ? "" : " (unavailable)"}
                  </option>
                ))}
              </select>
            </label>
            {model && !model.available && <p className="error" role="alert">{model.detail}</p>}
            <label className="field">Text to speak
              <textarea rows={4} maxLength={2000} value={text} onChange={(e) => updateDraft(feature, { text: e.target.value })} />
            </label>
            <label className="field">Voice
              {voiceOptions.length ? (
                <select value={voice || ""} onChange={(e) => updateDraft(feature, { voice: e.target.value })}>
                  {voiceOptions.map(({ id, label }, index) => <option key={`${id || label || "voice"}-${index}`} value={id}>{label}</option>)}
                  {voice && !voiceOptions.some((entry) => entry.id === voice) && <option value={voice}>{voice} (currently unavailable)</option>}
                </select>
              ) : (
                <p className="muted">No voices are available for the selected speech model.</p>
              )}
            </label>
          </>
        )}
        {mode === "tts" && voicesError && <p className="error" role="alert">{voicesError}</p>}
        <button className="primary" disabled={!ready || busy} onClick={run}>
          {busy ? "Working…" : title === "Vocal Separation" ? "Separate vocals" : mode === "tts" ? "Generate speech" : "Convert voice"}
        </button>
        {task && (
          <p className="status" role="status">
            {task.status === "recovering" ? "Checking task status with the backend…" : task.message}
            {task.device ? ` · ${task.device}` : ""}
          </p>
        )}
        {task?.pollError && <p className="error" role="alert">Cannot check task status right now: {task.pollError}. Retrying automatically.</p>}
        {task?.error && <p className="error" role="alert">{task.error}</p>}
        {task?.status === "unavailable" && (
          <p className="error" role="alert">This task cannot be recovered. The backend may have restarted or its task record may have expired. Start a new task to continue.</p>
        )}
      </section>
      {job?.status === "completed" && job.outputs.length > 0 && (
        <section className="card">
          {mode === "convert" && file && (
            <div className="result compare-original">
              <AudioPlayer src={api.streamUrl(file.file_id)} label="Original source" />
            </div>
          )}
          <JobResult job={job} />
        </section>
      )}
    </>
  );
}

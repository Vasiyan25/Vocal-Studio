const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, options);
  } catch {
    throw new Error("Cannot reach the backend. Start it with: uvicorn app.main:app --reload --port 8000");
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d) => d.msg).join("; ");
    } catch { /* keep default */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

const json = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request("/api/health"),
  models: () => request("/api/models"),
  voices: () => request("/api/voices"),
  ttsVoices: (modelId) => modelId
    ? request(`/api/tts/voices/${encodeURIComponent(modelId)}`)
    : request("/api/tts/voices"),
  upload: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/api/audio/upload", { method: "POST", body: form });
  },
  streamUrl: (fileId) => `${BASE}/api/audio/${fileId}/stream`,
  separate: (fileId) => request("/api/audio/separate", json({ file_id: fileId })),
  tts: (text, voiceId, modelId) =>
    request("/api/audio/tts", json({ text, voice_id: voiceId || undefined, model_id: modelId || "tts" })),
  convert: (fileId, referenceFileId) =>
    request("/api/audio/convert", json({ file_id: fileId, reference_file_id: referenceFileId })),
  convertRvc: (fileId, voiceModel) =>
    request("/api/audio/convert-rvc", json({ file_id: fileId, voice_model: voiceModel })),
  job: (id) => request(`/api/jobs/${id}`),
  jobStreamUrl: (id, stem) => `${BASE}/api/jobs/${id}/stream?stem=${encodeURIComponent(stem)}`,
  downloadUrl: (id, stem) => `${BASE}/api/jobs/${id}/download?stem=${encodeURIComponent(stem)}`,
};

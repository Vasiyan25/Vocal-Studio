import { useEffect, useMemo, useState } from "react";
import { ArrowRight, AudioLines, CheckCircle2, Clock3, Mic2, Scissors, Sparkles, Volume2 } from "lucide-react";
import AudioPlayer from "../components/AudioPlayer.jsx";
import JobResult from "../components/JobResult.jsx";
import { api } from "../services/api.js";

const SAMPLE_CATEGORIES = [
  {
    id: "separation",
    title: "Vocal Separation",
    description: "Compare a spoken Piper sample with the vocals and instrumental stems produced by Demucs.",
    icon: Scissors,
    samples: [
      {
        title: "Original spoken sample",
        src: "/demos/tts/piper-lessac-welcome.wav",
        description: "Piper source speech: “A little melody can brighten the whole day.” This is speech, not a music track.",
        download: "Download source sample (WAV)",
      },
      {
        title: "Demucs vocal stem",
        src: "/demos/separation/piper-speech-vocals.wav",
        description: "Vocal stem from a real Demucs separation job using the Piper source sample.",
        download: "Download vocal stem (WAV)",
      },
      {
        title: "Demucs instrumental stem",
        src: "/demos/separation/piper-speech-instrumental.wav",
        description: "Instrumental stem returned by the same Demucs job. Speech input may leave little instrumental content.",
        download: "Download instrumental stem (WAV)",
      },
    ],
    livePage: "separation",
    liveLabel: "Separate your audio",
  },
  {
    id: "tts",
    title: "Text-to-Speech",
    description: "Listen to a prepared Piper sample or try live speech generation below.",
    icon: AudioLines,
    samples: [{
      title: "Piper speech sample",
      src: "/demos/tts/piper-lessac-welcome.wav",
      description: "Generated with en_US-lessac-medium: “A little melody can brighten the whole day.”",
      download: "Download Piper sample (WAV)",
    }],
    livePage: "tts",
    liveLabel: "Open full TTS workspace",
  },
  {
    id: "voice-conversion",
    title: "Voice Conversion",
    description: "Listen to a matching Piper source and its genuine output from the installed RVC checkpoint.",
    icon: Mic2,
    samples: [
      {
        title: "Original Piper voice",
        src: "/demos/tts/piper-lessac-welcome.wav",
        description: "The source audio used for this conversion.",
        download: "Download original (WAV)",
      },
      {
        title: "RVC output · default checkpoint",
        src: "/demos/voice-conversion/piper-to-default.wav",
        description: "Actual RVC output generated from the source above with the installed default.pth checkpoint. The checkpoint label is not a claim about a person's identity.",
        download: "Download converted sample (WAV)",
      },
    ],
    livePage: "convert",
    liveLabel: "Open live voice conversion",
  },
  {
    id: "voice-lab",
    title: "Voice Lab",
    description: "Preview the configured Piper voice. Voice Lab processing is not active in this app yet.",
    icon: Volume2,
    samples: [{
      title: "Piper voice preview",
      src: "/demos/tts/piper-lessac-welcome.wav",
      description: "Prepared Piper speech preview; this is not a Voice Lab processing output.",
      download: "Download voice preview (WAV)",
    }],
    livePage: "voicelab",
    liveLabel: "Explore Voice Lab",
  },
];

function SampleCard({ sample }) {
  const hasAudio = typeof sample.src === "string" && sample.src.startsWith("/demos/");
  return (
    <div className="demo-sample">
      <div className="demo-sample-heading">
        <span>{sample.title}</span>
        {!hasAudio && <span className="demo-coming"><Clock3 size={13} aria-hidden="true" /> Coming soon</span>}
      </div>
      {hasAudio ? (
        <>
          <AudioPlayer src={sample.src} label={sample.title} />
          {sample.description && <p className="demo-note">{sample.description}</p>}
          {sample.download && <a className="download" href={sample.src} download>{sample.download}</a>}
        </>
      ) : (
        <>
          <p className="muted">No verified demo audio is published for this sample yet.</p>
          <p className="demo-note">Playback will be enabled when an authorized audio file is added.</p>
        </>
      )}
    </div>
  );
}

function LiveTts({ available, voiceName }) {
  const [text, setText] = useState("A little melody can brighten the whole day.");
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!job?.job_id || !busy) return undefined;
    let cancelled = false;
    let timer;
    async function poll() {
      try {
        const result = await api.job(job.job_id);
        if (cancelled) return;
        setJob(result);
        if (result.status === "completed") {
          setBusy(false);
        } else if (result.status === "failed") {
          setError(result.error || "Speech generation failed.");
          setBusy(false);
        } else {
          timer = setTimeout(poll, 1200);
        }
      } catch (cause) {
        if (!cancelled) {
          setError(cause.message);
          setBusy(false);
        }
      }
    }
    timer = setTimeout(poll, 500);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [job?.job_id, busy]);

  async function generate(event) {
    event.preventDefault();
    setError("");
    setBusy(true);
    setJob(null);
    try {
      const started = await api.tts(text);
      setJob(started);
    } catch (cause) {
      setError(cause.message);
      setBusy(false);
    }
  }

  return (
    <section className="demo-live" aria-labelledby="live-tts-title">
      <div className="demo-live-title">
        <span className="live-dot" aria-hidden="true" />
        <div>
          <p className="eyebrow">LIVE PROCESSING · REAL BACKEND</p>
          <h3 id="live-tts-title">Generate speech with Piper</h3>
        </div>
      </div>
      <p className="muted">This sends your text to the installed TTS model and creates new audio only when you press the button. It does not play a prepared sample.</p>
      <label className="field">Example sentence
        <textarea rows={3} maxLength={2000} value={text} onChange={(event) => setText(event.target.value)} disabled={!available || busy} />
      </label>
      <label className="field">Installed backend voice
        <select value={available ? voiceName : ""} disabled>
          {available
            ? <option value={voiceName}>{voiceName}</option>
            : <option value="">Piper is not ready</option>}
        </select>
      </label>
      {!available && <p className="demo-unavailable">Live speech generation is unavailable until Piper is ready.</p>}
      <button className="primary" type="button" disabled={!available || busy || !text.trim()} onClick={generate}>
        {busy ? "Generating…" : "Generate with Piper"}
      </button>
      {busy && <p className="status" role="status">{job?.message || "Submitting speech job…"}</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {job?.status === "completed" && job.outputs?.length > 0 && <JobResult job={job} />}
    </section>
  );
}

export default function DemoGallery({ models, onNavigate }) {
  const [voices, setVoices] = useState([]);
  const [voicesError, setVoicesError] = useState("");
  const ttsModel = models.find((model) => model.id === "tts");
  const rvcModel = models.find((model) => model.id === "rvc");
  const voiceName = useMemo(
    () => ttsModel?.detail?.match(/Piper voice ([A-Za-z0-9._-]+)/)?.[1] || "",
    [ttsModel],
  );

  useEffect(() => {
    api.voices().then(setVoices).catch((cause) => setVoicesError(cause.message));
  }, []);

  return (
    <div className="demo-gallery">
      <header className="demo-hero">
        <div className="demo-hero-icon"><Sparkles size={22} aria-hidden="true" /></div>
        <p className="eyebrow">VOCAL STUDIO · LISTEN & EXPLORE</p>
        <h1>Try a Demo</h1>
        <p className="muted">Explore the studio before uploading your own audio. Sample demos and live AI processing are clearly separated.</p>
        <div className="demo-legend">
          <span><AudioLines size={15} aria-hidden="true" /> SAMPLE DEMO</span>
          <span><CheckCircle2 size={15} aria-hidden="true" /> LIVE PROCESSING</span>
        </div>
      </header>

      <div className="demo-grid">
        {SAMPLE_CATEGORIES.map(({ id, title, description, icon: Icon, samples, livePage, liveLabel }) => (
          <section className={`demo-card demo-${id}`} key={id} aria-labelledby={`demo-${id}-title`}>
            <div className="demo-card-heading">
              <span className="demo-icon"><Icon size={20} aria-hidden="true" /></span>
              <span className="demo-tag"><AudioLines size={13} aria-hidden="true" /> SAMPLE DEMO</span>
            </div>
            <h2 id={`demo-${id}-title`}>{title}</h2>
            <p className="muted">{description}</p>
            <div className="demo-samples">
              {samples.map((sample) => <SampleCard key={sample.title} sample={sample} />)}
            </div>
            {id === "tts" && <LiveTts available={Boolean(ttsModel?.available && voiceName)} voiceName={voiceName} />}
            {id === "voice-conversion" && (
              <div className="demo-installed-voices">
                <p className="eyebrow">INSTALLED VOICE MODELS</p>
                {voicesError ? <p className="error" role="alert">{voicesError}</p>
                  : voices.length
                    ? <ul>{voices.map((voice) => <li key={voice}>{voice}<span>Installed · demo sample not published</span></li>)}</ul>
                    : <p className="muted">{rvcModel?.available ? "No voice models are currently listed." : "RVC is not currently ready."}</p>}
                <p className="demo-note">The sample above was generated with the installed default checkpoint. Other installed models are not presented as sample audio unless a genuine, authorized before-and-after pair is added.</p>
              </div>
            )}
            <button className="demo-live-link" type="button" onClick={() => onNavigate(livePage)}>
              {liveLabel}<ArrowRight size={16} aria-hidden="true" />
            </button>
          </section>
        ))}
      </div>

      <aside className="demo-footer card">
        <h2>Have authorized samples to share?</h2>
        <p className="muted">Add the genuine audio files to the documented demo folders and update the sample manifest in the gallery. Don’t put private uploads, voice checkpoints, or unlabeled AI outputs in the public directory.</p>
      </aside>
    </div>
  );
}

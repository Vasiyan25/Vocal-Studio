import { useEffect, useRef, useState } from "react";
import { LoaderCircle, Pause, Play, Volume2 } from "lucide-react";

let activeAudio = null;

function formatTime(value) {
  if (!Number.isFinite(value) || value < 0) return "0:00";
  const seconds = Math.floor(value);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

export default function AudioPlayer({ src, label }) {
  const audio = useRef(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [volume, setVolume] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(Boolean(src));
  const [error, setError] = useState("");

  useEffect(() => {
    setDuration(0);
    setCurrentTime(0);
    setPlaying(false);
    setLoading(Boolean(src));
    setError("");
    return () => {
      if (activeAudio === audio.current) activeAudio = null;
    };
  }, [src]);

  function onPlay() {
    if (activeAudio && activeAudio !== audio.current) activeAudio.pause();
    activeAudio = audio.current;
    setPlaying(true);
  }

  function togglePlayback() {
    if (!audio.current || !src) return;
    setError("");
    if (audio.current.paused) {
      audio.current.play().catch(() => {
        setPlaying(false);
        setError("Playback failed. The audio file may be unavailable or unsupported.");
      });
    } else {
      audio.current.pause();
    }
  }

  function seek(event) {
    if (audio.current && duration > 0) {
      const nextTime = Number(event.target.value);
      audio.current.currentTime = nextTime;
      setCurrentTime(nextTime);
    }
  }

  function changeVolume(event) {
    const nextVolume = Number(event.target.value);
    setVolume(nextVolume);
    if (audio.current) audio.current.volume = nextVolume;
  }

  return (
    <figure className="player">
      {label && <figcaption>{label}</figcaption>}
      <audio
        ref={audio}
        preload="metadata"
        src={src || undefined}
        onLoadedMetadata={(event) => {
          setDuration(Number.isFinite(event.currentTarget.duration) ? event.currentTarget.duration : 0);
          setLoading(false);
        }}
        onDurationChange={(event) => {
          if (Number.isFinite(event.currentTarget.duration)) setDuration(event.currentTarget.duration);
        }}
        onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
        onWaiting={() => setLoading(true)}
        onCanPlay={() => setLoading(false)}
        onPlay={onPlay}
        onPause={() => setPlaying(false)}
        onEnded={() => {
          setPlaying(false);
          setCurrentTime(0);
          if (audio.current) audio.current.currentTime = 0;
        }}
        onError={() => {
          setLoading(false);
          setError("Audio could not be loaded. The file may be unavailable or expired from temporary storage.");
        }}
      />
      <div className="audio-controls">
        <button
          className="audio-play"
          type="button"
          aria-label={playing ? `Pause ${label || "audio"}` : `Play ${label || "audio"}`}
          onClick={togglePlayback}
          disabled={!src || loading}
        >
          {loading ? <LoaderCircle className="spin" size={19} aria-hidden="true" /> : playing ? <Pause size={19} aria-hidden="true" /> : <Play size={19} aria-hidden="true" />}
        </button>
        <span className="audio-time" aria-label="Current time">{formatTime(currentTime)}</span>
        <input
          className="audio-seek"
          type="range"
          min="0"
          max={duration || 0}
          step="0.1"
          value={Math.min(currentTime, duration || 0)}
          onChange={seek}
          aria-label={`Seek ${label || "audio"}`}
          disabled={!src || !duration}
          style={{ "--audio-progress": duration ? `${(currentTime / duration) * 100}%` : "0%" }}
        />
        <span className="audio-time" aria-label="Duration">{formatTime(duration)}</span>
        <label className="audio-volume" aria-label={`Volume for ${label || "audio"}`}>
          <Volume2 size={17} aria-hidden="true" />
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={volume}
            onChange={changeVolume}
            aria-label={`Volume ${label || "audio"}`}
            disabled={!src}
          />
        </label>
      </div>
      {error && <p className="audio-error" role="alert">{error}</p>}
    </figure>
  );
}

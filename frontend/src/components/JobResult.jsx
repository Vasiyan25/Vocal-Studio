import { Download } from "lucide-react";
import AudioPlayer from "./AudioPlayer.jsx";
import { api } from "../services/api.js";

const LABELS = {
  vocals: "Vocals",
  instrumental: "Instrumental",
  speech: "Generated speech",
  converted: "Converted audio",
};

/** Shows one player and one download link per output file of a finished job. */
export default function JobResult({ job }) {
  return (
    <div className="results">
      <h2>Results</h2>
      {job.outputs.map((stem) => (
        <div key={stem} className="result">
          <AudioPlayer src={api.jobStreamUrl(job.job_id, stem)} label={LABELS[stem] || stem} />
          <a className="download" href={api.downloadUrl(job.job_id, stem)}>
            <Download size={16} aria-hidden="true" /> Download {(LABELS[stem] || stem).toLowerCase()} (WAV)
          </a>
        </div>
      ))}
    </div>
  );
}

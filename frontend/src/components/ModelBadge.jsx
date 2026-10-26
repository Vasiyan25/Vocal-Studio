import { CheckCircle2, CircleSlash } from "lucide-react";

export default function ModelBadge({ model }) {
  const Icon = model.available ? CheckCircle2 : CircleSlash;
  return (
    <li className={`model ${model.available ? "ok" : "off"}`}>
      <Icon size={18} aria-hidden="true" />
      <div>
        <strong>{model.name}</strong> <span className="muted">{model.feature}</span>
        <p>{model.available ? "Ready" : model.detail}</p>
      </div>
    </li>
  );
}

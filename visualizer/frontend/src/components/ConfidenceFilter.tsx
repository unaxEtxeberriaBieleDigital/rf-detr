interface ConfidenceFilterProps {
  minConfidence: number;
  onChange: (value: number) => void;
}

/** Slider that lets the user hide predictions below a given confidence threshold. */
export default function ConfidenceFilter({ minConfidence, onChange }: ConfidenceFilterProps) {
  return (
    <div className="confidence-filter">
      <label htmlFor="confidence-slider">Confianza mínima de las predicciones: {minConfidence.toFixed(2)}</label>
      <input
        id="confidence-slider"
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={minConfidence}
        onChange={(e) => onChange(Number(e.currentTarget.value))}
      />
    </div>
  );
}

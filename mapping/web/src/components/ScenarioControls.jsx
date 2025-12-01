import React from "react";

function ScenarioControls({ scenarios, selectedId, onSelect, isPlaying, onTogglePlay }) {
  return (
    <div className="controls-panel">
      <div className="controls-header">POI Prediction Demo</div>
      <div className="controls-row">
        <label className="controls-label" htmlFor="scenario-select">
          Scenario
        </label>
        <select
          id="scenario-select"
          className="controls-select"
          value={selectedId || ""}
          onChange={(e) => onSelect(e.target.value)}
        >
          {scenarios.length === 0 && <option value="">Loading</option>}
          {scenarios.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name || s.id}
            </option>
          ))}
        </select>
      </div>
      <button className="controls-button" type="button" onClick={onTogglePlay}>
        {isPlaying ? "Pause" : "Play"}
      </button>
    </div>
  );
}

export default ScenarioControls;

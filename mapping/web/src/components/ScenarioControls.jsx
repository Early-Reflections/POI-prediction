import React from "react";

function ScenarioControls({
  scenarios,
  selectedId,
  onSelect,
  isPlaying,
  onTogglePlay,
  dataSources,
  dataSourceId,
  onChangeDataSource,
  zoomBias,
  onChangeZoomBias
}) {
  return (
    <div className="controls-panel">
      <div className="controls-header">POI Prediction Demo</div>

      <div className="controls-row">
        <label className="controls-label" htmlFor="dataset-select">
          Dataset
        </label>
        <select
          id="dataset-select"
          className="controls-select"
          value={dataSourceId}
          onChange={(e) => onChangeDataSource(e.target.value)}
        >
          {Array.isArray(dataSources) &&
            dataSources.map((d) => (
              <option key={d.id} value={d.id}>
                {d.label || d.id}
              </option>
            ))}
        </select>
      </div>

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

      <div className="controls-row">
        <label className="controls-label" htmlFor="zoom-bias">
          Zoom bias
        </label>
        <input
          id="zoom-bias"
          className="controls-range"
          type="range"
          min="-3"
          max="3"
          step="0.5"
          value={zoomBias}
          onChange={(e) => onChangeZoomBias(parseFloat(e.target.value))}
        />
        <span className="controls-value">{zoomBias.toFixed(1)}</span>
      </div>

      <button className="controls-button" type="button" onClick={onTogglePlay}>
        {isPlaying ? "Pause" : "Play"}
      </button>
    </div>
  );
}

export default ScenarioControls;

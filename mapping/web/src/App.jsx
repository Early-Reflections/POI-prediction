import React, { useEffect, useState } from "react";
import MapView from "./components/MapView.jsx";
import ScenarioControls from "./components/ScenarioControls.jsx";

const ANIMATION_DURATION_MS = 14000;

const DATA_SOURCES = [
  { id: "nyc-real", label: "NYC - real predictions", path: "/data/demo-nyc-real.json" },
  { id: "nyc-synth", label: "NYC - synthetic demo", path: "/data/demo-nyc.json" },
  { id: "tky-real", label: "Tokyo - real predictions", path: "/data/demo-tky-real.json" }
];

function App() {
  const [scenarios, setScenarios] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [timeMs, setTimeMs] = useState(0);
  const [dataSourceId, setDataSourceId] = useState("nyc-real");
  const [zoomBias, setZoomBias] = useState(-1);
  const [datasetVersion, setDatasetVersion] = useState(0);

  useEffect(() => {
    const source = DATA_SOURCES.find((d) => d.id === dataSourceId) || DATA_SOURCES[0];
    console.log("[App] Loading dataset", dataSourceId, "from", source.path);
    fetch(source.path)
      .then((res) => {
        if (!res.ok) {
          console.error("[App] Dataset fetch HTTP error", res.status, res.statusText, {
            dataSourceId,
            path: source.path
          });
        }
        return res.json();
      })
      .then((data) => {
        const list = Array.isArray(data.scenarios) ? data.scenarios : [];
        console.log("[App] Dataset loaded", {
          dataSourceId,
          scenarioCount: list.length,
          rawDataset: data && data.dataset
        });
        setScenarios(list);
        if (list.length > 0) {
          setSelectedId(list[0].id);
        } else {
          setSelectedId(null);
        }
        setTimeMs(0);
        setDatasetVersion((v) => v + 1);
      })
      .catch((error) => {
        console.error("[App] Failed to load dataset", dataSourceId, "from", source.path, error);
        setScenarios([]);
        setSelectedId(null);
      });
  }, [dataSourceId]);

  useEffect(() => {
    if (!isPlaying) {
      console.log("[App] Animation paused", {
        dataSourceId,
        scenarioCount: scenarios.length
      });
      return;
    }
    console.log("[App] Animation loop starting", {
      dataSourceId,
      scenarioCount: scenarios.length
    });
    let frameId;
    let last = performance.now();
    const loop = (now) => {
      const delta = now - last;
      last = now;

      setTimeMs((prev) => {
        let total = prev + delta;
        if (total >= ANIMATION_DURATION_MS) {
          total = total % ANIMATION_DURATION_MS;
          setSelectedId((currentId) => {
            if (!scenarios.length) {
              return currentId;
            }
            const currentIndex = Math.max(
              0,
              scenarios.findIndex((s) => s.id === currentId)
            );
            const nextIndex = (currentIndex + 1) % scenarios.length;
            return scenarios[nextIndex].id;
          });
        }
        return total;
      });
      frameId = requestAnimationFrame(loop);
    };
    frameId = requestAnimationFrame(loop);
    return () => {
      if (frameId) {
        cancelAnimationFrame(frameId);
      }
      console.log("[App] Animation loop stopped", {
        dataSourceId
      });
    };
  }, [isPlaying, scenarios]);

  useEffect(() => {
    if (!selectedId) {
      console.log("[App] No scenario selected for dataset", dataSourceId);
      return;
    }
    console.log("[App] Selected scenario changed", {
      dataSourceId,
      selectedId
    });
  }, [selectedId, dataSourceId]);

  const activeScenario = scenarios.find((s) => s.id === selectedId) || null;

  const handleSelectScenario = (id) => {
    console.log("[App] Scenario clicked in UI", {
      previousSelectedId: selectedId,
      nextSelectedId: id,
      dataSourceId
    });
    setSelectedId(id);
    setTimeMs(0);
  };

  return (
    <div className="app-root">
      <MapView
        scenario={activeScenario}
        timeMs={timeMs}
        durationMs={ANIMATION_DURATION_MS}
        zoomBias={zoomBias}
        datasetId={dataSourceId}
        datasetVersion={datasetVersion}
      />
      <ScenarioControls
        scenarios={scenarios}
        selectedId={selectedId}
        onSelect={handleSelectScenario}
        isPlaying={isPlaying}
        onTogglePlay={() => setIsPlaying((v) => !v)}
        dataSources={DATA_SOURCES}
        dataSourceId={dataSourceId}
        onChangeDataSource={setDataSourceId}
        zoomBias={zoomBias}
        onChangeZoomBias={setZoomBias}
      />
    </div>
  );
}

export default App;

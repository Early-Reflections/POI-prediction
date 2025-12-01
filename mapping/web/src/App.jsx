import React, { useEffect, useState } from "react";
import MapView from "./components/MapView.jsx";
import ScenarioControls from "./components/ScenarioControls.jsx";

const ANIMATION_DURATION_MS = 14000;

function App() {
  const [scenarios, setScenarios] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [timeMs, setTimeMs] = useState(0);

  useEffect(() => {
    //fetch("/data/demo-nyc.json")
    fetch("/data/demo-nyc-real.json")
      .then((res) => res.json())
      .then((data) => {
        const list = Array.isArray(data.scenarios) ? data.scenarios : [];
        setScenarios(list);
        if (list.length > 0) {
          setSelectedId(list[0].id);
        }
      })
      .catch(() => {
        setScenarios([]);
      });
  }, []);

  useEffect(() => {
    if (!isPlaying) {
      return;
    }
    let frameId;
    let last = performance.now();
    const loop = (now) => {
      const delta = now - last;
      last = now;
      // Cap effective update rate at ~60 FPS to avoid unnecessary CPU churn
      if (delta < 1000 / 75) {
        frameId = requestAnimationFrame(loop);
        return;
      }
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
    };
  }, [isPlaying, scenarios]);

  const activeScenario = scenarios.find((s) => s.id === selectedId) || null;

  const handleSelectScenario = (id) => {
    setSelectedId(id);
    setTimeMs(0);
  };

  return (
    <div className="app-root">
      <MapView scenario={activeScenario} timeMs={timeMs} durationMs={ANIMATION_DURATION_MS} />
      <ScenarioControls
        scenarios={scenarios}
        selectedId={selectedId}
        onSelect={handleSelectScenario}
        isPlaying={isPlaying}
        onTogglePlay={() => setIsPlaying((v) => !v)}
      />
    </div>
  );
}

export default App;

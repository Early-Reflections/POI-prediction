import React, { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { ScatterplotLayer, PathLayer } from "@deck.gl/layers";

const MAP_STYLE = "/styles/osm-streets.json";
const DEFAULT_ZOOM = 15; // closer, walk-scale view
const CLOSE_ZOOM = 16;   // tighter zoom when focusing a scenario
const DEFAULT_PITCH = 35;
const ZOOM_BIAS = -1;     // global offset applied to all auto zoom levels

// Increasing `ZOOM_BIAS` makes the map **more zoomed in** (closer).
// - `ZOOM_BIAS = +1` → everything one level **closer**.  
// - `ZOOM_BIAS = -1` → everything one level **farther out**.

function computeAnimationState(scenario, timeMs, durationMs) {
  if (!scenario || !Array.isArray(scenario.points) || scenario.points.length === 0) {
    return null;
  }
  const points = scenario.points;
  const phase = Math.max(0, Math.min(1, durationMs > 0 ? timeMs / durationMs : 0));
  const travelPortion = 0.6;
  const travelPhase = Math.min(phase / travelPortion, 1);
  const showPredictions = phase >= travelPortion;

  const idxFloat = travelPhase * (points.length - 1);
  const currentIndex = Math.max(0, Math.min(points.length - 1, Math.round(idxFloat)));
  const visibleCount = currentIndex + 1;
  const historyPoints = points.slice(0, visibleCount);
  const currentPoint = points[currentIndex];

  return {
    points,
    historyPoints,
    currentPoint,
    showPredictions,
    phase
  };
}

function computeScenarioViewport(scenario, zoomBias = ZOOM_BIAS) {
  if (!scenario) {
    return {
      center: [-74.006, 40.7128],
      zoom: DEFAULT_ZOOM
    };
  }
  const coords = [];
  if (Array.isArray(scenario.points)) {
    scenario.points.forEach((p) => {
      if (p && typeof p.lon === "number" && typeof p.lat === "number") {
        coords.push([p.lon, p.lat]);
      }
    });
  }
  if (Array.isArray(scenario.predictedPois)) {
    scenario.predictedPois.forEach((p) => {
      if (p && typeof p.lon === "number" && typeof p.lat === "number") {
        coords.push([p.lon, p.lat]);
      }
    });
  }
  if (coords.length === 0) {
    return {
      center: [-74.006, 40.7128],
      zoom: DEFAULT_ZOOM
    };
  }
  let minLng = coords[0][0];
  let maxLng = coords[0][0];
  let minLat = coords[0][1];
  let maxLat = coords[0][1];
  for (let i = 1; i < coords.length; i += 1) {
    const [lng, lat] = coords[i];
    if (lng < minLng) minLng = lng;
    if (lng > maxLng) maxLng = lng;
    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;
  }
  const spanLat = maxLat - minLat;
  const spanLng = maxLng - minLng;
  const span = Math.max(spanLat, spanLng);
  let zoom;
  if (span < 0.003) {
    zoom = 16;
  } else if (span < 0.01) {
    zoom = 15;
  } else if (span < 0.03) {
    zoom = 14;
  } else if (span < 0.1) {
    zoom = 13;
  } else if (span < 0.3) {
    zoom = 12;
  } else {
    zoom = 11;
  }
  zoom = Math.max(3, zoom + zoomBias);
  const centerLng = (minLng + maxLng) / 2;
  const centerLat = (minLat + maxLat) / 2;
  return {
    center: [centerLng, centerLat],
    zoom
  };
}

function buildLayers(scenario, timeMs, durationMs) {
  if (!scenario) {
    return [];
  }
  const state = computeAnimationState(scenario, timeMs, durationMs);
  if (!state) {
    return [];
  }
  const { historyPoints, currentPoint, showPredictions, phase } = state;
  const predictions = Array.isArray(scenario.predictedPois) ? scenario.predictedPois : [];
  if (historyPoints.length === 0) {
    return [];
  }
  const pathPoints = historyPoints.map((p) => [p.lon, p.lat]);

  const pathGlowLayer = new PathLayer({
    id: "trajectory-path-glow",
    data: [{ path: pathPoints }],
    getPath: (d) => d.path,
    getColor: () => [15, 23, 42, 200], // dark navy glow to separate from map
    getWidth: () => 10,
    widthUnits: "pixels",
    rounded: true
  });

  // Build per-segment data so we can apply a color gradient along the path
  const pathSegments = [];
  for (let i = 0; i < pathPoints.length - 1; i += 1) {
    const t = (pathPoints.length <= 2) ? 1 : i / (pathPoints.length - 2); // 0 (old) -> 1 (recent)
    pathSegments.push({ path: [pathPoints[i], pathPoints[i + 1]], _ageT: t });
  }

  const pathLayerAll = new PathLayer({
    id: "trajectory-path-all",
    data: pathSegments,
    getPath: (d) => d.path,
    getColor: (d) => {
      const tRaw = typeof d._ageT === "number" ? d._ageT : 0;
      const t = Math.max(0, Math.min(1, tRaw));
      // Multi-stop gradient: purple (old) -> violet -> blue -> aqua (new)
      const stops = [
        { t: 0.0, color: [88, 28, 135] },   // deep purple
        { t: 0.33, color: [139, 92, 246] }, // violet
        { t: 0.66, color: [59, 130, 246] }, // blue
        { t: 1.0, color: [45, 212, 191] }   // aqua/teal
      ];

      let c0 = stops[0].color;
      let c1 = stops[stops.length - 1].color;
      let localT = 0;
      for (let i = 0; i < stops.length - 1; i += 1) {
        const s0 = stops[i];
        const s1 = stops[i + 1];
        if (t >= s0.t && t <= s1.t) {
          const span = s1.t - s0.t || 1;
          localT = (t - s0.t) / span;
          c0 = s0.color;
          c1 = s1.color;
          break;
        }
      }
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * localT);
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * localT);
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * localT);
      return [r, g, b, 235];
    },
    getWidth: () => 4,
    widthUnits: "pixels",
    rounded: true
  });

  const historyDotsLayer = new ScatterplotLayer({
    id: "trajectory-history-dots",
    data: historyPoints.map((p, idx) => ({ ...p, _ageIdx: idx })),
    getPosition: (d) => [d.lon, d.lat],
    getRadius: () => 6,
    getFillColor: (d) => {
      const idx = typeof d._ageIdx === "number" ? d._ageIdx : 0;
      const maxIdx = Math.max(1, historyPoints.length - 1);
      const t = idx / maxIdx;
      const base = [59, 130, 246];
      const alpha = 80 + Math.round(140 * t);
      return [base[0], base[1], base[2], alpha];
    },
    radiusUnits: "pixels",
    stroked: false,
    parameters: {
      depthTest: false
    }
  });

  const userLayer = new ScatterplotLayer({
    id: "trajectory-user",
    data: [currentPoint],
    getPosition: (d) => [d.lon, d.lat],
    getRadius: () => 10,
    getFillColor: () => [59, 130, 246, 230],
    radiusUnits: "pixels",
    stroked: true,
    getLineColor: () => [248, 250, 252, 255],
    lineWidthUnits: "pixels",
    getLineWidth: () => 2,
    parameters: {
      depthTest: false
    }
  });

  const layers = [pathGlowLayer, pathLayerAll, historyDotsLayer, userLayer];

  if (showPredictions && predictions.length > 0) {
    const ranked = predictions
      .slice()
      .sort((a, b) => (b.score || 0) - (a.score || 0))
      .map((p, idx) => ({ ...p, rank: idx + 1 }));
    const maxRank = ranked.length;
    const pulse = 0.6 + 0.3 * Math.sin(phase * Math.PI * 2);
    const others = ranked.filter((d) => d.rank !== 1);
    const top1 = ranked.filter((d) => d.rank === 1);

    const predLayerOthers = new ScatterplotLayer({
      id: "predicted-pois-others",
      data: others,
      getPosition: (d) => [d.lon, d.lat],
      getRadius: (d) => {
        const rankWeight = maxRank - (d.rank || maxRank) + 1;
        return 10 + rankWeight * 3;
      },
      getFillColor: () => [16, 185, 129, 220],
      stroked: true,
      getLineColor: () => [15, 118, 110, 220],
      lineWidthUnits: "pixels",
      getLineWidth: () => 1.5,
      radiusUnits: "pixels",
      opacity: 0.95,
      parameters: {
        depthTest: false
      }
    });

    const predLayerTop1 = new ScatterplotLayer({
      id: "predicted-pois-top1",
      data: top1,
      getPosition: (d) => [d.lon, d.lat],
      getRadius: (d) => {
        const rankWeight = maxRank - (d.rank || maxRank) + 1;
        return 10 + rankWeight * 3;
      },
      getFillColor: () => [219, 39, 119, 255 * pulse],
      stroked: true,
      getLineColor: () => [76, 29, 149, 255],
      lineWidthUnits: "pixels",
      getLineWidth: () => 3,
      radiusUnits: "pixels",
      opacity: 0.98,
      parameters: {
        depthTest: false
      }
    });

    layers.push(predLayerOthers, predLayerTop1);
  }

  return layers;
}

function MapView({ scenario, timeMs, durationMs, zoomBias, datasetId, datasetVersion }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const overlayRef = useRef(null);
  const lastLoggedSecondRef = useRef(null);
  const velocityRef = useRef({ lng: 0, lat: 0 });
  const lastTimeMsRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) {
      return;
    }
    // Compute a per-scenario viewport so each trajectory gets an appropriate
    // zoom level and center based on its spatial extent (short walks vs. long trips).
    const viewport = computeScenarioViewport(scenario, zoomBias);
    console.log("[MapView] Initializing map", {
      datasetId,
      scenarioId: scenario && scenario.id,
      center: viewport.center,
      zoom: viewport.zoom,
      zoomBias
    });
    const initialCenter = viewport.center;

    // Create the MapLibre map.
    //  - container: DOM node that will host the canvas
    //  - style: MapLibre style JSON (here a local OSM streets style)
    //  - center / zoom: come from computeScenarioViewport so each trajectory
    //    gets a reasonable framing based on its geographic span
    //  - pitch: fixed tilt for a slightly 3D view
    //  - bearing: small rotation so the scene feels less flat
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: MAP_STYLE,
      center: initialCenter,
      zoom: viewport.zoom,
      pitch: DEFAULT_PITCH,
      bearing: -10
    });
    const overlay = new MapboxOverlay({
      interleaved: true,
      layers: []
    });
    map.addControl(overlay);
    mapRef.current = map;
    overlayRef.current = overlay;
    return () => {
      if (overlayRef.current && mapRef.current) {
        mapRef.current.removeControl(overlayRef.current);
      }
      if (mapRef.current) {
        mapRef.current.remove();
      }
      overlayRef.current = null;
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!mapRef.current || !scenario || !scenario.points || scenario.points.length === 0) {
      return;
    }
    const viewport = computeScenarioViewport(scenario, zoomBias);
    const map = mapRef.current;
    console.log("[MapView] jumpTo on scenario/zoomBias change", {
      datasetId,
      scenarioId: scenario && scenario.id,
      center: viewport.center,
      zoom: viewport.zoom,
      zoomBias
    });
    map.jumpTo({
      center: viewport.center,
      zoom: viewport.zoom,
      bearing: map.getBearing(),
      pitch: map.getPitch()
    });
    velocityRef.current = { lng: 0, lat: 0 };
    lastTimeMsRef.current = null;
  }, [scenario, zoomBias]);

  // When a new dataset has finished loading, teleport instantly
  // to the new viewport to avoid a long animated traverse across the globe.
  useEffect(() => {
    if (!mapRef.current || !scenario || !scenario.points || scenario.points.length === 0) {
      return;
    }
    const viewport = computeScenarioViewport(scenario, zoomBias);
    const map = mapRef.current;
    console.log("[MapView] jumpTo on dataset load", {
      datasetId,
      datasetVersion,
      scenarioId: scenario && scenario.id,
      center: viewport.center,
      zoom: viewport.zoom,
      zoomBias,
      currentCenter: map.getCenter()
    });
    map.jumpTo({
      center: viewport.center,
      zoom: viewport.zoom,
      bearing: map.getBearing(),
      pitch: map.getPitch()
    });
    velocityRef.current = { lng: 0, lat: 0 };
    lastTimeMsRef.current = null;
  }, [datasetVersion]);

  useEffect(() => {
    if (!mapRef.current || !scenario || !scenario.points || scenario.points.length === 0) {
      return;
    }
    const state = computeAnimationState(scenario, timeMs, durationMs);
    if (!state || !state.currentPoint) {
      return;
    }
    const currentPoint = state.currentPoint;
    const currentSecond = Math.floor(timeMs / 1000);
    if (lastLoggedSecondRef.current !== currentSecond) {
      lastLoggedSecondRef.current = currentSecond;
      console.log("[MapView] animation tick", {
        datasetId,
        scenarioId: scenario && scenario.id,
        timeMs,
        phase: state.phase,
        currentPoint: currentPoint
          ? { lon: currentPoint.lon, lat: currentPoint.lat }
          : null
      });
    }
    const viewport = computeScenarioViewport(scenario, zoomBias);
    const map = mapRef.current;

    const prevTimeMs = lastTimeMsRef.current;
    let dtMs;
    if (prevTimeMs == null) {
      dtMs = 16; // ~60 FPS default for first step
    } else {
      dtMs = timeMs - prevTimeMs;
    }
    lastTimeMsRef.current = timeMs;

    let dt = dtMs / 1000;
    if (!Number.isFinite(dt) || dt <= 0) {
      dt = 0.016;
    }
    dt = Math.min(Math.max(dt, 0.001), 0.05); // clamp between 1ms and 50ms

    const targetLng = currentPoint.lon;
    const targetLat = currentPoint.lat;
    const from = map.getCenter();

    const k = 3.0; // spring stiffness
    const c = 5.0; // damping

    const vel = velocityRef.current;
    const toTargetLng = targetLng - from.lng;
    const toTargetLat = targetLat - from.lat;

    const accLng = k * toTargetLng - c * vel.lng;
    const accLat = k * toTargetLat - c * vel.lat;

    const nextVelLng = vel.lng + accLng * dt;
    const nextVelLat = vel.lat + accLat * dt;

    const nextLng = from.lng + nextVelLng * dt;
    const nextLat = from.lat + nextVelLat * dt;

    velocityRef.current = { lng: nextVelLng, lat: nextVelLat };

    map.jumpTo({
      center: [nextLng, nextLat],
      zoom: viewport.zoom,
      bearing: map.getBearing(),
      pitch: map.getPitch()
    });
  }, [scenario, timeMs, durationMs, zoomBias]);

  useEffect(() => {
    if (!overlayRef.current) {
      return;
    }
    const layers = buildLayers(scenario, timeMs, durationMs);
    overlayRef.current.setProps({ layers });
  }, [scenario, timeMs, durationMs]);

  return <div ref={containerRef} className="map-root" />;
}

export default MapView;

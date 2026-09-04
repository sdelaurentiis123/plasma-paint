export function createPainter(api, styleConfig) {
  let seedValue = 0;
  return {
    reset(seed) {
      seedValue = seed;
      api.reset(seed);
    },
    renderFrame(frameFeatures, time, persistentState) {
      api.createPaper({ color: styleConfig.paper || "#f2ede2", grain: styleConfig.grain || 0.05 });
      api.setPalette({ colors: ["#233a4d", "#2f6f73", "#d6aa62", "#596b86", "#9e4b3f"] });
      api.fadeLayer({ layer: "tracked-pigment", retention: styleConfig.persistence || 0.86 });
      api.washRegion({ raster: "density", opacity: 0.18, bleed: styleConfig.bleed || 0.35 });
      for (const contour of frameFeatures.contours) {
        api.strokePath({
          points: contour.points,
          width: contour.sign > 0 ? 0.006 : 0.0035,
          opacity: contour.sign > 0 ? 0.34 : 0.22,
          pigment: contour.sign > 0 ? "positive_fluctuation" : "negative_fluctuation",
          source: "density_contour"
        });
      }
      for (const vector of frameFeatures.vectors.density_gradient) {
        const length = 0.008 + 0.018 * vector.magnitude;
        const endX = Math.max(0, Math.min(1, vector.x + vector.dx * length));
        const endZ = Math.max(0, Math.min(1, vector.z + vector.dz * length));
        api.dryBrushPath({
          points: [[vector.x, vector.z], [endX, endZ]],
          width: 0.0015 + 0.002 * vector.magnitude,
          opacity: 0.08 + 0.14 * vector.magnitude,
          pigment: "edge",
          source: "density_gradient_direction"
        });
      }
      for (const filament of frameFeatures.filaments) {
        const radius = Math.max(0.006, Math.min(0.12, 0.008 + 1.4 * Math.sqrt(filament.area_fraction)));
        api.dab({
          center: filament.centroid,
          radius,
          opacity: Math.max(0.12, Math.min(0.62, 0.18 + 0.07 * Math.abs(filament.peak))),
          pigment: filament.sign > 0 ? "positive_fluctuation" : "negative_fluctuation",
          trackId: filament.track_id,
          source: "filament"
        });
      }
      for (const event of frameFeatures.events) {
        if (event.type === "merge" || event.type === "birth") {
          const matching = frameFeatures.filaments.find((item) => item.track_id === (event.to_track_id ?? event.track_ids[0]));
          if (matching) {
            api.poolPigment({ center: matching.centroid, radius: 0.018, opacity: 0.20, source: event.type });
          }
        }
      }
      api.scatterGrain({ amount: 0.04, source: "paper_only", seed: seedValue + time });
      api.composite({ mode: "multiply" });
    }
  };
}


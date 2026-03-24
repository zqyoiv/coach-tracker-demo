const d = window.dashboardData || {};

function baseChartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "rgba(15,23,42,.95)",
        borderColor: "rgba(255,255,255,.12)",
        borderWidth: 1,
        titleColor: "#e5e7eb",
        bodyColor: "#e5e7eb",
      },
    },
    scales: {
      x: {
        grid: { color: "rgba(255,255,255,.06)" },
        ticks: { color: "rgba(229,231,235,.9)" },
      },
      y: {
        grid: { color: "rgba(255,255,255,.06)" },
        ticks: { color: "rgba(229,231,235,.9)" },
      },
    },
  };
}

function makeHorizontalBarChart(canvasId, labels, counts, color) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "events",
          data: counts,
          backgroundColor: color,
          borderColor: color,
          borderWidth: 1,
        },
      ],
    },
    options: {
      ...baseChartOptions(),
      indexAxis: "y",
      scales: {
        ...baseChartOptions().scales,
        x: {
          ...baseChartOptions().scales.x,
          beginAtZero: true,
        },
      },
    },
  });
  return chart;
}

function makeStackedHorizontalBars(canvasId, labels, byZone, zoneColors) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const zones = Object.keys(byZone || {});
  const datasets = zones.map((z, i) => ({
    label: z,
    data: byZone[z],
    backgroundColor: zoneColors[i % zoneColors.length],
    borderColor: zoneColors[i % zoneColors.length],
    borderWidth: 1,
  }));

  return new Chart(ctx, {
    type: "bar",
    data: { labels, datasets },
    options: {
      ...baseChartOptions(),
      indexAxis: "y",
      plugins: {
        ...baseChartOptions().plugins,
        legend: { display: true, labels: { color: "rgba(229,231,235,.95)" } },
      },
      scales: {
        ...baseChartOptions().scales,
        x: {
          ...baseChartOptions().scales.x,
          stacked: true,
          beginAtZero: true,
        },
        y: {
          ...baseChartOptions().scales.y,
          stacked: true,
        },
      },
    },
  });
}

function makeGroupedHorizontalBars(canvasId, labels, byZone, zoneColors) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  const zones = Object.keys(byZone || {});
  const datasets = zones.map((z, i) => ({
    label: z,
    data: byZone[z],
    backgroundColor: zoneColors[i % zoneColors.length],
    borderColor: zoneColors[i % zoneColors.length],
    borderWidth: 1,
  }));

  return new Chart(ctx, {
    type: "bar",
    data: { labels, datasets },
    options: {
      ...baseChartOptions(),
      indexAxis: "y",
      plugins: {
        ...baseChartOptions().plugins,
        legend: { display: true, labels: { color: "rgba(229,231,235,.95)" } },
      },
      scales: {
        ...baseChartOptions().scales,
        x: {
          ...baseChartOptions().scales.x,
          stacked: false,
          beginAtZero: true,
        },
        y: {
          ...baseChartOptions().scales.y,
          stacked: false,
        },
      },
    },
  });
}

function makeLineChart(canvasId, labels, counts) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  return new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "events",
          data: counts,
          borderColor: "#8B5CF6",
          backgroundColor: "rgba(139,92,246,.18)",
          borderWidth: 2,
          tension: 0.25,
          fill: true,
          pointRadius: 4,
          pointHoverRadius: 5,
        },
      ],
    },
    options: {
      ...baseChartOptions(),
      plugins: {
        ...baseChartOptions().plugins,
        legend: { display: false },
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,.06)" },
          ticks: { color: "rgba(229,231,235,.9)" },
        },
        y: {
          grid: { color: "rgba(255,255,255,.06)" },
          ticks: { color: "rgba(229,231,235,.9)" },
          beginAtZero: true,
        },
      },
    },
  });
}

function makeCameraBarChart(canvasId, labels, counts, color) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  return new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "events",
          data: counts,
          backgroundColor: color,
          borderColor: color,
          borderWidth: 1,
        },
      ],
    },
    options: {
      ...baseChartOptions(),
      plugins: {
        ...baseChartOptions().plugins,
        legend: { display: false },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: "rgba(229,231,235,.9)" },
        },
        y: {
          grid: { color: "rgba(255,255,255,.06)" },
          ticks: { color: "rgba(229,231,235,.9)" },
          beginAtZero: true,
        },
      },
    },
  });
}

function makeAvgDwellBarChart(canvasId, labels, values, color) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  return new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "avg dwell (sec)",
          data: values,
          backgroundColor: color,
          borderColor: color,
          borderWidth: 1,
        },
      ],
    },
    options: {
      ...baseChartOptions(),
      plugins: {
        ...baseChartOptions().plugins,
        legend: { display: false },
        tooltip: {
          ...baseChartOptions().plugins.tooltip,
          callbacks: {
            label: (ctx) => ` ${ctx.parsed.y.toFixed(2)} sec`,
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: "rgba(229,231,235,.9)" },
        },
        y: {
          grid: { color: "rgba(255,255,255,.06)" },
          ticks: { color: "rgba(229,231,235,.9)" },
          beginAtZero: true,
          title: {
            display: true,
            text: "seconds",
            color: "rgba(229,231,235,.9)",
          },
        },
      },
    },
  });
}

// Render charts
const dwellBins = d.dwell_bins || [];
const dwellAll = d.dwell_all_counts || [];
const dwellByZone = d.dwell_by_zone || {};
const dailyLabels = d.daily_labels || [];
const dailyCounts = d.daily_counts || [];
const cameraLabels = d.camera_labels || [];
const cameraCounts = d.camera_counts || [];
const avgDwellCameraLabels = d.avg_dwell_camera_labels || [];
const avgDwellCameraValues = d.avg_dwell_camera_values || [];

makeHorizontalBarChart("dwellAll", dwellBins, dwellAll, "#8B5CF6");
makeGroupedHorizontalBars("dwellByZone", dwellBins, dwellByZone, ["#8B5CF6", "#22C55E", "#60A5FA", "#F59E0B", "#EF4444", "#14B8A6"]);
makeLineChart("dailyDist", dailyLabels, dailyCounts);
makeCameraBarChart("cameraDist", cameraLabels, cameraCounts, "#8B5CF6");
makeAvgDwellBarChart("avgDwellCamera", avgDwellCameraLabels, avgDwellCameraValues, "#22C55E");


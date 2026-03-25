const byCamera = window.compareByCamera || {};
const labels = window.compareLabels || { primary: "Primary", baseline: "Baseline" };

const PURPLE = "#8B5CF6";
const ORANGE = "#F59E0B";
const BLUE = "#60A5FA";
const GREY = "rgba(148,163,184,.55)";

function baseOpts() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        labels: { color: "rgba(229,231,235,.95)" },
      },
      tooltip: {
        backgroundColor: "rgba(15,23,42,.95)",
        borderColor: "rgba(255,255,255,.12)",
        borderWidth: 1,
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
        beginAtZero: true,
      },
    },
  };
}

function renderCompare(canvasId, block) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (!block) return;

  const p = block.primary || {};
  const b = block.baseline || {};

  const chartLabels = ["Avg dwell (s)", "Total dwell (hr)", "Unique customers"];
  const primaryData = [p.avg_dwell ?? 0, p.total_dwell_hr ?? 0, p.customers ?? 0];
  const baselineData = [b.avg_dwell ?? 0, b.total_dwell_hr ?? 0, b.customers ?? 0];

  new Chart(ctx, {
    type: "bar",
    data: {
      labels: chartLabels,
      datasets: [
        {
          label: labels.primary || "Primary",
          data: primaryData,
          backgroundColor: [PURPLE, ORANGE, BLUE],
          borderColor: [PURPLE, ORANGE, BLUE],
          borderWidth: 1,
        },
        {
          label: labels.baseline || "Baseline",
          data: baselineData,
          backgroundColor: [GREY, GREY, GREY],
          borderColor: "rgba(148,163,184,.85)",
          borderWidth: 1,
        },
      ],
    },
    options: {
      ...baseOpts(),
      plugins: {
        ...baseOpts().plugins,
        tooltip: {
          ...baseOpts().plugins.tooltip,
          callbacks: {
            label: (ctx) => {
              const v = ctx.parsed.y;
              const i = ctx.dataIndex;
              if (i === 0) return ` ${v.toFixed(2)} s`;
              if (i === 1) return ` ${v.toFixed(2)} hr`;
              return ` ${v}`;
            },
          },
        },
      },
    },
  });
}

function renderAllCameras() {
  const keys = Object.keys(byCamera);
  const camNums = keys
    .map((k) => parseInt(k, 10))
    .filter((n) => !Number.isNaN(n))
    .sort((a, b) => a - b);

  for (const n of camNums) {
    const cam = byCamera[n] || byCamera[String(n)];
    if (!cam) continue;
    renderCompare(`compareAm-${n}`, cam.am);
    renderCompare(`comparePm-${n}`, cam.pm);
  }
}

renderAllCameras();

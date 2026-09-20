/**
 * Chart.js Visualization Helpers for Explainability, Dimensionality Reduction & Metrics
 */

const MedAiCharts = {
  instances: {},

  destroyChart(id) {
    if (this.instances[id]) {
      this.instances[id].destroy();
      delete this.instances[id];
    }
  },

  // 1. SHAP Waterfall / Feature Impact Bar Chart
  renderShapChart(canvasId, contributions) {
    this.destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const topItems = contributions.slice(0, 10).reverse();
    const labels = topItems.map(c => c.feature_name);
    const data = topItems.map(c => c.shap_value);
    const colors = topItems.map(c => c.shap_value > 0 ? "rgba(239, 68, 68, 0.85)" : "rgba(16, 185, 129, 0.85)");

    this.instances[canvasId] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "SHAP Impact on Risk",
          data: data,
          backgroundColor: colors,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (item) => `Impact: ${item.raw > 0 ? "+" : ""}${item.raw.toFixed(4)} log-odds`
            }
          }
        },
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.06)" },
            ticks: { color: "#94a3b8" }
          },
          y: {
            grid: { display: false },
            ticks: { color: "#f8fafc", font: { size: 11 } }
          }
        }
      }
    });
  },

  // 2. PCA & UMAP 2D Manifold Scatter Plot with Query Patient Embedding
  renderDimReductionScatter(canvasId, points, queryCoords, method = "PCA") {
    this.destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Group dataset points by risk category
    const lowPts = points.filter(p => p.risk_category === "Low Risk").map(p => ({ x: p.x, y: p.y }));
    const modPts = points.filter(p => p.risk_category === "Moderate Risk").map(p => ({ x: p.x, y: p.y }));
    const highPts = points.filter(p => p.risk_category === "High Risk").map(p => ({ x: p.x, y: p.y }));

    const datasets = [
      {
        label: "Low Risk Cohort",
        data: lowPts,
        backgroundColor: "rgba(16, 185, 129, 0.6)",
        pointRadius: 4,
        pointHoverRadius: 6
      },
      {
        label: "Moderate Risk Cohort",
        data: modPts,
        backgroundColor: "rgba(245, 158, 11, 0.6)",
        pointRadius: 4,
        pointHoverRadius: 6
      },
      {
        label: "High Risk Cohort",
        data: highPts,
        backgroundColor: "rgba(239, 68, 68, 0.6)",
        pointRadius: 4,
        pointHoverRadius: 6
      }
    ];

    // Overlay current query patient if provided
    if (queryCoords && queryCoords.length >= 2) {
      datasets.push({
        label: "★ Current Patient Embedding",
        data: [{ x: queryCoords[0], y: queryCoords[1] }],
        backgroundColor: "#38bdf8",
        borderColor: "#ffffff",
        borderWidth: 2,
        pointRadius: 10,
        pointHoverRadius: 12
      });
    }

    this.instances[canvasId] = new Chart(ctx, {
      type: "scatter",
      data: { datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "top",
            labels: { color: "#94a3b8", font: { size: 11 } }
          },
          tooltip: {
            callbacks: {
              label: (item) => `${item.dataset.label}: (${item.raw.x.toFixed(2)}, ${item.raw.y.toFixed(2)})`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: `${method} Component 1`, color: "#94a3b8" },
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#64748b" }
          },
          y: {
            title: { display: true, text: `${method} Component 2`, color: "#94a3b8" },
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#64748b" }
          }
        }
      }
    });
  },

  // 3. Model Benchmark Comparison Bar Chart
  renderBenchmarkComparison(canvasId, models) {
    this.destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const names = models.map(m => m.model_name);
    const f1s = models.map(m => (m.f1_macro * 100).toFixed(1));
    const rocs = models.map(m => (m.roc_auc_ovr * 100).toFixed(1));

    this.instances[canvasId] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: names,
        datasets: [
          {
            label: "Macro F1-Score (%)",
            data: f1s,
            backgroundColor: "rgba(59, 130, 246, 0.8)",
            borderRadius: 4
          },
          {
            label: "ROC-AUC OVR (%)",
            data: rocs,
            backgroundColor: "rgba(6, 182, 212, 0.8)",
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#94a3b8" } }
        },
        scales: {
          x: { ticks: { color: "#f8fafc", font: { size: 10 } }, grid: { display: false } },
          y: { min: 60, max: 100, ticks: { color: "#64748b" }, grid: { color: "rgba(255,255,255,0.05)" } }
        }
      }
    });
  },

  // 4. Feature Ablation Experiment Chart
  renderAblationChart(canvasId, ablationList) {
    this.destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const labels = ablationList.map(a => a.subset_name);
    const f1Scores = ablationList.map(a => a.macro_f1);

    this.instances[canvasId] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "Macro F1-Score",
          data: f1Scores,
          backgroundColor: f1Scores.map((s, idx) => idx === 0 ? "rgba(16, 185, 129, 0.85)" : "rgba(147, 197, 253, 0.7)"),
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { min: 0.4, max: 1.0, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
          y: { ticks: { color: "#f8fafc", font: { size: 11 } }, grid: { display: false } }
        }
      }
    });
  },

  // 5. Calibration Reliability Diagram
  renderCalibrationChart(canvasId, calib) {
    this.destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const pred = calib.prob_pred;
    const trueP = calib.prob_true;

    this.instances[canvasId] = new Chart(ctx, {
      type: "line",
      data: {
        labels: pred.map(p => (p * 100).toFixed(0) + "%"),
        datasets: [
          {
            label: "Model Calibration",
            data: trueP.map(p => p * 100),
            borderColor: "#3b82f6",
            backgroundColor: "rgba(59, 130, 246, 0.2)",
            borderWidth: 2,
            pointRadius: 6,
            fill: false
          },
          {
            label: "Perfect Calibration",
            data: pred.map(p => p * 100),
            borderColor: "#64748b",
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#94a3b8" } } },
        scales: {
          x: { title: { display: true, text: "Mean Predicted Probability", color: "#94a3b8" }, ticks: { color: "#64748b" } },
          y: { title: { display: true, text: "Empirical Fraction of Positives", color: "#94a3b8" }, min: 0, max: 100, ticks: { color: "#64748b" } }
        }
      }
    });
  }
};

window.MedAiCharts = MedAiCharts;

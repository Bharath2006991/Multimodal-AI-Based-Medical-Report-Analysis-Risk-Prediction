/**
 * Multimodal Medical AI Platform - Core Frontend Application Controller
 */

// Application State
const AppState = {
  activeView: "dashboard",
  currentUser: { email: "doctor@medai.org", full_name: "Dr. Sarah Lin, MD", role: "Clinician Researcher" },
  currentPatient: null,
  extractedItems: [],
  currentPrediction: null,
  benchmarks: null,
  researchData: null,
  pcaData: null,
  umapData: null,
  historyRecords: []
};

// Initialization on DOM ready
document.addEventListener("DOMContentLoaded", async () => {
  initNavigation();
  initFormCalculations();
  initUploadDropzones();
  initSampleDataLoaders();
  initPredictionTriggers();
  initHistoryActions();

  // Check auth or initialize demo session
  try {
    const me = await window.medAiApi.getDemoToken();
    AppState.currentUser = me;
    updateUserProfileUI();
  } catch (err) {
    console.warn("Using offline demo user state:", err);
  }

  // Preload initial dashboard metrics and benchmarks
  loadDashboardData();
});

// ==========================================
// 1. Navigation & View Switching
// ==========================================
function initNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  navItems.forEach(item => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      const targetView = item.getAttribute("data-view");
      switchView(targetView);
    });
  });
}

function switchView(viewName) {
  AppState.activeView = viewName;

  // Update nav active states
  document.querySelectorAll(".nav-item").forEach(el => {
    if (el.getAttribute("data-view") === viewName) {
      el.classList.add("active");
    } else {
      el.classList.remove("active");
    }
  });

  // Hide all sections, display target
  document.querySelectorAll(".view-section").forEach(sec => {
    sec.classList.remove("active");
  });

  const targetSec = document.getElementById(`view-${viewName}`);
  if (targetSec) {
    targetSec.classList.add("active");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  // Update Page Title
  const titles = {
    dashboard: "Clinical Intelligence Dashboard",
    "patient-intake": "Multimodal Patient Intake & Clinical Parameters",
    "report-upload": "Medical Document Processing & OCR",
    prediction: "Multimodal Risk Prediction & Clinical Insights",
    explainability: "Explainable AI (SHAP & Manifold Projections)",
    "dim-reduction": "Dimensionality Reduction: PCA vs UMAP",
    benchmarking: "Machine Learning Model Benchmark Suite",
    research: "Research Experiments: Ablation & Calibration",
    history: "Analysis Audit Logs & Historical Records",
    about: "System Architecture, Methodology & Research Disclaimer"
  };
  const titleEl = document.getElementById("current-page-title");
  if (titleEl && titles[viewName]) {
    titleEl.textContent = titles[viewName];
  }

  // Contextual view refresh
  if (viewName === "benchmarking" && !AppState.benchmarks) {
    loadBenchmarkingData();
  } else if (viewName === "dim-reduction") {
    loadDimReductionData();
  } else if (viewName === "research" && !AppState.researchData) {
    loadResearchData();
  } else if (viewName === "history") {
    loadHistoryData();
  }
}

function updateUserProfileUI() {
  const nameEl = document.getElementById("user-display-name");
  const roleEl = document.getElementById("user-display-role");
  if (nameEl) nameEl.textContent = AppState.currentUser.user_name || AppState.currentUser.full_name || "Dr. Clinician";
  if (roleEl) roleEl.textContent = AppState.currentUser.role || "Clinician Researcher";
}

// ==========================================
// 2. BMI Calculation & Form Handling
// ==========================================
function initFormCalculations() {
  const heightInput = document.getElementById("form-height");
  const weightInput = document.getElementById("form-weight");
  const bmiInput = document.getElementById("form-bmi");

  function calcBmi() {
    const h = parseFloat(heightInput.value);
    const w = parseFloat(weightInput.value);
    if (h > 50 && w > 20) {
      const bmi = (w / ((h / 100) ** 2)).toFixed(1);
      bmiInput.value = bmi;
    }
  }

  if (heightInput && weightInput) {
    heightInput.addEventListener("input", calcBmi);
    weightInput.addEventListener("input", calcBmi);
  }
}

function getFormData() {
  return {
    demographics: {
      patient_id: document.getElementById("form-patient-id").value || "PAT-CURRENT",
      age: parseFloat(document.getElementById("form-age").value) || 52,
      sex: document.getElementById("form-sex").value || "male",
      height_cm: parseFloat(document.getElementById("form-height").value) || 175,
      weight_kg: parseFloat(document.getElementById("form-weight").value) || 82,
      bmi: parseFloat(document.getElementById("form-bmi").value) || 26.8,
      smoking_status: document.getElementById("form-smoking").value || "never",
      physical_activity: document.getElementById("form-activity").value || "moderate",
      family_history_cad: document.getElementById("form-family-cad").checked
    },
    vitals_labs: {
      systolic_bp: parseFloat(document.getElementById("form-sbp").value) || 135,
      diastolic_bp: parseFloat(document.getElementById("form-dbp").value) || 85,
      heart_rate: parseFloat(document.getElementById("form-hr").value) || 74,
      fasting_glucose: parseFloat(document.getElementById("form-glucose").value) || 110,
      hba1c: parseFloat(document.getElementById("form-hba1c").value) || 5.8,
      total_cholesterol: parseFloat(document.getElementById("form-tc").value) || 220,
      hdl_cholesterol: parseFloat(document.getElementById("form-hdl").value) || 44,
      ldl_cholesterol: parseFloat(document.getElementById("form-ldl").value) || 140,
      triglycerides: parseFloat(document.getElementById("form-tg").value) || 180,
      hemoglobin: parseFloat(document.getElementById("form-hb").value) || 14.5
    },
    symptoms: {
      chest_pain: document.getElementById("form-sym-chest-pain").checked,
      shortness_of_breath: document.getElementById("form-sym-sob").checked,
      fatigue: document.getElementById("form-sym-fatigue").checked,
      dizziness: document.getElementById("form-sym-dizziness").checked,
      palpitations: document.getElementById("form-sym-palpitations").checked,
      notes: document.getElementById("form-clinical-notes").value || ""
    }
  };
}

function setFormData(profile) {
  const d = profile.demographics;
  const v = profile.vitals_labs;
  const s = profile.symptoms;

  document.getElementById("form-patient-id").value = d.patient_id;
  document.getElementById("form-age").value = d.age;
  document.getElementById("form-sex").value = d.sex;
  document.getElementById("form-height").value = d.height_cm;
  document.getElementById("form-weight").value = d.weight_kg;
  document.getElementById("form-bmi").value = d.bmi;
  document.getElementById("form-smoking").value = d.smoking_status;
  document.getElementById("form-activity").value = d.physical_activity;
  document.getElementById("form-family-cad").checked = !!d.family_history_cad;

  document.getElementById("form-sbp").value = v.systolic_bp;
  document.getElementById("form-dbp").value = v.diastolic_bp;
  document.getElementById("form-hr").value = v.heart_rate;
  document.getElementById("form-glucose").value = v.fasting_glucose;
  document.getElementById("form-hba1c").value = v.hba1c;
  document.getElementById("form-tc").value = v.total_cholesterol;
  document.getElementById("form-hdl").value = v.hdl_cholesterol;
  document.getElementById("form-ldl").value = v.ldl_cholesterol;
  document.getElementById("form-tg").value = v.triglycerides;
  document.getElementById("form-hb").value = v.hemoglobin;

  document.getElementById("form-sym-chest-pain").checked = !!s.chest_pain;
  document.getElementById("form-sym-sob").checked = !!s.shortness_of_breath;
  document.getElementById("form-sym-fatigue").checked = !!s.fatigue;
  document.getElementById("form-sym-dizziness").checked = !!s.dizziness;
  document.getElementById("form-sym-palpitations").checked = !!s.palpitations;
  document.getElementById("form-clinical-notes").value = s.notes || "";
}

// ==========================================
// 3. Sample Patient Profiles Loaders
// ==========================================
let cachedSampleProfiles = [];

async function initSampleDataLoaders() {
  try {
    const data = await window.medAiApi.getSampleData();
    cachedSampleProfiles = data.profiles || [];
  } catch (err) {
    console.warn("Error loading sample profiles:", err);
  }

  // Attach buttons
  const btnLow = document.getElementById("btn-sample-low");
  const btnMod = document.getElementById("btn-sample-mod");
  const btnHigh = document.getElementById("btn-sample-high");

  if (btnLow) btnLow.addEventListener("click", () => loadSampleIndex(0));
  if (btnMod) btnMod.addEventListener("click", () => loadSampleIndex(1));
  if (btnHigh) btnHigh.addEventListener("click", () => loadSampleIndex(2));

  // Quick Action Buttons on Dashboard
  const dashBtnLow = document.getElementById("dash-btn-low");
  const dashBtnHigh = document.getElementById("dash-btn-high");
  if (dashBtnLow) {
    dashBtnLow.addEventListener("click", () => {
      loadSampleIndex(0);
      switchView("patient-intake");
    });
  }
  if (dashBtnHigh) {
    dashBtnHigh.addEventListener("click", () => {
      loadSampleIndex(2);
      switchView("patient-intake");
    });
  }
}

function loadSampleIndex(idx) {
  if (cachedSampleProfiles[idx]) {
    setFormData(cachedSampleProfiles[idx].patient_data);
    showNotification(`Loaded ${cachedSampleProfiles[idx].profile_name}`, "success");
  }
}

// ==========================================
// 4. File Upload & OCR Extraction
// ==========================================
function initUploadDropzones() {
  // PDF Dropzone
  setupDropzone(
    "dropzone-pdf",
    "file-input-pdf",
    async (file) => {
      const sex = document.getElementById("form-sex").value || "male";
      showUploadProgress("Analyzing PDF report text and table structures...");
      try {
        const res = await window.medAiApi.uploadPdf(file, sex);
        handleExtractionResult(res);
      } catch (err) {
        showNotification(err.message, "danger");
      } finally {
        hideUploadProgress();
      }
    }
  );

  // Image Dropzone
  setupDropzone(
    "dropzone-img",
    "file-input-img",
    async (file) => {
      const sex = document.getElementById("form-sex").value || "male";
      showUploadProgress("Executing OpenCV filtering and EasyOCR recognition...");
      try {
        const res = await window.medAiApi.uploadImage(file, sex);
        handleExtractionResult(res);
      } catch (err) {
        showNotification(err.message, "danger");
      } finally {
        hideUploadProgress();
      }
    }
  );

  // Raw Text Extraction button
  const btnExtractText = document.getElementById("btn-extract-raw-text");
  if (btnExtractText) {
    btnExtractText.addEventListener("click", async () => {
      const text = document.getElementById("raw-clinical-text-input").value;
      if (!text.trim()) {
        showNotification("Please enter clinical notes before extracting", "warning");
        return;
      }
      try {
        const sex = document.getElementById("form-sex").value || "male";
        const res = await window.medAiApi.extractText(text, sex);
        handleExtractionResult(res);
      } catch (err) {
        showNotification(err.message, "danger");
      }
    });
  }

  // 1-Click Load Sample PDF & Image
  const btnLoadSamplePdf = document.getElementById("btn-load-sample-pdf");
  if (btnLoadSamplePdf) {
    btnLoadSamplePdf.addEventListener("click", async () => {
      showUploadProgress("Loading sample metabolic panel PDF from server...");
      try {
        const res = await fetch("/api/sample-data/pdf");
        const blob = await res.blob();
        const file = new File([blob], "sample_report_metabolic.pdf", { type: "application/pdf" });
        const data = await window.medAiApi.uploadPdf(file, "male");
        handleExtractionResult(data);
      } catch (err) {
        showNotification("Failed to load sample PDF: " + err.message, "danger");
      } finally {
        hideUploadProgress();
      }
    });
  }

  const btnLoadSampleImg = document.getElementById("btn-load-sample-img");
  if (btnLoadSampleImg) {
    btnLoadSampleImg.addEventListener("click", async () => {
      showUploadProgress("Loading and running OCR on sample laboratory image...");
      try {
        const res = await fetch("/api/sample-data/image");
        const blob = await res.blob();
        const file = new File([blob], "sample_scanned_report.png", { type: "image/png" });
        const data = await window.medAiApi.uploadImage(file, "female");
        handleExtractionResult(data);
      } catch (err) {
        showNotification("Failed to load sample image: " + err.message, "danger");
      } finally {
        hideUploadProgress();
      }
    });
  }

  // Apply Extracted Items to Patient Form
  const btnApplyExtracted = document.getElementById("btn-apply-extracted-to-form");
  if (btnApplyExtracted) {
    btnApplyExtracted.addEventListener("click", () => {
      applyExtractedToForm();
      showNotification("Extracted measurements mapped to patient intake form!", "success");
      switchView("patient-intake");
    });
  }
}

function setupDropzone(zoneId, inputId, onFileSelected) {
  const zone = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  if (!zone || !input) return;

  zone.addEventListener("click", () => input.click());

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("dragover");
  });

  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));

  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      onFileSelected(e.dataTransfer.files[0]);
    }
  });

  input.addEventListener("change", (e) => {
    if (e.target.files.length) {
      onFileSelected(e.target.files[0]);
    }
  });
}

function handleExtractionResult(result) {
  AppState.extractedItems = result.extracted_measurements || [];

  // Show raw extracted text preview
  const rawBox = document.getElementById("extracted-raw-text-box");
  if (rawBox) rawBox.textContent = result.extracted_text;

  // Show status badges for symptoms
  const symDiv = document.getElementById("extracted-symptoms-container");
  if (symDiv) {
    let html = "";
    (result.detected_symptoms || []).forEach(s => {
      html += `<span class="badge badge-danger">Detected: ${s.replace(/_/g, ' ')}</span> `;
    });
    (result.negated_symptoms || []).forEach(s => {
      html += `<span class="badge badge-success">Negated: ${s.replace(/_/g, ' ')}</span> `;
    });
    if (!html) html = "<span style='color:#64748b; font-size:12px;'>No symptoms detected in text.</span>";
    symDiv.innerHTML = html;
  }

  // Populate Verification Table
  renderExtractionTable(AppState.extractedItems);

  // Reveal results container
  const resContainer = document.getElementById("extraction-results-wrapper");
  if (resContainer) resContainer.style.display = "block";

  showNotification(`Extracted ${AppState.extractedItems.length} laboratory biomarkers with ${(result.confidence_score * 100).toFixed(0)}% confidence`, "success");
}

function renderExtractionTable(items) {
  const tbody = document.getElementById("extraction-table-body");
  if (!tbody) return;

  if (!items || items.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:#64748b; padding:20px;">No structured lab values extracted.</td></tr>`;
    return;
  }

  tbody.innerHTML = items.map((item, idx) => {
    const badgeClass = `badge-${item.status.toLowerCase()}`;
    return `
      <tr>
        <td><strong>${item.display_name}</strong><br><small style="color:#64748b;">${item.category}</small></td>
        <td>
          <input type="number" step="0.1" class="form-control" style="width:110px; padding:4px 8px;"
            value="${item.value}" onchange="updateExtractedValue(${idx}, this.value)">
        </td>
        <td>${item.unit}</td>
        <td style="color:#94a3b8;">${item.reference_range}</td>
        <td><span class="badge ${badgeClass}">${item.status.replace(/_/g, ' ')}</span></td>
        <td style="font-size:12px; color:#cbd5e1;">${item.interpretation}</td>
      </tr>
    `;
  }).join("");
}

window.updateExtractedValue = function(idx, val) {
  if (AppState.extractedItems[idx]) {
    AppState.extractedItems[idx].value = parseFloat(val);
  }
};

function applyExtractedToForm() {
  const mapInputs = {
    systolic_bp: "form-sbp",
    diastolic_bp: "form-dbp",
    heart_rate: "form-hr",
    fasting_glucose: "form-glucose",
    hba1c: "form-hba1c",
    total_cholesterol: "form-tc",
    hdl_cholesterol: "form-hdl",
    ldl_cholesterol: "form-ldl",
    triglycerides: "form-tg",
    hemoglobin: "form-hb",
    bmi: "form-bmi"
  };

  AppState.extractedItems.forEach(item => {
    const inputId = mapInputs[item.test_name];
    if (inputId) {
      const el = document.getElementById(inputId);
      if (el) el.value = item.value;
    }
  });
}

function showUploadProgress(msg) {
  const bar = document.getElementById("upload-progress-card");
  const text = document.getElementById("upload-progress-text");
  if (bar) bar.style.display = "block";
  if (text) text.textContent = msg;
}

function hideUploadProgress() {
  const bar = document.getElementById("upload-progress-card");
  if (bar) bar.style.display = "none";
}

// ==========================================
// 5. Prediction & Multimodal Inference
// ==========================================
function initPredictionTriggers() {
  const btnPredict = document.getElementById("btn-run-prediction");
  if (btnPredict) {
    btnPredict.addEventListener("click", () => executePrediction());
  }

  const btnConsensus = document.getElementById("btn-run-consensus");
  if (btnConsensus) {
    btnConsensus.addEventListener("click", () => executeConsensusAnalysis());
  }
}

async function executePrediction() {
  const patientData = getFormData();
  const modelSelect = document.getElementById("select-model-choice");
  const selectedModel = modelSelect ? modelSelect.value : "XGBoost";

  showNotification("Executing Multimodal Preprocessing & ML Ensemble...", "info");

  try {
    const payload = {
      patient_data: patientData,
      extracted_items: AppState.extractedItems,
      model_name: selectedModel
    };

    const res = await window.medAiApi.predictRisk(payload);
    AppState.currentPrediction = res;
    displayPredictionResults(res);
    switchView("prediction");
    showNotification("Clinical Risk Prediction complete!", "success");
    loadDashboardData(); // Refresh recent counts
  } catch (err) {
    showNotification("Prediction failed: " + err.message, "danger");
  }
}

function displayPredictionResults(pred) {
  // Hero Card
  const heroCard = document.getElementById("risk-hero-card");
  const catBadge = document.getElementById("pred-risk-category");
  const scoreVal = document.getElementById("pred-risk-score");
  const confVal = document.getElementById("pred-confidence");
  const modelUsed = document.getElementById("pred-model-used");

  const tier = pred.risk_category === "High Risk" ? "high" : (pred.risk_category === "Moderate Risk" ? "moderate" : "low");

  if (heroCard) {
    heroCard.className = `risk-hero-card ${tier}`;
  }
  if (catBadge) {
    catBadge.textContent = pred.risk_category.toUpperCase();
    catBadge.className = `risk-hero-badge ${tier}`;
  }
  if (scoreVal) scoreVal.textContent = `${(pred.risk_score * 100).toFixed(1)}%`;
  if (confVal) confVal.textContent = `${(pred.confidence * 100).toFixed(1)}%`;
  if (modelUsed) modelUsed.textContent = pred.model_used;

  // Probability Meter Bars
  const pLow = (pred.probabilities["Low Risk"] * 100).toFixed(1);
  const pMod = (pred.probabilities["Moderate Risk"] * 100).toFixed(1);
  const pHigh = (pred.probabilities["High Risk"] * 100).toFixed(1);

  const fillLow = document.getElementById("prob-fill-low");
  const fillMod = document.getElementById("prob-fill-mod");
  const fillHigh = document.getElementById("prob-fill-high");

  if (fillLow) fillLow.style.width = `${pLow}%`;
  if (fillMod) fillMod.style.width = `${pMod}%`;
  if (fillHigh) fillHigh.style.width = `${pHigh}%`;

  const txtLow = document.getElementById("prob-text-low");
  const txtMod = document.getElementById("prob-text-mod");
  const txtHigh = document.getElementById("prob-text-high");

  if (txtLow) txtLow.textContent = `${pLow}%`;
  if (txtMod) txtMod.textContent = `${pMod}%`;
  if (txtHigh) txtHigh.textContent = `${pHigh}%`;

  // Abnormal Findings Tags
  const findingsContainer = document.getElementById("pred-abnormal-findings");
  if (findingsContainer) {
    if (pred.abnormal_findings.length === 0) {
      findingsContainer.innerHTML = `<span style="color:#10b981; font-weight:500;">✓ All evaluated laboratory analytes within optimal reference intervals.</span>`;
    } else {
      findingsContainer.innerHTML = pred.abnormal_findings.map(item => `
        <div style="background-color:rgba(255,255,255,0.03); border:1px solid var(--border-subtle); border-radius:8px; padding:10px 14px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;">
          <div>
            <strong>${item.display_name}</strong> (${item.raw_value_str})
            <div style="font-size:12px; color:#94a3b8;">Reference: ${item.reference_range}</div>
          </div>
          <span class="badge badge-${item.status.toLowerCase()}">${item.status.replace(/_/g, ' ')}</span>
        </div>
      `).join("");
    }
  }

  // Clinical Explanations
  const clinicianEl = document.getElementById("pred-clinician-narrative");
  const patientEl = document.getElementById("pred-patient-narrative");
  if (clinicianEl) clinicianEl.textContent = pred.clinical_explanation;
  if (patientEl) patientEl.textContent = pred.patient_friendly_summary;

  // Render SHAP Waterfall Chart
  if (pred.top_contributing_features && pred.top_contributing_features.length > 0) {
    window.MedAiCharts.renderShapChart("chart-shap-prediction", pred.top_contributing_features);
  }

  // Render PCA & UMAP Projections
  loadDimReductionData(pred.pca_coordinates, pred.umap_coordinates);
}

async function executeConsensusAnalysis() {
  const patientData = getFormData();
  showNotification("Evaluating consensus across all 5 benchmarked models...", "info");

  try {
    const payload = {
      patient_data: patientData,
      extracted_items: AppState.extractedItems
    };
    const data = await window.medAiApi.analyzeConsensus(payload);
    
    // Display Modal
    const modal = document.getElementById("consensus-modal");
    const body = document.getElementById("consensus-modal-body");

    const cons = data.model_consensus;
    body.innerHTML = `
      <div style="margin-bottom:16px;">
        <div style="font-size:13px; color:#94a3b8;">Consensus Risk Category:</div>
        <div style="font-size:22px; font-weight:700; color:var(--accent-cyan);">${cons.consensus_category.toUpperCase()}</div>
        <div style="font-size:12px; color:#64748b;">Inter-Model Concordance: ${(cons.concordance_rate * 100).toFixed(0)}% agreement across ${cons.models_evaluated} architectures</div>
      </div>
      <table class="clinical-table" style="font-size:12px;">
        <thead>
          <tr><th>Model Architecture</th><th>Predicted Tier</th><th>Risk Score</th><th>Confidence</th></tr>
        </thead>
        <tbody>
          ${cons.evaluations.map(e => `
            <tr>
              <td><strong>${e.model_name}</strong></td>
              <td><span class="badge badge-${e.risk_category === 'High Risk' ? 'high' : (e.risk_category === 'Moderate Risk' ? 'moderate' : 'low')}">${e.risk_category}</span></td>
              <td>${(e.risk_score * 100).toFixed(1)}%</td>
              <td>${(e.confidence * 100).toFixed(1)}%</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;

    modal.classList.add("open");
  } catch (err) {
    showNotification("Consensus evaluation failed: " + err.message, "danger");
  }
}

// ==========================================
// 6. Dimensionality Reduction (PCA & UMAP)
// ==========================================
async function loadDimReductionData(queryPca, queryUmap) {
  try {
    if (!AppState.pcaData) {
      AppState.pcaData = await window.medAiApi.getDimReduction("PCA");
    }
    if (!AppState.umapData) {
      AppState.umapData = await window.medAiApi.getDimReduction("UMAP");
    }

    const pcaQuery = queryPca || (AppState.currentPrediction ? AppState.currentPrediction.pca_coordinates : null);
    const umapQuery = queryUmap || (AppState.currentPrediction ? AppState.currentPrediction.umap_coordinates : null);

    window.MedAiCharts.renderDimReductionScatter("chart-pca-manifold", AppState.pcaData.points, pcaQuery, "PCA");
    window.MedAiCharts.renderDimReductionScatter("chart-umap-manifold", AppState.umapData.points, umapQuery, "UMAP");

    const pcaExpEl = document.getElementById("pca-variance-badge");
    if (pcaExpEl && AppState.pcaData.cumulative_variance) {
      pcaExpEl.textContent = `Cumulative Explained Variance: ${(AppState.pcaData.cumulative_variance * 100).toFixed(1)}%`;
    }
  } catch (err) {
    console.warn("Dimensionality reduction load error:", err);
  }
}

// ==========================================
// 7. Benchmarking & Performance Suite
// ==========================================
async function loadBenchmarkingData() {
  try {
    const data = await window.medAiApi.getModelMetrics();
    AppState.benchmarks = data;

    // Render Table
    const tbody = document.getElementById("benchmark-table-body");
    if (tbody) {
      tbody.innerHTML = data.models.map(m => {
        const isBest = m.model_name === data.best_model_name;
        return `
          <tr style="${isBest ? 'background-color:rgba(59,130,246,0.08); font-weight:600;' : ''}">
            <td>${m.model_name} ${isBest ? '<span class="badge badge-info" style="margin-left:6px;">Champion ★</span>' : ''}</td>
            <td>${(m.accuracy * 100).toFixed(1)}%</td>
            <td>${(m.f1_macro * 100).toFixed(1)}%</td>
            <td>${(m.precision_macro * 100).toFixed(1)}%</td>
            <td>${(m.recall_macro * 100).toFixed(1)}%</td>
            <td>${(m.roc_auc_ovr * 100).toFixed(1)}%</td>
            <td>${m.cv_mean.toFixed(3)} ± ${m.cv_std.toFixed(3)}</td>
            <td>${m.brier_score.toFixed(3)}</td>
            <td>${m.prediction_latency_ms.toFixed(2)} ms</td>
          </tr>
        `;
      }).join("");
    }

    // Render Comparison Chart
    window.MedAiCharts.renderBenchmarkComparison("chart-benchmark-bars", data.models);
  } catch (err) {
    console.warn("Error loading benchmarks:", err);
  }
}

// ==========================================
// 8. Research Experiments
// ==========================================
async function loadResearchData() {
  try {
    const data = await window.medAiApi.getResearchExperiments();
    AppState.researchData = data;

    // Render Ablation Chart
    if (data.feature_ablation) {
      window.MedAiCharts.renderAblationChart("chart-research-ablation", data.feature_ablation);
    }

    // Render Calibration Reliability
    if (data.calibration) {
      window.MedAiCharts.renderCalibrationChart("chart-research-calibration", data.calibration);
      const eceEl = document.getElementById("research-ece-val");
      if (eceEl) eceEl.textContent = `Expected Calibration Error (ECE): ${data.calibration.expected_calibration_error}`;
    }
  } catch (err) {
    console.warn("Error loading research experiments:", err);
  }
}

// ==========================================
// 9. History Audit Logs
// ==========================================
async function loadHistoryData() {
  try {
    const data = await window.medAiApi.getHistory();
    AppState.historyRecords = data;

    const tbody = document.getElementById("history-table-body");
    if (!tbody) return;

    if (data.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:#64748b; padding:24px;">No historical analyses recorded yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = data.map(r => {
      const tier = r.risk_category === "High Risk" ? "high" : (r.risk_category === "Moderate Risk" ? "moderate" : "low");
      return `
        <tr>
          <td><span style="font-family:var(--font-mono); font-size:12px;">${r.id}</span></td>
          <td style="color:#94a3b8; font-size:12px;">${r.timestamp}</td>
          <td><strong>${r.patient_id}</strong> (${r.age}y / ${r.sex})</td>
          <td><span class="badge badge-${tier}">${r.risk_category}</span></td>
          <td>${(r.risk_score * 100).toFixed(1)}%</td>
          <td>${r.model_used}</td>
          <td style="font-size:12px; color:#cbd5e1; max-width:300px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${r.summary}</td>
        </tr>
      `;
    }).join("");
  } catch (err) {
    console.warn("Error loading history:", err);
  }
}

function initHistoryActions() {
  const btnClear = document.getElementById("btn-clear-history");
  if (btnClear) {
    btnClear.addEventListener("click", async () => {
      if (confirm("Clear all historical analysis logs?")) {
        await window.medAiApi.clearHistory();
        loadHistoryData();
        showNotification("Audit history cleared", "info");
      }
    });
  }

  const btnExport = document.getElementById("btn-export-history");
  if (btnExport) {
    btnExport.addEventListener("click", () => {
      const json = JSON.stringify(AppState.historyRecords, null, 2);
      const blob = new Blob([json], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `medical_ai_history_${new Date().toISOString().slice(0,10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }
}

// ==========================================
// 10. Dashboard Overview Stats
// ==========================================
async function loadDashboardData() {
  try {
    const history = await window.medAiApi.getHistory();
    const totalCountEl = document.getElementById("dash-total-analyses");
    if (totalCountEl) totalCountEl.textContent = history.length;

    if (history.length > 0) {
      const highCount = history.filter(h => h.risk_category === "High Risk").length;
      const highPctEl = document.getElementById("dash-high-risk-pct");
      if (highPctEl) highPctEl.textContent = `${((highCount / history.length) * 100).toFixed(0)}% High Risk`;

      // Render recent 4 history in dashboard
      const dashRecent = document.getElementById("dash-recent-stream");
      if (dashRecent) {
        dashRecent.innerHTML = history.slice(0, 4).map(h => `
          <div style="display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid var(--border-subtle);">
            <div>
              <strong>${h.patient_id}</strong> (${h.age}y, ${h.sex})
              <div style="font-size:11px; color:#64748b;">${h.timestamp} • ${h.model_used}</div>
            </div>
            <span class="badge badge-${h.risk_category === 'High Risk' ? 'high' : (h.risk_category === 'Moderate Risk' ? 'moderate' : 'low')}">${h.risk_category}</span>
          </div>
        `).join("");
      }
    }
  } catch (err) {
    console.warn("Dashboard overview error:", err);
  }
}

// Notification Toasts
function showNotification(message, type = "info") {
  const container = document.getElementById("notification-toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `alert alert-${type}`;
  toast.style.boxShadow = "var(--shadow-md)";
  toast.style.animation = "fadeIn 0.2s ease";
  toast.innerHTML = `<span>${message}</span>`;

  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

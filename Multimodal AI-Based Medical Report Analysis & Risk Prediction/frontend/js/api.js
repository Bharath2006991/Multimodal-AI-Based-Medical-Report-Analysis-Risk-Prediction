/**
 * Frontend API Client for Multimodal Medical AI Platform
 */

const API_BASE = window.location.origin;

class MedAiApiClient {
  constructor() {
    this.token = localStorage.getItem("medai_token") || null;
  }

  setToken(token) {
    this.token = token;
    if (token) {
      localStorage.setItem("medai_token", token);
    } else {
      localStorage.removeItem("medai_token");
    }
  }

  getHeaders(isJson = true) {
    const headers = {};
    if (isJson) {
      headers["Content-Type"] = "application/json";
    }
    if (this.token) {
      headers["Authorization"] = `Bearer ${this.token}`;
    }
    return headers;
  }

  async login(email, password) {
    const res = await fetch(`${API_BASE}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Authentication failed");
    }
    const data = await res.json();
    this.setToken(data.access_token);
    return data;
  }

  async getDemoToken() {
    const res = await fetch(`${API_BASE}/api/auth/demo-token`, { method: "POST" });
    const data = await res.json();
    this.setToken(data.access_token);
    return data;
  }

  async uploadPdf(file, sex = "male") {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("patient_sex", sex);

    const res = await fetch(`${API_BASE}/api/upload/pdf`, {
      method: "POST",
      headers: this.getHeaders(false),
      body: formData
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "PDF upload failed");
    }
    return await res.json();
  }

  async uploadImage(file, sex = "male") {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("patient_sex", sex);

    const res = await fetch(`${API_BASE}/api/upload/image`, {
      method: "POST",
      headers: this.getHeaders(false),
      body: formData
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Image upload failed");
    }
    return await res.json();
  }

  async extractText(raw_text, sex = "male") {
    const res = await fetch(`${API_BASE}/api/extract?patient_sex=${encodeURIComponent(sex)}`, {
      method: "POST",
      headers: this.getHeaders(true),
      body: JSON.stringify({ raw_text })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Text extraction failed");
    }
    return await res.json();
  }

  async predictRisk(payload) {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: this.getHeaders(true),
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction failed");
    }
    return await res.json();
  }

  async analyzeConsensus(payload) {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: this.getHeaders(true),
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Consensus analysis failed");
    }
    return await res.json();
  }

  async getModelMetrics() {
    const res = await fetch(`${API_BASE}/api/model/metrics`, { headers: this.getHeaders() });
    if (!res.ok) throw new Error("Failed to fetch model metrics");
    return await res.json();
  }

  async getModelFeatures() {
    const res = await fetch(`${API_BASE}/api/model/features`, { headers: this.getHeaders() });
    if (!res.ok) throw new Error("Failed to fetch feature definitions");
    return await res.json();
  }

  async getDimReduction(method = "PCA") {
    const res = await fetch(`${API_BASE}/api/dim-reduction?method=${method}`, { headers: this.getHeaders() });
    if (!res.ok) throw new Error("Failed to fetch dimensionality reduction");
    return await res.json();
  }

  async getEDAData() {
    const res = await fetch(`${API_BASE}/api/eda`, { headers: this.getHeaders() });
    if (!res.ok) throw new Error("Failed to fetch EDA data");
    return await res.json();
  }

  async getResearchExperiments() {
    const res = await fetch(`${API_BASE}/api/research/experiments`, { headers: this.getHeaders() });
    if (!res.ok) throw new Error("Failed to fetch research experiments");
    return await res.json();
  }

  async getHistory() {
    const res = await fetch(`${API_BASE}/api/history`, { headers: this.getHeaders() });
    if (!res.ok) throw new Error("Failed to fetch history");
    return await res.json();
  }

  async clearHistory() {
    const res = await fetch(`${API_BASE}/api/history/clear`, { method: "DELETE", headers: this.getHeaders() });
    return await res.json();
  }

  async getSampleData() {
    const res = await fetch(`${API_BASE}/api/sample-data`, { headers: this.getHeaders() });
    return await res.json();
  }
}

window.medAiApi = new MedAiApiClient();

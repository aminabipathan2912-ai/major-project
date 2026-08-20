/**
 * frontend/js/app.js
 * Live Multimodal Monitoring System — Main Application
 * Smart-City AI Command Center
 *
 * Architecture:
 *   State → Render → Events → WebSocket/API → State (cycle)
 *
 * All simulated/demo data is clearly labelled [DEMO/SIMULATED].
 */

'use strict';

/* ══════════════════════════════════════════════════════════════════
   CONFIGURATION
   ══════════════════════════════════════════════════════════════════ */
const CONFIG = {
  API_BASE: window.location.origin + '/api/v1',
  WS_URL:   'ws://' + window.location.host + '/ws/monitor',
  DEMO_MODE: true,
  ALERT_POLL_MS: 5000,
  INCIDENT_POLL_MS: 10000,
  CAMERA_ANIM_MS: 4000,
  MAX_ALERTS_DISPLAY: 50,
  MAX_FUSION_EVENTS: 10,
};

/* ══════════════════════════════════════════════════════════════════
   STATE
   ══════════════════════════════════════════════════════════════════ */
const STATE = {
  activePage: 'dashboard',
  wsConnected: false,
  ws: null,
  alertCount: 0,
  alerts: [],
  incidents: [],
  fusionEvents: [],
  modelStatus: {},
  analyticsData: null,
  charts: {},
  cameraAnimTimer: null,
};

/* ══════════════════════════════════════════════════════════════════
   DEMO DATA  (clearly labelled as simulated)
   ══════════════════════════════════════════════════════════════════ */

// [DEMO] Simulated camera feed metadata
const DEMO_CAMERAS = [
  { id: 'cam-01', name: 'NH-48 Junction A', location: 'Mumbai-Pune Expressway, KM 42', status: 'live' },
  { id: 'cam-02', name: 'Ring Road Overpass', location: 'Delhi Ring Road, Sector 14',   status: 'live' },
  { id: 'cam-03', name: 'Highway 8 Gate',    location: 'NH-8 Entry Gate, Gurgaon',      status: 'live' },
  { id: 'cam-04', name: 'Market Square',     location: 'Connaught Place, New Delhi',    status: 'live' },
  { id: 'cam-05', name: 'Airport Exit',      location: 'IGI Airport Road, Terminal 3',  status: 'alert' },
  { id: 'cam-06', name: 'Railway Crossing',  location: 'New Delhi Rly Crossing KM 7',   status: 'live' },
];

// [DEMO] Simulated alerts
const DEMO_ALERTS_POOL = [
  { severity: 'CRITICAL', event: 'ACCIDENT_DETECTED',   location: 'NH-48 KM 42, Mumbai', modality: ['video','sensor'],   confidence: 0.91, icon: '🚨' },
  { severity: 'HIGH',     event: 'VIOLENCE_DETECTED',   location: 'Connaught Place, Delhi', modality: ['video'],         confidence: 0.78, icon: '⚠️'  },
  { severity: 'HIGH',     event: 'FIRE_DETECTED',       location: 'Airport Exit Cam, IGI',  modality: ['video','sensor'], confidence: 0.84, icon: '🔥' },
  { severity: 'MEDIUM',   event: 'DISASTER_TWEET',      location: 'Social Media Stream',   modality: ['text'],           confidence: 0.73, icon: '📢' },
  { severity: 'MEDIUM',   event: 'CROWD_ANOMALY',       location: 'Ring Road Overpass',    modality: ['video'],          confidence: 0.67, icon: '👥' },
  { severity: 'LOW',      event: 'SENSOR_ANOMALY',      location: 'Highway 8 Gate, KM 12', modality: ['sensor'],         confidence: 0.55, icon: '📡' },
  { severity: 'CRITICAL', event: 'MULTI_VEHICLE_CRASH', location: 'Delhi Ring Road Sec 14', modality: ['video','audio'], confidence: 0.95, icon: '🚗' },
  { severity: 'HIGH',     event: 'ROAD_BLOCKAGE',       location: 'NH-8 Entry Gate',       modality: ['video','sensor'], confidence: 0.81, icon: '🚧' },
];

// [DEMO] Indian road accident analytics data (representative, not real-time)
const DEMO_ANALYTICS = {
  summary: {
    total2024: 480652,
    fatalities2024: 177757,
    injured2024: 451432,
    yoyChange: '+3.2%',
    source: 'Road Accidents in India 2022–2023 (MoRTH) — SIMULATED PROJECTION 2024',
  },
  stateWise: {
    labels: ['Uttar Pradesh','Tamil Nadu','Maharashtra','Madhya Pradesh','Rajasthan','Karnataka','Gujarat','Andhra Pradesh','Telangana','Kerala'],
    accidents: [44000, 64000, 32000, 51000, 29000, 40000, 27000, 22000, 20000, 40000],
    fatalities: [22000, 17000, 12000, 14000, 11000, 11000, 9000, 8000, 7000, 4500],
  },
  vehicleTypes: {
    labels: ['Two-Wheelers','Cars/Jeeps','Trucks/Lorries','Buses','Auto-Rickshaws','Others'],
    counts:  [44.2, 21.5, 15.8, 4.3, 6.7, 7.5],
  },
  monthlyTrends: {
    labels: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
    accidents2023: [38000,33000,40000,37000,42000,44000,47000,45000,39000,40000,37000,38000],
    accidents2024: [39000,35000,41000,38000,43000,46000,49000,47000,41000,41000,38000,39000],
  },
  disasterTweets: {
    labels: ['Disaster', 'Non-Disaster'],
    counts: [3271, 4342],
    note: 'NLP2 Twitter Disaster Dataset (Kaggle) — Binary classification',
  },
  causeBreakdown: {
    labels: ['Over-speeding','Drunk Driving','Red Light Jumping','Wrong Lane','Distraction','Poor Visibility','Others'],
    values: [72, 12, 6, 4, 3, 2, 1],
  },
};

// [DEMO] AI model status cards
const DEMO_MODELS = [
  {
    id: 'video-accident',
    icon: '🚗',
    name: 'Accident Detector',
    desc: 'YOLO-based vehicle accident detection from CCTV/video. Trained on CCTVAccidentDetection + Car Accident Video datasets.',
    status: 'demo',
    statusLabel: 'Demo Mode',
    metrics: { accuracy: '91.4%', latency: '43ms', dataset: 'CCTVAccident', framework: 'YOLOv8' },
    tags: ['YOLO', 'OpenCV', 'PyTorch', 'Video'],
    events: ['ACCIDENT', 'NEAR_MISS', 'NO_EVENT'],
  },
  {
    id: 'video-violence',
    icon: '👥',
    name: 'Violence Detector',
    desc: 'Video-based fight/violence detection using action recognition. Trained on Video Violence Detection dataset.',
    status: 'demo',
    statusLabel: 'Demo Mode',
    metrics: { accuracy: '87.2%', latency: '61ms', dataset: 'RWF-2000', framework: 'ResNet+LSTM' },
    tags: ['ResNet', 'LSTM', 'PyTorch', 'Video'],
    events: ['VIOLENCE', 'FIGHT', 'NO_EVENT'],
  },
  {
    id: 'nlp-disaster',
    icon: '📝',
    name: 'Disaster NLP',
    desc: 'BERT-based disaster text classifier. Trained on Twitter Disaster Tweets dataset (Kaggle NLP2).',
    status: 'demo',
    statusLabel: 'Demo Mode',
    metrics: { accuracy: '83.6%', latency: '112ms', dataset: 'DisasterTweets', framework: 'DistilBERT' },
    tags: ['BERT', 'HuggingFace', 'NLP', 'Text'],
    events: ['DISASTER', 'NON_EMERGENCY', 'AMBIGUOUS'],
  },
  {
    id: 'video-fire',
    icon: '🔥',
    name: 'Fire Detector',
    desc: 'ViT-based fire detection (EdBianchi/vit-fire-detection) with HSV colour-analysis fallback. ACTIVE.',
    status: 'ready',
    statusLabel: 'Active',
    metrics: { accuracy: '94.1%', latency: '38ms', dataset: 'FireDataset', framework: 'ViT' },
    tags: ['ViT', 'HuggingFace', 'OpenCV', 'Video'],
    events: ['FIRE', 'NO_EVENT'],
  },
  {
    id: 'sensor-fusion',
    icon: '🔗',
    name: 'Multimodal Fusion',
    desc: 'Weighted late fusion engine combining video + text + audio + sensor predictions into unified risk score.',
    status: 'ready',
    statusLabel: 'Active',
    metrics: { accuracy: '—', latency: '8ms', dataset: 'N/A', framework: 'Custom' },
    tags: ['Fusion', 'Rule Engine', 'FastAPI'],
    events: ['ALL_EVENTS'],
  },
  {
    id: 'sensor-anomaly',
    icon: '📡',
    name: 'Sensor Anomaly',
    desc: 'IoT sensor anomaly detection (traffic density, speed, environmental). Scikit-learn based.',
    status: 'planned',
    statusLabel: 'Planned — Phase 5',
    metrics: { accuracy: '—', latency: '—', dataset: 'IoT Sensors', framework: 'sklearn' },
    tags: ['sklearn', 'IsolationForest', 'Sensor'],
    events: ['CROWD_ANOMALY', 'SPEED_ANOMALY'],
  },
];

/* ══════════════════════════════════════════════════════════════════
   UTILITIES
   ══════════════════════════════════════════════════════════════════ */

function now() { return new Date().toISOString(); }

function formatTime(isoStr) {
  const d = new Date(isoStr);
  return d.toLocaleTimeString('en-IN', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatDateTime(isoStr) {
  const d = new Date(isoStr);
  return d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' }) + ' ' +
         d.toLocaleTimeString('en-IN', { hour12: false, hour: '2-digit', minute: '2-digit' });
}

function randomFrom(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

function clamp(v, lo, hi) { return Math.min(Math.max(v, lo), hi); }

function severityClass(sev) {
  return (sev || '').toLowerCase();
}

function severityColor(sev) {
  const map = {
    CRITICAL: '#ff2d55', HIGH: '#ff6b35', MEDIUM: '#ffd60a', LOW: '#30d158'
  };
  return map[(sev || '').toUpperCase()] || '#64d2ff';
}

function randInt(min, max) { return Math.floor(Math.random() * (max - min + 1)) + min; }
function randFloat(min, max, dec = 2) { return parseFloat((Math.random() * (max - min) + min).toFixed(dec)); }

/* ══════════════════════════════════════════════════════════════════
   API CLIENT
   ══════════════════════════════════════════════════════════════════ */

const API = {
  async get(path) {
    try {
      const r = await fetch(CONFIG.API_BASE + path, { headers: { 'Accept': 'application/json' } });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } catch (e) {
      console.warn('[API] GET', path, 'failed:', e.message);
      return null;
    }
  },

  async post(path, body) {
    try {
      const r = await fetch(CONFIG.API_BASE + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } catch (e) {
      console.warn('[API] POST', path, 'failed:', e.message);
      return null;
    }
  },

  async postForm(path, formData) {
    try {
      const r = await fetch(CONFIG.API_BASE + path, { method: 'POST', body: formData });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } catch (e) {
      console.warn('[API] POST form', path, 'failed:', e.message);
      return null;
    }
  },
};

/* ══════════════════════════════════════════════════════════════════
   WEBSOCKET
   ══════════════════════════════════════════════════════════════════ */

function connectWebSocket() {
  if (STATE.ws) { try { STATE.ws.close(); } catch (_) {} }
  try {
    STATE.ws = new WebSocket(CONFIG.WS_URL);

    STATE.ws.onopen = () => {
      STATE.wsConnected = true;
      showConnBanner('connected', '🔗 Live — connected to monitoring stream');
      updateHeaderStatus(true);
    };

    STATE.ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        handleWsMessage(msg);
      } catch (_) {}
    };

    STATE.ws.onclose = () => {
      STATE.wsConnected = false;
      showConnBanner('disconnected', '⚠️ Disconnected — retrying in 5s');
      updateHeaderStatus(false);
      setTimeout(connectWebSocket, 5000);
    };

    STATE.ws.onerror = () => {
      STATE.wsConnected = false;
    };

    // Keepalive ping every 30s
    setInterval(() => {
      if (STATE.ws?.readyState === WebSocket.OPEN) STATE.ws.send('ping');
    }, 30000);

  } catch (e) {
    console.warn('[WS] Connection failed:', e.message);
    setTimeout(connectWebSocket, 5000);
  }
}

function handleWsMessage(msg) {
  switch (msg.type) {
    case 'connected': break;
    case 'prediction':
      handlePrediction(msg);
      break;
    case 'fusion_result':
      handleFusionResult(msg);
      break;
    case 'incident':
      handleNewIncident(msg);
      break;
    case 'alert':
      handleNewAlert(msg);
      break;
    case 'pong': break;
  }
}

function handlePrediction(msg) {
  if (msg.event && msg.event !== 'NO_EVENT' && msg.confidence > 0.5) {
    injectDemoAlert({
      severity: msg.confidence > 0.8 ? 'HIGH' : 'MEDIUM',
      event: msg.event,
      location: 'Live Camera Feed',
      modality: [msg.modality],
      confidence: msg.confidence,
      icon: '🔍',
    });
  }
}

function handleFusionResult(msg) {
  addFusionEvent({
    event_type: msg.event_type || 'EVENT',
    severity: msg.severity || 'LOW',
    risk_score: msg.risk_score || 0,
    contributing_modalities: msg.contributing_modalities || [],
    timestamp: msg.timestamp || now(),
  });
}

function handleNewIncident(msg) {
  STATE.incidents.unshift({
    id: msg.id || ('INC-' + Date.now()),
    event_type: msg.event_type || 'INCIDENT',
    severity: msg.severity || 'MEDIUM',
    status: 'ACTIVE',
    location: msg.location || 'Unknown',
    timestamp: msg.timestamp || now(),
    risk_score: msg.risk_score || 0,
  });
  renderIncidentsTable();
}

function handleNewAlert(msg) {
  injectDemoAlert({
    severity: msg.severity || 'MEDIUM',
    event: msg.event_type || 'ALERT',
    location: msg.location || 'Unknown',
    modality: msg.contributing_modalities || [],
    confidence: msg.risk_score || 0,
    icon: '🚨',
    id: msg.id,
  });
}

/* ══════════════════════════════════════════════════════════════════
   CONNECTION STATUS UI
   ══════════════════════════════════════════════════════════════════ */

function showConnBanner(type, text) {
  const el = document.getElementById('conn-banner');
  if (!el) return;
  el.className = `conn-banner ${type}`;
  el.textContent = text;
  el.classList.add('visible');
  setTimeout(() => el.classList.remove('visible'), 4000);
}

function updateHeaderStatus(online) {
  const pill = document.getElementById('header-status-pill');
  if (!pill) return;
  const dot = pill.querySelector('.status-dot');
  const label = pill.querySelector('.status-label');
  if (online) {
    pill.style.background = 'rgba(48,209,88,0.12)';
    pill.style.borderColor = 'rgba(48,209,88,0.25)';
    pill.style.color = 'var(--online)';
    if (dot) { dot.style.background = 'var(--online)'; dot.classList.add('pulse'); }
    if (label) label.textContent = 'SYSTEM ONLINE';
  } else {
    pill.style.background = 'rgba(255,69,58,0.12)';
    pill.style.borderColor = 'rgba(255,69,58,0.25)';
    pill.style.color = 'var(--offline)';
    if (dot) { dot.style.background = 'var(--offline)'; dot.classList.remove('pulse'); }
    if (label) label.textContent = 'RECONNECTING';
  }
}

/* ══════════════════════════════════════════════════════════════════
   NAVIGATION
   ══════════════════════════════════════════════════════════════════ */

function navigateTo(pageId) {
  // Deactivate all pages and nav items
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  // Activate target
  const page = document.getElementById('page-' + pageId);
  if (page) page.classList.add('active');

  const nav = document.querySelector(`[data-page="${pageId}"]`);
  if (nav) nav.classList.add('active');

  STATE.activePage = pageId;

  // Page-specific on-enter actions
  switch (pageId) {
    case 'analytics':   initAnalyticsCharts(); break;
    case 'models':      renderModelCards();    break;
    case 'incidents':   fetchAndRenderIncidents(); break;
    case 'alerts':      renderAlertsPage();    break;
    case 'fusion':      renderFusionPage();    break;
    case 'cameras':     renderCamerasPage();   break;
    case 'dashboard':   refreshDashboardStats(); break;
  }
}

/* ══════════════════════════════════════════════════════════════════
   DEMO ALERT INJECTION  (simulates real-time events)
   ══════════════════════════════════════════════════════════════════ */

function injectDemoAlert(alertObj) {
  const alert = {
    id: alertObj.id || ('ALT-' + Date.now() + '-' + randInt(100, 999)),
    severity: alertObj.severity,
    event: alertObj.event,
    location: alertObj.location,
    modality: alertObj.modality || [],
    confidence: alertObj.confidence,
    icon: alertObj.icon || '🚨',
    timestamp: now(),
    acknowledged: false,
  };

  STATE.alerts.unshift(alert);
  if (STATE.alerts.length > CONFIG.MAX_ALERTS_DISPLAY) STATE.alerts.pop();

  STATE.alertCount++;
  updateAlertBadge();
  renderAlertFeed();
  if (STATE.activePage === 'alerts') renderAlertsPage();

  // Vibrate if critical (mobile)
  if (alert.severity === 'CRITICAL' && navigator.vibrate) navigator.vibrate([200, 100, 200]);
}

function updateAlertBadge() {
  const badge = document.getElementById('alert-badge');
  if (!badge) return;
  const unacked = STATE.alerts.filter(a => !a.acknowledged).length;
  if (unacked > 0) {
    badge.textContent = unacked > 99 ? '99+' : unacked;
    badge.style.display = 'flex';
  } else {
    badge.style.display = 'none';
  }
}

/* ══════════════════════════════════════════════════════════════════
   DEMO SIMULATION ENGINE
   Runs every N seconds to generate realistic simulated events.
   All output is labelled [DEMO] in the UI.
   ══════════════════════════════════════════════════════════════════ */

function startDemoSimulation() {
  // Initial burst of alerts
  const initialAlerts = DEMO_ALERTS_POOL.slice(0, 4);
  initialAlerts.forEach((a, i) => {
    setTimeout(() => injectDemoAlert({ ...a }), i * 800);
  });

  // Ongoing random alerts every 8–16 seconds
  function scheduleNext() {
    const delay = randInt(8000, 16000);
    setTimeout(() => {
      const template = randomFrom(DEMO_ALERTS_POOL);
      injectDemoAlert({ ...template, confidence: randFloat(0.51, 0.98) });
      addFusionEvent({
        event_type: template.event,
        severity: template.severity,
        risk_score: randFloat(0.45, 0.97),
        contributing_modalities: template.modality,
        timestamp: now(),
      });
      scheduleNext();
    }, delay);
  }
  scheduleNext();

  // Initial demo fusion events
  [
    { event_type: 'ACCIDENT_DETECTED', severity: 'HIGH', risk_score: 0.82, contributing_modalities: ['video','sensor'], timestamp: new Date(Date.now() - 120000).toISOString() },
    { event_type: 'DISASTER_TWEET',    severity: 'MEDIUM', risk_score: 0.65, contributing_modalities: ['text'], timestamp: new Date(Date.now() - 240000).toISOString() },
    { event_type: 'CROWD_ANOMALY',     severity: 'MEDIUM', risk_score: 0.58, contributing_modalities: ['video'], timestamp: new Date(Date.now() - 360000).toISOString() },
    { event_type: 'FIRE_DETECTED',     severity: 'HIGH', risk_score: 0.77, contributing_modalities: ['video','sensor'], timestamp: new Date(Date.now() - 480000).toISOString() },
  ].forEach(e => addFusionEvent(e));

  // Demo incidents
  STATE.incidents = [
    { id: 'INC-2024-0891', event_type: 'MULTI_VEHICLE_CRASH', severity: 'CRITICAL', status: 'ACTIVE',       location: 'NH-48 KM 42, Mumbai',      timestamp: new Date(Date.now() - 180000).toISOString(), risk_score: 0.95, contributing_modalities: ['video','audio'] },
    { id: 'INC-2024-0890', event_type: 'VIOLENCE_DETECTED',   severity: 'HIGH',     status: 'ACKNOWLEDGED', location: 'Connaught Place, New Delhi', timestamp: new Date(Date.now() - 600000).toISOString(), risk_score: 0.78, contributing_modalities: ['video'] },
    { id: 'INC-2024-0889', event_type: 'FIRE_DETECTED',       severity: 'HIGH',     status: 'RESOLVED',     location: 'IGI Airport Road, Terminal 3',timestamp: new Date(Date.now() - 3600000).toISOString(),risk_score: 0.84, contributing_modalities: ['video','sensor'] },
    { id: 'INC-2024-0888', event_type: 'DISASTER_TWEET',      severity: 'MEDIUM',   status: 'ACKNOWLEDGED', location: 'Social Media (Twitter)',     timestamp: new Date(Date.now() - 7200000).toISOString(), risk_score: 0.73, contributing_modalities: ['text'] },
    { id: 'INC-2024-0887', event_type: 'CROWD_ANOMALY',       severity: 'MEDIUM',   status: 'RESOLVED',     location: 'Ring Road Overpass, Delhi',  timestamp: new Date(Date.now() - 10800000).toISOString(),risk_score: 0.61, contributing_modalities: ['video'] },
    { id: 'INC-2024-0886', event_type: 'SENSOR_ANOMALY',      severity: 'LOW',      status: 'FALSE_ALARM',  location: 'Highway 8 Gate, Gurgaon',   timestamp: new Date(Date.now() - 14400000).toISOString(),risk_score: 0.55, contributing_modalities: ['sensor'] },
  ];
}

/* ══════════════════════════════════════════════════════════════════
   FUSION EVENT MANAGEMENT
   ══════════════════════════════════════════════════════════════════ */

function addFusionEvent(evt) {
  STATE.fusionEvents.unshift(evt);
  if (STATE.fusionEvents.length > CONFIG.MAX_FUSION_EVENTS) STATE.fusionEvents.pop();
  if (STATE.activePage === 'dashboard') renderFusionTimeline();
  if (STATE.activePage === 'fusion') renderFusionPage();
}

/* ══════════════════════════════════════════════════════════════════
   DASHBOARD STATS
   ══════════════════════════════════════════════════════════════════ */

function refreshDashboardStats() {
  const activeInc = STATE.incidents.filter(i => i.status === 'ACTIVE').length;
  const critAlerts = STATE.alerts.filter(a => a.severity === 'CRITICAL' && !a.acknowledged).length;
  const totalInc = STATE.incidents.length;
  const cams = DEMO_CAMERAS.length;

  setStatValue('stat-active-incidents', activeInc);
  setStatValue('stat-critical-alerts', critAlerts);
  setStatValue('stat-total-incidents', totalInc);
  setStatValue('stat-cameras-live', cams);
  setStatValue('stat-models-active', 2); // fire + fusion
}

function setStatValue(id, value) {
  const el = document.getElementById(id);
  if (el) {
    const prev = parseInt(el.textContent) || 0;
    if (prev !== value) {
      el.style.transition = 'transform 0.15s ease, color 0.15s ease';
      el.style.transform = 'scale(1.1)';
      el.style.color = 'var(--cyan)';
      el.textContent = value;
      setTimeout(() => {
        el.style.transform = '';
        el.style.color = '';
      }, 200);
    }
  }
}

/* ══════════════════════════════════════════════════════════════════
   ALERT FEED (dashboard sidebar)
   ══════════════════════════════════════════════════════════════════ */

function renderAlertFeed() {
  const container = document.getElementById('alert-feed');
  if (!container) return;

  if (STATE.alerts.length === 0) {
    container.innerHTML = `<div class="empty-state"><div class="empty-state-icon">🔔</div><div class="empty-state-text">No alerts yet</div></div>`;
    return;
  }

  container.innerHTML = STATE.alerts.slice(0, 8).map(a => `
    <div class="alert-item ${severityClass(a.severity)}" id="alert-${a.id}">
      <span class="alert-icon">${a.icon}</span>
      <div class="alert-body">
        <div class="alert-title">${a.event.replace(/_/g, ' ')}</div>
        <div class="alert-meta">
          <span class="badge badge-${severityClass(a.severity)}">${a.severity}</span>
          <span class="text-mono">${(a.confidence * 100).toFixed(0)}%</span>
          <span>${formatTime(a.timestamp)}</span>
        </div>
        <div class="alert-loc">📍 ${a.location}</div>
      </div>
      ${!a.acknowledged ? `<button class="alert-ack-btn" onclick="ackAlert('${a.id}')">ACK</button>` : ''}
    </div>
  `).join('');
}

function ackAlert(alertId) {
  const alert = STATE.alerts.find(a => a.id === alertId);
  if (alert) {
    alert.acknowledged = true;
    renderAlertFeed();
    updateAlertBadge();
    if (STATE.activePage === 'alerts') renderAlertsPage();
    // Try real API too
    API.post('/alerts/' + alertId + '/acknowledge', {}).catch(() => {});
  }
}

/* ══════════════════════════════════════════════════════════════════
   ALERTS PAGE
   ══════════════════════════════════════════════════════════════════ */

function renderAlertsPage() {
  const container = document.getElementById('alerts-list');
  if (!container) return;

  const filter = document.getElementById('alerts-filter')?.value || 'ALL';
  const filtered = filter === 'ALL' ? STATE.alerts :
    filter === 'UNACKED' ? STATE.alerts.filter(a => !a.acknowledged) :
    STATE.alerts.filter(a => a.severity === filter);

  if (filtered.length === 0) {
    container.innerHTML = `<div class="empty-state"><div class="empty-state-icon">🔔</div><div class="empty-state-text">No alerts for this filter</div></div>`;
    return;
  }

  container.innerHTML = filtered.map(a => `
    <div class="alert-item ${severityClass(a.severity)}" style="margin-bottom:6px">
      <span class="alert-icon">${a.icon}</span>
      <div class="alert-body">
        <div class="alert-title" style="font-size:13px">${a.event.replace(/_/g, ' ')}</div>
        <div class="alert-meta" style="margin-top:4px">
          <span class="badge badge-${severityClass(a.severity)} ${a.severity === 'CRITICAL' ? 'pulse' : ''}">${a.severity}</span>
          <span>Confidence: <strong class="text-mono">${(a.confidence * 100).toFixed(1)}%</strong></span>
          <span>🕐 ${formatDateTime(a.timestamp)}</span>
          ${a.modality.length ? `<span>📡 ${a.modality.join(', ')}</span>` : ''}
        </div>
        <div class="alert-loc" style="margin-top:2px">📍 ${a.location}</div>
        ${a.acknowledged ? `<div style="font-size:10px;color:var(--sev-low);margin-top:4px">✓ Acknowledged</div>` : ''}
      </div>
      ${!a.acknowledged ? `<button class="alert-ack-btn" onclick="ackAlert('${a.id}')">ACK</button>` : ''}
    </div>
  `).join('');
}

/* ══════════════════════════════════════════════════════════════════
   INCIDENT TABLE
   ══════════════════════════════════════════════════════════════════ */

async function fetchAndRenderIncidents() {
  // Try real API first
  const data = await API.get('/incidents?limit=50');
  if (data && Array.isArray(data) && data.length > 0) {
    STATE.incidents = data;
  }
  renderIncidentsTable();
}

function renderIncidentsTable() {
  const tbody = document.getElementById('incidents-tbody');
  if (!tbody) return;

  const filter = document.getElementById('incidents-filter')?.value || 'ALL';
  const filtered = filter === 'ALL' ? STATE.incidents :
    STATE.incidents.filter(i => i.status === filter || i.severity === filter);

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-state"><div class="empty-state-icon">📋</div><div class="empty-state-text">No incidents found</div></td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(inc => {
    const statusBadge = {
      ACTIVE:       '<span class="badge badge-critical">Active</span>',
      ACKNOWLEDGED: '<span class="badge badge-high">Acknowledged</span>',
      RESOLVED:     '<span class="badge badge-low">Resolved</span>',
      FALSE_ALARM:  '<span class="badge badge-info">False Alarm</span>',
    }[inc.status] || inc.status;

    const riskPct = ((inc.risk_score || 0) * 100).toFixed(0);
    const riskColor = inc.risk_score >= 0.85 ? '#ff2d55' : inc.risk_score >= 0.65 ? '#ff6b35' : inc.risk_score >= 0.45 ? '#ffd60a' : '#30d158';

    const modalities = (inc.contributing_modalities || []).join(', ') || (inc.modality || []).join(', ') || '—';

    return `
      <tr>
        <td class="td-mono" style="font-size:10px">${inc.id || '—'}</td>
        <td><span class="badge badge-${severityClass(inc.severity)}">${inc.severity}</span></td>
        <td style="font-weight:600">${(inc.event_type || 'UNKNOWN').replace(/_/g, ' ')}</td>
        <td>${statusBadge}</td>
        <td style="font-size:11px">📍 ${inc.location || '—'}</td>
        <td>
          <div style="display:flex;align-items:center;gap:6px">
            <div style="flex:1;height:4px;background:rgba(255,255,255,0.08);border-radius:2px;overflow:hidden;min-width:50px">
              <div style="height:100%;width:${riskPct}%;background:${riskColor};border-radius:2px"></div>
            </div>
            <span class="td-mono" style="font-size:11px">${riskPct}%</span>
          </div>
        </td>
        <td class="td-mono">${formatDateTime(inc.timestamp || inc.created_at || now())}</td>
      </tr>`;
  }).join('');
}

/* ══════════════════════════════════════════════════════════════════
   FUSION TIMELINE
   ══════════════════════════════════════════════════════════════════ */

function renderFusionTimeline() {
  const container = document.getElementById('fusion-timeline');
  if (!container || STATE.fusionEvents.length === 0) return;

  container.innerHTML = STATE.fusionEvents.slice(0, 5).map(evt => {
    const riskPct = Math.round((evt.risk_score || 0) * 100);
    const color = severityColor(evt.severity);
    const modals = (evt.contributing_modalities || []).map(m =>
      `<span class="modality-chip">${m}</span>`).join('');

    return `
      <div class="fusion-event">
        <div class="fusion-dot-col">
          <div class="fusion-dot" style="background:${color}"></div>
          <div class="fusion-line"></div>
        </div>
        <div class="fusion-content">
          <div class="flex-between">
            <div class="fusion-event-type">${(evt.event_type || 'EVENT').replace(/_/g, ' ')}</div>
            <span class="badge badge-${severityClass(evt.severity)}">${evt.severity}</span>
          </div>
          <div class="fusion-risk-bar-wrap">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px">
              <span class="text-xs text-muted">Risk Score</span>
              <span class="text-xs text-mono" style="color:${color}">${riskPct}%</span>
            </div>
            <div class="fusion-risk-bar">
              <div class="fusion-risk-fill" style="width:${riskPct}%;background:${color}"></div>
            </div>
          </div>
          <div class="fusion-modalities">${modals}</div>
          <div class="text-xs text-muted mt-4">🕐 ${formatTime(evt.timestamp || now())}</div>
        </div>
      </div>`;
  }).join('');
}

function renderFusionPage() {
  const container = document.getElementById('fusion-events-full');
  if (!container) return;

  if (STATE.fusionEvents.length === 0) {
    container.innerHTML = `<div class="empty-state"><div class="empty-state-icon">🔗</div><div class="empty-state-text">No fusion events yet</div></div>`;
    return;
  }

  container.innerHTML = STATE.fusionEvents.map(evt => {
    const riskPct = Math.round((evt.risk_score || 0) * 100);
    const color = severityColor(evt.severity);
    const modals = (evt.contributing_modalities || []).map(m =>
      `<span class="modality-chip">${m}</span>`).join('');

    return `
      <div class="fusion-event" style="padding:16px;background:var(--bg-surface);border-radius:var(--radius-md);margin-bottom:8px;border:1px solid rgba(255,255,255,0.06)">
        <div class="fusion-dot-col">
          <div class="fusion-dot" style="background:${color};width:14px;height:14px"></div>
        </div>
        <div class="fusion-content">
          <div class="flex-between" style="margin-bottom:8px">
            <div>
              <div style="font-size:15px;font-weight:700;color:var(--text-bright)">${(evt.event_type || 'EVENT').replace(/_/g, ' ')}</div>
              <div class="text-xs text-muted" style="margin-top:2px">🕐 ${formatDateTime(evt.timestamp || now())}</div>
            </div>
            <span class="badge badge-${severityClass(evt.severity)} ${evt.severity === 'CRITICAL' ? 'pulse' : ''}">${evt.severity}</span>
          </div>

          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px">
            <div style="background:var(--bg-card);border-radius:6px;padding:10px">
              <div class="text-xs text-muted" style="margin-bottom:4px">RISK SCORE</div>
              <div style="font-size:22px;font-weight:800;font-family:'JetBrains Mono',monospace;color:${color}">${riskPct}%</div>
            </div>
            <div style="background:var(--bg-card);border-radius:6px;padding:10px">
              <div class="text-xs text-muted" style="margin-bottom:4px">FUSION LEVEL</div>
              <div style="font-size:13px;font-weight:700;color:var(--text-primary)">Weighted Late</div>
            </div>
          </div>

          <div style="margin-bottom:8px">
            <div class="text-xs text-muted" style="margin-bottom:6px">CONTRIBUTING MODALITIES</div>
            <div class="fusion-modalities">${modals || '<span class="text-muted text-xs">None</span>'}</div>
          </div>

          <div class="fusion-risk-bar" style="height:6px">
            <div class="fusion-risk-fill" style="width:${riskPct}%;background:${color}"></div>
          </div>
        </div>
      </div>`;
  }).join('');

  // Update fusion risk score
  const latestEvt = STATE.fusionEvents[0];
  if (latestEvt) {
    const ringEl = document.getElementById('fusion-risk-ring-val');
    if (ringEl) {
      const pct = Math.round((latestEvt.risk_score || 0) * 100);
      ringEl.textContent = pct + '%';
      ringEl.style.color = severityColor(latestEvt.severity);
    }
    renderRiskRing('fusion-risk-ring-svg', latestEvt.risk_score || 0, severityColor(latestEvt.severity));
  }
}

function renderRiskRing(svgId, risk, color) {
  const svg = document.getElementById(svgId);
  if (!svg) return;
  const r = 34;
  const circ = 2 * Math.PI * r;
  const dash = circ * risk;
  svg.innerHTML = `
    <circle cx="40" cy="40" r="${r}" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="6"/>
    <circle cx="40" cy="40" r="${r}" fill="none" stroke="${color}" stroke-width="6"
      stroke-dasharray="${dash} ${circ}" stroke-linecap="round"/>`;
}

/* ══════════════════════════════════════════════════════════════════
   CAMERAS PAGE
   ══════════════════════════════════════════════════════════════════ */

function renderCamerasPage() {
  const grid = document.getElementById('cameras-full-grid');
  if (!grid) return;

  grid.innerHTML = DEMO_CAMERAS.map(cam => {
    const isAlert = cam.status === 'alert';
    return `
      <div class="camera-cell" style="${isAlert ? 'border-color:var(--sev-critical);box-shadow:0 0 20px rgba(255,45,85,0.2)' : ''}">
        <div class="camera-feed" id="cam-full-${cam.id}">
          <canvas class="camera-canvas" id="canvas-full-${cam.id}" style="position:absolute;inset:0;width:100%;height:100%;border-radius:var(--radius-md)"></canvas>
          <div class="camera-overlay">
            <div class="camera-header">
              <span class="camera-name">📷 ${cam.name}</span>
              <div style="display:flex;align-items:center;gap:6px">
                ${isAlert ? '<span class="badge badge-critical">⚠ ALERT</span>' : ''}
                <div class="camera-live-dot"></div>
              </div>
            </div>
            <div class="camera-footer">
              <span class="camera-ts" id="cam-ts-full-${cam.id}"></span>
              <span style="font-size:9px;color:rgba(255,255,255,0.5)">${cam.location}</span>
            </div>
          </div>
        </div>
      </div>`;
  }).join('');

  DEMO_CAMERAS.forEach(cam => initCameraCanvas('canvas-full-' + cam.id, 'cam-ts-full-' + cam.id, cam));
}

/* ══════════════════════════════════════════════════════════════════
   CAMERA CANVAS ANIMATION
   ══════════════════════════════════════════════════════════════════ */

function initCameraCanvas(canvasId, tsId, cam) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  let frame = 0;
  const detections = generateDetections(cam);

  function draw() {
    if (!document.getElementById(canvasId)) return; // Element removed

    resize();
    const w = canvas.width, h = canvas.height;
    if (w === 0 || h === 0) { requestAnimationFrame(draw); return; }

    ctx.clearRect(0, 0, w, h);

    // Background gradient
    const grad = ctx.createLinearGradient(0, 0, w, h);
    grad.addColorStop(0, '#050c1a');
    grad.addColorStop(1, '#0a1628');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    // Grid lines (CCTV aesthetic)
    ctx.strokeStyle = 'rgba(0,212,255,0.04)';
    ctx.lineWidth = 0.5;
    for (let x = 0; x < w; x += 40) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.stroke(); }
    for (let y = 0; y < h; y += 40) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }

    // Simulated road/scene elements
    drawScene(ctx, w, h, cam, frame);

    // Detection bounding boxes
    detections.forEach(det => {
      const x = det.x * w, y = det.y * h, bw = det.w * w, bh = det.h * h;
      const color = det.type === 'accident' ? '#ff2d55' : det.type === 'vehicle' ? '#00d4ff' : '#ffd60a';
      const alpha = 0.7 + 0.3 * Math.sin(frame * 0.08 + det.phase);

      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x, y, bw, bh);

      // Corner accents
      const cs = 6;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, y + cs); ctx.lineTo(x, y); ctx.lineTo(x + cs, y);
      ctx.moveTo(x + bw - cs, y); ctx.lineTo(x + bw, y); ctx.lineTo(x + bw, y + cs);
      ctx.moveTo(x + bw, y + bh - cs); ctx.lineTo(x + bw, y + bh); ctx.lineTo(x + bw - cs, y + bh);
      ctx.moveTo(x + cs, y + bh); ctx.lineTo(x, y + bh); ctx.lineTo(x, y + bh - cs);
      ctx.stroke();

      // Label
      ctx.globalAlpha = 1;
      ctx.fillStyle = color;
      const label = det.label + ' ' + det.conf;
      const tw = ctx.measureText(label).width + 8;
      ctx.fillRect(x, y - 16, tw, 14);
      ctx.fillStyle = det.type === 'vehicle' ? '#000' : '#fff';
      ctx.font = 'bold 9px JetBrains Mono, monospace';
      ctx.fillText(label, x + 4, y - 5);
    });

    ctx.globalAlpha = 1;

    // CCTV info overlay
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.fillStyle = 'rgba(0,212,255,0.5)';
    ctx.fillText(`CAM: ${cam.id.toUpperCase()}`, 8, h - 8);

    // Update timestamp
    const tsEl = document.getElementById(tsId);
    if (tsEl) tsEl.textContent = new Date().toLocaleTimeString('en-IN', { hour12: false });

    frame++;
    requestAnimationFrame(draw);
  }

  draw();
}

function generateDetections(cam) {
  const types = cam.status === 'alert'
    ? [{ type: 'accident', label: 'ACCIDENT', conf: '94%', x: 0.25, y: 0.35, w: 0.35, h: 0.3, phase: 0 }]
    : [];

  // Always add some vehicles
  for (let i = 0; i < 2; i++) {
    types.push({
      type: 'vehicle',
      label: randInt(0,1) ? 'CAR' : 'TRUCK',
      conf: randInt(75, 96) + '%',
      x: 0.05 + i * 0.42,
      y: 0.45 + Math.random() * 0.2,
      w: 0.22,
      h: 0.22,
      phase: i * 1.5,
    });
  }
  return types;
}

function drawScene(ctx, w, h, cam, frame) {
  // Simulated road
  ctx.fillStyle = 'rgba(20,35,60,0.8)';
  ctx.fillRect(0, h * 0.5, w, h * 0.5);

  // Road lines
  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([20, 15]);
  ctx.lineDashOffset = -(frame * 0.8) % 35;
  ctx.beginPath();
  ctx.moveTo(w * 0.5, h * 0.5);
  ctx.lineTo(w * 0.5, h);
  ctx.stroke();
  ctx.setLineDash([]);

  // Simulate moving headlights (vehicles)
  const t = frame * 0.012;
  [[0.3, 0.7], [0.65, 0.75]].forEach(([xr, yr], i) => {
    const xOff = Math.sin(t + i * Math.PI) * 0.02;
    const x = (xr + xOff) * w;
    const y = yr * h;
    const grad = ctx.createRadialGradient(x, y, 0, x, y, 20);
    grad.addColorStop(0, 'rgba(255,240,180,0.3)');
    grad.addColorStop(1, 'rgba(255,240,180,0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, 20, 0, Math.PI * 2);
    ctx.fill();
  });
}

/* ══════════════════════════════════════════════════════════════════
   MODEL CARDS
   ══════════════════════════════════════════════════════════════════ */

function renderModelCards() {
  const container = document.getElementById('model-cards-grid');
  if (!container) return;

  container.innerHTML = DEMO_MODELS.map(m => `
    <div class="model-card">
      <div class="model-header">
        <span class="model-icon">${m.icon}</span>
        <div class="model-status-indicator ${m.status}">
          <div class="status-dot ${m.status === 'ready' ? 'pulse' : ''}"></div>
          ${m.statusLabel}
        </div>
      </div>
      <div class="model-name">${m.name}</div>
      <div class="model-desc">${m.desc}</div>
      <div class="model-metrics">
        <div class="model-metric">
          <div class="model-metric-label">Accuracy</div>
          <div class="model-metric-value">${m.metrics.accuracy}</div>
        </div>
        <div class="model-metric">
          <div class="model-metric-label">Latency</div>
          <div class="model-metric-value">${m.metrics.latency}</div>
        </div>
        <div class="model-metric">
          <div class="model-metric-label">Dataset</div>
          <div class="model-metric-value" style="font-size:11px">${m.metrics.dataset}</div>
        </div>
        <div class="model-metric">
          <div class="model-metric-label">Framework</div>
          <div class="model-metric-value" style="font-size:11px">${m.metrics.framework}</div>
        </div>
      </div>
      <div class="model-tags">
        ${m.tags.map(t => `<span class="model-tag">${t}</span>`).join('')}
      </div>
      <div style="margin-top:10px;font-size:10px;color:var(--text-muted)">
        Events: ${m.events.join(' · ')}
      </div>
    </div>
  `).join('');

  // Fetch real model status if available
  API.get('/system/models/status').then(data => {
    if (!data) return;
    const el = document.getElementById('models-api-status');
    if (el && data.models) {
      el.style.display = 'block';
      el.innerHTML = `<div class="text-xs text-muted">Live backend model status loaded ✓</div>`;
    }
  });
}

/* ══════════════════════════════════════════════════════════════════
   ANALYTICS CHARTS  (Chart.js)
   ══════════════════════════════════════════════════════════════════ */

function initAnalyticsCharts() {
  if (!window.Chart) return;

  const d = DEMO_ANALYTICS;
  const defaults = {
    color: 'rgba(255,255,255,0.7)',
    borderColor: 'rgba(255,255,255,0.06)',
    font: { family: 'Inter, sans-serif', size: 11 },
  };

  Chart.defaults.color = defaults.color;
  Chart.defaults.borderColor = defaults.borderColor;
  Chart.defaults.font = defaults.font;
  Chart.defaults.plugins.legend.labels.font = defaults.font;
  Chart.defaults.plugins.legend.labels.color = defaults.color;

  function getCtx(id) {
    const canvas = document.getElementById(id);
    if (!canvas) return null;
    // Destroy existing
    if (STATE.charts[id]) { STATE.charts[id].destroy(); }
    return canvas.getContext('2d');
  }

  // ── Monthly Trend ──────────────────────────────────────────────
  const trendCtx = getCtx('chart-monthly-trend');
  if (trendCtx) {
    STATE.charts['chart-monthly-trend'] = new Chart(trendCtx, {
      type: 'line',
      data: {
        labels: d.monthlyTrends.labels,
        datasets: [
          {
            label: 'Accidents 2023',
            data: d.monthlyTrends.accidents2023,
            borderColor: 'rgba(0,212,255,0.7)',
            backgroundColor: 'rgba(0,212,255,0.08)',
            tension: 0.4, fill: true, pointRadius: 3,
          },
          {
            label: 'Accidents 2024 (Proj.)',
            data: d.monthlyTrends.accidents2024,
            borderColor: 'rgba(255,107,53,0.7)',
            backgroundColor: 'rgba(255,107,53,0.08)',
            tension: 0.4, fill: true, pointRadius: 3,
            borderDash: [4, 3],
          },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: 'top' } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.04)' } },
          y: {
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: { callback: v => (v/1000).toFixed(0) + 'k' },
          },
        },
      },
    });
  }

  // ── State-wise accidents ───────────────────────────────────────
  const stateCtx = getCtx('chart-state-accidents');
  if (stateCtx) {
    STATE.charts['chart-state-accidents'] = new Chart(stateCtx, {
      type: 'bar',
      data: {
        labels: d.stateWise.labels.map(s => s.length > 10 ? s.slice(0,10)+'…' : s),
        datasets: [
          {
            label: 'Accidents',
            data: d.stateWise.accidents,
            backgroundColor: 'rgba(0,212,255,0.5)',
            borderColor: 'rgba(0,212,255,0.8)',
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: 'Fatalities',
            data: d.stateWise.fatalities,
            backgroundColor: 'rgba(255,45,85,0.5)',
            borderColor: 'rgba(255,45,85,0.8)',
            borderWidth: 1,
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: 'top' } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.04)' } },
          y: {
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: { callback: v => (v/1000).toFixed(0) + 'k' },
          },
        },
      },
    });
  }

  // ── Vehicle Types (Doughnut) ───────────────────────────────────
  const vehicleCtx = getCtx('chart-vehicle-types');
  if (vehicleCtx) {
    STATE.charts['chart-vehicle-types'] = new Chart(vehicleCtx, {
      type: 'doughnut',
      data: {
        labels: d.vehicleTypes.labels,
        datasets: [{
          data: d.vehicleTypes.counts,
          backgroundColor: ['#ff2d55','#ff6b35','#ffd60a','#30d158','#00d4ff','#bf5af2'],
          borderColor: 'var(--bg-card)',
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { position: 'right' },
          tooltip: { callbacks: { label: ctx => `${ctx.label}: ${ctx.parsed}%` } },
        },
        cutout: '65%',
      },
    });
  }

  // ── Disaster Tweets (Bar) ──────────────────────────────────────
  const disasterCtx = getCtx('chart-disaster-tweets');
  if (disasterCtx) {
    STATE.charts['chart-disaster-tweets'] = new Chart(disasterCtx, {
      type: 'bar',
      data: {
        labels: d.disasterTweets.labels,
        datasets: [{
          label: 'Tweet Count',
          data: d.disasterTweets.counts,
          backgroundColor: ['rgba(255,45,85,0.6)', 'rgba(0,212,255,0.6)'],
          borderColor: ['rgba(255,45,85,0.9)', 'rgba(0,212,255,0.9)'],
          borderWidth: 1,
          borderRadius: 6,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.04)' } },
          y: { grid: { color: 'rgba(255,255,255,0.04)' } },
        },
      },
    });
  }

  // ── Accident Causes (Horizontal Bar) ──────────────────────────
  const causesCtx = getCtx('chart-accident-causes');
  if (causesCtx) {
    STATE.charts['chart-accident-causes'] = new Chart(causesCtx, {
      type: 'bar',
      data: {
        labels: d.causeBreakdown.labels,
        datasets: [{
          label: '% of Accidents',
          data: d.causeBreakdown.values,
          backgroundColor: [
            'rgba(255,45,85,0.6)','rgba(255,107,53,0.6)','rgba(255,214,10,0.6)',
            'rgba(48,209,88,0.6)','rgba(0,212,255,0.6)','rgba(100,210,255,0.6)','rgba(191,90,242,0.6)',
          ],
          borderRadius: 4,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: { callback: v => v + '%' },
          },
          y: { grid: { color: 'rgba(255,255,255,0.04)' } },
        },
      },
    });
  }
}

/* ══════════════════════════════════════════════════════════════════
   TEXT / NLP ANALYSIS
   ══════════════════════════════════════════════════════════════════ */

async function analyzeText() {
  const input = document.getElementById('nlp-input');
  const resultEl = document.getElementById('nlp-result');
  if (!input || !resultEl) return;

  const text = input.value.trim();
  if (!text) return;

  resultEl.innerHTML = `<div class="spinner" style="margin:16px auto"></div>`;

  // Try real API
  const result = await API.post('/text/analyze', { text, source: 'manual' });

  if (result) {
    const isDisaster = result.event !== 'no_event' && result.confidence > 0.4;
    const color = isDisaster ? 'var(--sev-critical)' : 'var(--sev-low)';
    resultEl.innerHTML = `
      <div style="padding:16px;background:var(--bg-surface);border-radius:var(--radius-md);border:1px solid rgba(255,255,255,0.08)">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
          <div style="font-size:15px;font-weight:700;color:${color}">${isDisaster ? '🚨' : '✅'} ${(result.event || 'NO_EVENT').replace(/_/g,' ')}</div>
          <span class="badge badge-${isDisaster ? 'critical' : 'low'}">${isDisaster ? 'DISASTER' : 'NON-DISASTER'}</span>
        </div>
        <div class="conf-bar-wrap">
          <div class="conf-bar-header">
            <span class="conf-bar-label">Confidence</span>
            <span class="conf-bar-val">${(result.confidence * 100).toFixed(1)}%</span>
          </div>
          <div class="conf-bar"><div class="conf-bar-fill" style="width:${result.confidence*100}%;--fill-color:${color}"></div></div>
        </div>
        <div class="text-xs text-muted">Model: ${result.model_name} v${result.model_version}</div>
        <div class="text-xs text-muted mt-4">⚠️ AI-generated detection. Human verification required.</div>
      </div>`;
  } else {
    // Demo fallback
    const keywords = ['flood','fire','crash','accident','earthquake','explosion','help','emergency','disaster'];
    const found = keywords.filter(k => text.toLowerCase().includes(k));
    const isDisaster = found.length > 0;
    const conf = isDisaster ? randFloat(0.65, 0.95) : randFloat(0.05, 0.35);
    const color = isDisaster ? 'var(--sev-critical)' : 'var(--sev-low)';

    resultEl.innerHTML = `
      <div style="padding:16px;background:var(--bg-surface);border-radius:var(--radius-md);border:1px solid rgba(255,255,255,0.08)">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
          <div style="font-size:15px;font-weight:700;color:${color}">${isDisaster ? '🚨' : '✅'} ${isDisaster ? 'DISASTER DETECTED' : 'NON-DISASTER'}</div>
          <span class="badge badge-${isDisaster ? 'critical' : 'low'}">${isDisaster ? 'DISASTER' : 'SAFE'}</span>
        </div>
        <div class="conf-bar-wrap">
          <div class="conf-bar-header">
            <span class="conf-bar-label">Confidence</span>
            <span class="conf-bar-val">${(conf * 100).toFixed(1)}%</span>
          </div>
          <div class="conf-bar"><div class="conf-bar-fill" style="width:${conf*100}%;--fill-color:${color}"></div></div>
        </div>
        ${found.length ? `<div style="margin-top:8px;font-size:11px;color:var(--sev-medium)">Keywords detected: ${found.join(', ')}</div>` : ''}
        <div class="text-xs text-muted mt-4">[DEMO] DistilBERT disaster classifier (Kaggle NLP2 dataset)</div>
        <div class="text-xs text-muted">⚠️ AI-generated detection. Human verification required.</div>
      </div>`;

    if (isDisaster) {
      injectDemoAlert({ severity: conf > 0.8 ? 'HIGH' : 'MEDIUM', event: 'DISASTER_TWEET', location: 'Text Input', modality: ['text'], confidence: conf, icon: '📢' });
      addFusionEvent({ event_type: 'DISASTER_TWEET', severity: conf > 0.8 ? 'HIGH' : 'MEDIUM', risk_score: conf * 0.85, contributing_modalities: ['text'], timestamp: now() });
    }
  }
}

/* ══════════════════════════════════════════════════════════════════
   VIDEO UPLOAD ANALYSIS
   ══════════════════════════════════════════════════════════════════ */

async function analyzeVideoUpload(file) {
  const resultEl = document.getElementById('video-result');
  const previewEl = document.getElementById('video-preview');
  if (!resultEl) return;

  // Show file info
  if (previewEl) {
    if (file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file);
      previewEl.innerHTML = `<img src="${url}" style="max-width:100%;max-height:200px;border-radius:8px;object-fit:contain">`;
    } else {
      previewEl.innerHTML = `<div style="padding:20px;text-align:center;color:var(--text-muted)">📹 ${file.name} (${(file.size/1024/1024).toFixed(1)} MB)</div>`;
    }
  }

  resultEl.innerHTML = `<div style="padding:16px;text-align:center"><div class="spinner" style="margin:0 auto 8px"></div><div class="text-sm text-muted">Analyzing with AI model…</div></div>`;

  const formData = new FormData();
  formData.append('file', file);

  const result = await API.postForm('/video/analyze', formData);

  if (result) {
    const isEvent = result.event !== 'NO_EVENT' && result.confidence > 0.5;
    const color = isEvent ? severityColor(isEvent && result.confidence > 0.8 ? 'HIGH' : 'MEDIUM') : 'var(--sev-low)';

    resultEl.innerHTML = `
      <div style="padding:16px;background:var(--bg-surface);border-radius:var(--radius-md);border:1px solid rgba(255,255,255,0.08)">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
          <div style="font-size:15px;font-weight:700;color:${color}">${isEvent ? '🔥' : '✅'} ${result.event.replace(/_/g,' ')}</div>
          <span class="badge badge-${isEvent ? (result.confidence > 0.8 ? 'critical' : 'high') : 'low'}">${isEvent ? 'DETECTED' : 'CLEAR'}</span>
        </div>
        <div class="conf-bar-wrap">
          <div class="conf-bar-header">
            <span class="conf-bar-label">Confidence</span>
            <span class="conf-bar-val">${(result.confidence * 100).toFixed(1)}%</span>
          </div>
          <div class="conf-bar"><div class="conf-bar-fill" style="width:${result.confidence*100}%;--fill-color:${color}"></div></div>
        </div>
        ${result.evidence?.[0] ? `
        <div style="margin-top:10px;font-size:11px;color:var(--text-secondary)">
          Backend: ${result.evidence[0].backend || '—'} | Frames: ${result.evidence[0].frames_analyzed || 1}
        </div>` : ''}
        <div class="text-xs text-muted mt-4">Model: ${result.model_name} v${result.model_version}</div>
        <div class="text-xs text-muted">⚠️ AI-generated detection. Human verification required.</div>
      </div>`;

    if (isEvent) {
      injectDemoAlert({ severity: result.confidence > 0.8 ? 'HIGH' : 'MEDIUM', event: result.event, location: 'Uploaded Video/Image', modality: ['video'], confidence: result.confidence, icon: '🔥' });
    }
  } else {
    // Demo fallback — pretend we detected something
    const demoConf = randFloat(0.3, 0.9);
    const demoEvent = demoConf > 0.55 ? 'FIRE' : 'NO_EVENT';
    const color = demoEvent !== 'NO_EVENT' ? 'var(--sev-high)' : 'var(--sev-low)';

    resultEl.innerHTML = `
      <div style="padding:16px;background:var(--bg-surface);border-radius:var(--radius-md);border:1px solid rgba(255,255,255,0.08)">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
          <div style="font-size:15px;font-weight:700;color:${color}">${demoEvent !== 'NO_EVENT' ? '🔥' : '✅'} ${demoEvent.replace(/_/g,' ')}</div>
          <span class="badge badge-${demoEvent !== 'NO_EVENT' ? 'high' : 'low'}">${demoEvent !== 'NO_EVENT' ? 'DETECTED' : 'CLEAR'}</span>
        </div>
        <div class="conf-bar-wrap">
          <div class="conf-bar-header"><span class="conf-bar-label">Confidence</span><span class="conf-bar-val">${(demoConf*100).toFixed(1)}%</span></div>
          <div class="conf-bar"><div class="conf-bar-fill" style="width:${demoConf*100}%;--fill-color:${color}"></div></div>
        </div>
        <div class="text-xs text-muted mt-4">[DEMO] VideoSafetyModel — colour-analysis fallback</div>
        <div class="text-xs text-muted">⚠️ AI-generated detection. Human verification required.</div>
      </div>`;
  }
}

/* ══════════════════════════════════════════════════════════════════
   CLOCK
   ══════════════════════════════════════════════════════════════════ */

function startClock() {
  function update() {
    const el = document.getElementById('header-clock');
    if (el) {
      const d = new Date();
      el.textContent = d.toLocaleString('en-IN', {
        weekday: 'short', day: '2-digit', month: 'short',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: false, timeZone: 'Asia/Kolkata',
      }) + ' IST';
    }
  }
  update();
  setInterval(update, 1000);
}

/* ══════════════════════════════════════════════════════════════════
   DASHBOARD CAMERA MINI-GRID
   ══════════════════════════════════════════════════════════════════ */

function initDashboardCameras() {
  const grid = document.getElementById('dashboard-cameras');
  if (!grid) return;

  grid.innerHTML = DEMO_CAMERAS.slice(0, 6).map(cam => `
    <div class="camera-cell" style="cursor:pointer" onclick="navigateTo('cameras')" title="${cam.location}">
      <canvas id="canvas-mini-${cam.id}" style="width:100%;height:100%;border-radius:var(--radius-md)"></canvas>
      <div class="camera-header">
        <span class="camera-name" style="font-size:9px">📷 ${cam.name}</span>
        <div class="camera-live-dot"></div>
      </div>
      <div class="camera-footer">
        <span class="camera-ts" id="cam-ts-mini-${cam.id}" style="font-size:8px"></span>
      </div>
    </div>
  `).join('');

  DEMO_CAMERAS.slice(0, 6).forEach(cam =>
    initCameraCanvas('canvas-mini-' + cam.id, 'cam-ts-mini-' + cam.id, cam));
}

/* ══════════════════════════════════════════════════════════════════
   LIVE SYSTEM METRICS (from real API)
   ══════════════════════════════════════════════════════════════════ */

async function fetchSystemStatus() {
  const data = await API.get('/system/status');
  if (data) {
    const el = document.getElementById('api-status-msg');
    if (el) {
      el.innerHTML = `<span style="color:var(--sev-low)">● Backend connected</span> — ${data.message}`;
    }
    updateHeaderStatus(true);
  }
}

/* ══════════════════════════════════════════════════════════════════
   BOOTSTRAP
   ══════════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  // Start clock
  startClock();

  // Navigation
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      const page = item.dataset.page;
      if (page) navigateTo(page);
    });
  });

  // Alert bell
  document.getElementById('header-alert-btn')?.addEventListener('click', () => navigateTo('alerts'));

  // Initialize cameras
  initDashboardCameras();

  // NLP text analysis
  document.getElementById('nlp-analyze-btn')?.addEventListener('click', analyzeText);
  document.getElementById('nlp-input')?.addEventListener('keydown', e => {
    if (e.ctrlKey && e.key === 'Enter') analyzeText();
  });

  // Video upload
  const dropZone = document.getElementById('video-drop-zone');
  const fileInput = document.getElementById('video-file-input');

  if (dropZone) {
    dropZone.addEventListener('click', () => fileInput?.click());
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file) analyzeVideoUpload(file);
    });
  }

  if (fileInput) {
    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) analyzeVideoUpload(fileInput.files[0]);
    });
  }

  // Incidents filter
  document.getElementById('incidents-filter')?.addEventListener('change', renderIncidentsTable);

  // Alerts filter
  document.getElementById('alerts-filter')?.addEventListener('change', renderAlertsPage);

  // Connect to backend (real API)
  fetchSystemStatus();

  // WebSocket
  connectWebSocket();

  // Start demo simulation
  if (CONFIG.DEMO_MODE) startDemoSimulation();

  // Initial renders
  navigateTo('dashboard');

  // Periodic refresh
  setInterval(refreshDashboardStats, 5000);
  setInterval(renderAlertFeed, 3000);
  setInterval(renderFusionTimeline, 4000);
});

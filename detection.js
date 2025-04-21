// detection.js - Enhanced with Quiet Mode + Custom Label Picker + Firestore Save/Load + Label Grouping + Search

const detectionModes = {
  home: [
    "person", "dog", "cat", "tv", "remote", "refrigerator", "microwave",
    "chair", "couch", "bed", "tree", "backpack", "cell phone", "umbrella"
  ],
  public: [
    "person", "car", "bus", "bicycle", "motorcycle", "traffic light", "stop sign",
    "bench", "truck", "tree", "backpack", "cell phone", "umbrella"
  ]
};

const cocoLabels = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
  "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
  "hair drier", "toothbrush"
];

let lastDetectionMessage = "";
let quietMode = false;
const socket = io(window.location.origin);

socket.on('speak', (data) => {
  if (data.message && !quietMode) {
    lastDetectionMessage = data.message;
    speak(data.message);
    const logList = document.getElementById("detectionList");
    const newItem = document.createElement("li");
    newItem.textContent = `${new Date().toLocaleTimeString()}: ${data.message}`;
    const placeholder = document.getElementById("placeholder");
    if (placeholder) placeholder.remove();
    logList.prepend(newItem);
    if (logList.children.length > 10) logList.removeChild(logList.lastChild);
  }
});

function speakLastDetection() {
  if (!quietMode && lastDetectionMessage) speak(lastDetectionMessage);
  else if (!quietMode) speak("No detection has been received yet.");
}

function clearDetectionLog() {
  document.getElementById("detectionList").innerHTML = "<li id='placeholder'>No detections yet.</li>";
  lastDetectionMessage = "";
  if (!quietMode) speak("Detection log cleared.");
}

function updateMode() {
  const mode = document.getElementById("modeSelect").value;
  document.getElementById("customLabelBox").style.display = mode === "custom" ? "block" : "none";
  if (!quietMode) speak(mode === "home" ? "Home mode activated" : mode === "public" ? "Public mode activated" : "Custom mode activated");
}

function updateConfig() {
  const mode = document.getElementById("modeSelect").value;
  let selectedLabels = [];
  if (mode === "custom") {
    document.querySelectorAll("#customLabels input[type=checkbox]:checked").forEach(cb => {
      selectedLabels.push(cb.value);
    });
    localStorage.setItem("custom_labels", JSON.stringify(selectedLabels));
    fetch('/save_custom_labels', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ labels: selectedLabels })
    });
  } else {
    selectedLabels = detectionModes[mode];
  }
  fetch('config', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ labels: selectedLabels })
  })
  .then(res => res.json())
  .then(data => {
    if (!quietMode) speak(`Detection config updated for ${mode} mode.`);
  })
  .catch(err => {
    if (!quietMode) speak("Failed to update detection config.");
  });
}

function generateCustomLabelUI() {
  const container = document.getElementById("customLabels");
  container.innerHTML = "";

  const searchInput = document.createElement("input");
  searchInput.type = "text";
  searchInput.placeholder = "Search labels...";
  searchInput.oninput = () => filterLabels(searchInput.value.toLowerCase());
  container.appendChild(searchInput);

  container.appendChild(document.createElement("br"));

  const controls = document.createElement("div");
  controls.innerHTML = `
    <button onclick="selectAllLabels()">Select All</button>
    <button onclick="deselectAllLabels()">Deselect All</button>
    <br><br>
  `;
  container.appendChild(controls);

  const groups = {
    Indoor: ["tv", "remote", "chair", "couch", "bed", "sink", "microwave", "refrigerator", "laptop", "clock"],
    Outdoor: ["car", "bus", "truck", "traffic light", "bench", "bicycle", "stop sign"],
    Animals: ["dog", "cat", "cow", "horse", "zebra", "elephant", "bird", "sheep"]
  };

  const saved = JSON.parse(localStorage.getItem("custom_labels") || "[]");

  for (const [group, labels] of Object.entries(groups)) {
    const groupTitle = document.createElement("strong");
    groupTitle.textContent = group;
    container.appendChild(groupTitle);
    container.appendChild(document.createElement("br"));

    labels.forEach(label => {
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = label;
      cb.id = `label_${label}`;
      cb.checked = saved.includes(label);

      const lbl = document.createElement("label");
      lbl.htmlFor = cb.id;
      lbl.textContent = label;

      container.appendChild(cb);
      container.appendChild(lbl);
      container.appendChild(document.createElement("br"));
    });
    container.appendChild(document.createElement("br"));
  }
}

function filterLabels(query) {
  document.querySelectorAll("#customLabels label").forEach(label => {
    const checkbox = document.getElementById(label.htmlFor);
    const match = label.textContent.toLowerCase().includes(query);
    label.style.display = match ? "inline" : "none";
    checkbox.style.display = match ? "inline" : "none";
  });
}

function selectAllLabels() {
  document.querySelectorAll("#customLabels input[type=checkbox]").forEach(cb => cb.checked = true);
}

function deselectAllLabels() {
  document.querySelectorAll("#customLabels input[type=checkbox]").forEach(cb => cb.checked = false);
}

function updateStatusPanel() {
  fetch('/get_config')
    .then(res => res.json())
    .then(data => {
      const modeText = data.indoor_mode ? '🏠 Home' :
                       data.custom_mode ? '🧩 Custom' : '🌆 Public';
      document.getElementById("modeLabel").textContent = modeText;
      document.getElementById("activeLabels").textContent = (data.filter_classes || []).join(", ");
      document.getElementById("toggleDetection").checked = data.detection_active;
      document.getElementById("toggleVoice").checked = data.voice_alert_enabled;
    });
}

document.getElementById("toggleDetection").addEventListener("change", (e) => {
  fetch("/update_config", {
    method: "POST",
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ detection_active: e.target.checked })
  });
});

document.getElementById("toggleVoice").addEventListener("change", (e) => {
  fetch("/update_config", {
    method: "POST",
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ voice_alert_enabled: e.target.checked })
  });
});

document.addEventListener("DOMContentLoaded", () => {
  updateMode();
  generateCustomLabelUI();
  fetch("/status")
    .then(res => res.json())
    .then(data => { quietMode = !!data.quiet_mode_enabled });
  updateStatusPanel();
});

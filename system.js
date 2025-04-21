// system.js - Enhanced with Smart Voice Commands + Log Deletion Prompt + Quiet Mode Toggle + Wake Word Toggle

let wakeListening = false;
let wakeWordEnabled = true; // ✅ NEW: Toggle state

function toggleIndoorMode() {
  const enabled = document.getElementById("indoorToggle").checked;
  fetch("/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ indoor_mode: enabled })
  })
    .then(res => res.json())
    .then(() => {
      speak(enabled ? "Indoor mode enabled" : "Indoor mode disabled");
    });
}

function toggleQuietMode() {
  const enabled = document.getElementById("quietToggle").checked;
  fetch("/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ quiet_mode_enabled: enabled })
  })
    .then(res => res.json())
    .then(() => {
      speak(enabled ? "Quiet mode enabled" : "Quiet mode disabled");
    });
}

function toggleSmartVideo() {
  const enabled = document.getElementById("smartVideoToggle").checked;
  fetch("/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smart_video_enabled: enabled })
  }).then(() => {
    speak(enabled ? "Smart video recording enabled" : "Smart video recording disabled");
  });
}

function toggleResMode() {
  const enabled = document.getElementById("resToggle").checked;
  fetch("/resolution_mode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ high_res_mode: enabled })
  }).then(() => {
    speak(enabled ? "High resolution preview enabled" : "Low resolution preview enabled");
  });
}


function toggleWakeWord() {
  wakeWordEnabled = document.getElementById("wakeToggle").checked;
  if (wakeWordEnabled && !wakeListening) {
    wakeRecognizer.start();
    wakeListening = true;
    speak("Wake word enabled");
  } else if (!wakeWordEnabled && wakeListening) {
    wakeRecognizer.stop();
    wakeListening = false;
    speak("Wake word disabled");
  }
}

function checkStatus() {
  fetch("/status")
    .then(res => res.json())
    .then(data => {
      document.getElementById("deviceName").textContent = navigator.userAgent;
      document.getElementById("currentMode").textContent = data.mode || "--";
      document.getElementById("quietStatus").textContent = data.quiet_mode_enabled ? "ON" : "OFF";
      document.getElementById("quietToggle").checked = !!data.quiet_mode_enabled;
      document.getElementById("wakeToggle").checked = wakeWordEnabled;
      speak(`Battery at ${data.battery} percent. Mode is ${data.mode}. Quiet mode is ${data.quiet_mode_enabled ? 'on' : 'off'}. Health status is ${data.health}`);
    });
}

function toggleLog() {
  const log = document.getElementById("log");
  log.style.display = log.style.display === "none" ? "block" : "none";
  if (log.style.display === "block") {
    log.textContent = "Loading...";
    fetch("/status")
      .then(res => res.json())
      .then(data => {
        log.innerHTML = `<strong>Status:</strong><br>Battery: ${data.battery}%<br>Health: ${data.health}`;
      });
  }
}

function deleteLogs() {
  const logs = [
    { key: "battery_logs", label: "Battery Logs" },
    { key: "motion_logs", label: "Motion Logs" },
    { key: "detection_logs", label: "Detection Logs" },
    { key: "location_logs", label: "Location Logs" },
    { key: "system_health_logs", label: "System Health Logs" },
    { key: "video_logs", label: "Video Logs" }
  ];

  const selected = prompt("🧹 Which logs do you want to delete?\nOptions:\n- all\n- battery, ultrasonic, detection...\n\n(Type comma-separated keys or 'all')");

  if (!selected) return speak("Log deletion cancelled.");
  const keys = selected.trim().toLowerCase() === "all"
    ? logs.map(l => l.key)
    : selected.split(",").map(k => k.trim());

  fetch("/delete_logs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ keys })
  })
    .then(res => res.json())
    .then(res => {
      if (res.status === "success") {
        const deleted = logs.filter(l => keys.includes(l.key)).map(l => l.label).join(", ");
        speak(`Deleted logs: ${deleted}`);
      } else {
        speak(`Failed to delete logs. ${res.message}`);
      }
    })
    .catch(() => speak("Error while deleting logs"));
}

const sectionNames = {
  dashboard: "Dashboard",
  nav: "Navigation Mode",
  detect: "Detection Panel",
  sensor: "Sensor Settings",
  system: "System Tools",
  voice: "Voice Commands"
};

function switchSection(id, el = null, speakEnabled = true) {
  document.querySelectorAll(".section").forEach(sec => sec.classList.remove("active"));
  const target = document.getElementById(id);
  if (target) target.classList.add("active");

  document.querySelectorAll(".nav-btn").forEach(btn => btn.classList.remove("active"));
  if (el) el.classList.add("active");

  const label = sectionNames[id] || id;

  // 🛡️ Prevent repeating same message twice
  if (speakEnabled && window.lastSpokenSection !== label) {
    speak(`${label} activated`);
    window.lastSpokenSection = label;
  }
}


let wakeRecognizer = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
wakeRecognizer.continuous = true;
wakeRecognizer.interimResults = false;
wakeRecognizer.lang = "en-US";

wakeRecognizer.onresult = function (event) {
  const transcript = event.results[event.results.length - 1][0].transcript.toLowerCase().trim();
  if (wakeWordEnabled && (transcript.includes("hat") || transcript.includes("smart hat"))) {
    speak("Yes? Listening now.");
    recognition.start();
  }
};

wakeRecognizer.onerror = function (event) {
  console.error("Wake recognizer error:", event.error);
};

function startWakeWordListener() {
  if (wakeWordEnabled && !wakeListening) {
    wakeRecognizer.start();
    wakeListening = true;
    console.log("Wake word listener activated");
  }
}

const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = "en-US";
recognition.interimResults = false;
recognition.maxAlternatives = 1;

function startListening() {
  speak("Listening. Please say a command.");
  recognition.start();
}

recognition.onresult = function (event) {
  const transcript = event.results[0][0].transcript.toLowerCase().trim();
  console.log("🎤 Heard:", transcript);
  handleVoiceCommand(transcript);
};

recognition.onerror = function (event) {
  console.error("Speech recognition error:", event.error);
  speak("Sorry, I didn't catch that.");
};

function shutdownPi() {
  console.log("Shutdown function triggered...");  // Debugging line
  fetch("/shutdown", { method: "POST" })
    .then(res => res.json())
    .then(data => {
      console.log("Shutdown response:", data);  // Debugging line
      speak(data.message);
    })
    .catch(err => {
      console.error("Shutdown error:", err);   // Debugging line
      speak("Shutdown failed");
    });
}

function switchAnalyticsView(section) {
  const frame = document.getElementById("analyticsFrame");
  if (frame) {
    frame.src = `/analytics/${section}`;
    speak(`Switched to ${section} analytics`);
  } else {
    speak("Analytics panel is not available.");
  }
}


const voiceCommands = {
  "dashboard": () => switchSection("dashboard", null, false),
  "open dashboard": () => switchSection("dashboard", null, false),
  "navigation": () => switchSection("nav", null, false),
  "open navigation": () => switchSection("nav", null, false),
  "detection": () => switchSection("detect", null, false),
  "open detection": () => switchSection("detect", null, false),
  "sensor": () => switchSection("sensor", null, false),
  "open sensors": () => switchSection("sensor", null, false),
  "system": () => switchSection("system", null, false),
  "open system": () => switchSection("system", null, false),
  "voice": () => switchSection("voice", null, false),
  "voice commands": () => switchSection("voice", null, false),
  "show performance analytics": () => switchAnalyticsView("performance"),
  "show detection graphs": () => switchAnalyticsView("detection"),
  "show interaction data": () => switchAnalyticsView("interaction"),
  "show video logs": () => switchAnalyticsView("videos"),


  "check battery": () => checkStatus(),

  "enable indoor mode": () => {
    document.getElementById("indoorToggle").checked = true;
    toggleIndoorMode();
  },
  "disable indoor mode": () => {
    document.getElementById("indoorToggle").checked = false;
    toggleIndoorMode();
  },
  "enable quiet mode": () => {
    document.getElementById("quietToggle").checked = true;
    toggleQuietMode();
  },
  "disable quiet mode": () => {
    document.getElementById("quietToggle").checked = false;
    toggleQuietMode();
  },
  "enable wake word": () => {
    document.getElementById("wakeToggle").checked = true;
    toggleWakeWord();
  },
  "disable wake word": () => {
    document.getElementById("wakeToggle").checked = false;
    toggleWakeWord();
  },

  "shut down": () => {
    shutdownPi();
    speak("Shutting down the system");
  },

  "enable smart video": () => {
  document.getElementById("smartVideoToggle").checked = true;
  toggleSmartVideo();
},
"disable smart video": () => {
  document.getElementById("smartVideoToggle").checked = false;
  toggleSmartVideo();
},
"enable high resolution": () => {
  document.getElementById("resToggle").checked = true;
  toggleResMode();
},
"disable high resolution": () => {
  document.getElementById("resToggle").checked = false;
  toggleResMode();
},

  "delete logs": () => deleteLogs(),
  "repeat detection": () => speakLastDetection(),
  "repeat message": () => speak(lastSpokenMessage),

  "where am i": () => speakDetailedLocation(),
  "get location": () => speakDetailedLocation(),

  "start voice navigation": () => startVoiceSearch(),
  "navigate to": () => startVoiceSearch(),

  "pause navigation": () => {
    toggleTracking();
    speak("Navigation paused");
  },
  "resume navigation": () => {
    toggleTracking();
    speak("Navigation resumed");
  }
};


function refreshWifiStatus() {
    fetch('/wifi_status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('wifi-ssid').textContent = data.ssid;
            document.getElementById('wifi-signal').textContent = data.signal;
            document.getElementById('wifi-ip').textContent = data.ip;
        })
        .catch(error => {
            console.error("Failed to fetch Wi-Fi status:", error);
        });
}

// Auto refresh on load
window.addEventListener('load', refreshWifiStatus);


function handleVoiceCommand(transcript) {
  for (const phrase in voiceCommands) {
    if (transcript.includes(phrase)) {
      voiceCommands[phrase]();
      return;
    }
  }
  speak("Sorry, I didn't understand that command.");
}

// Start wake word listening on load
window.addEventListener("DOMContentLoaded", startWakeWordListener);
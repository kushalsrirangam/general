// sensor.js - Handles ultrasonic thresholds and live video fullscreen toggle with quiet mode support

function applyThresholds() {
  const thresholds = {
    "Left Front": parseInt(document.getElementById("Left Front").value),
    "Left Middle": parseInt(document.getElementById("Left Middle").value),
    "Left Rear": parseInt(document.getElementById("Left Rear").value),
    "Right Front": parseInt(document.getElementById("Right Front").value),
    "Right Middle": parseInt(document.getElementById("Right Middle").value),
    "Right Rear": parseInt(document.getElementById("Right Rear").value),
  };

  fetch("/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ultrasonic_thresholds: thresholds })
  })
    .then(res => res.json())
    .then(data => {
      if (!Speech?.quiet) speak("Thresholds updated successfully");
    })
    .catch(err => {
      console.error("Failed to update thresholds:", err);
      if (!Speech?.quiet) speak("Failed to update thresholds");
    });
}

function toggleFullScreen(el) {
  if (!document.fullscreenElement) {
    el.requestFullscreen().catch(err => {
      console.error(`Error attempting to enable full-screen mode: ${err.message}`);
    });
  } else {
    document.exitFullscreen();
  }
}
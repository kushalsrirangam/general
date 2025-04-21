// dashboard.js - Handles the Dashboard functionality and real-time data updates

// Fetch motion status every 5 seconds
function updateMotionStatus() {
  fetch("/motion-status")
    .then(response => response.json())
    .then(data => {
      document.getElementById("motionStatus").textContent = data.motion;
    })
    .catch(err => console.error("Error fetching motion status:", err));
}

// Fetch analytics data every 5 seconds
function updateAnalytics() {
  fetch("/analytics")  // Assuming Flask has an endpoint for this
    .then(response => response.json())
    .then(data => {
      // Update analytics section
      document.getElementById("analytics").innerHTML = `
        <strong>Temperature:</strong> ${data.temperature} °C<br>
        <strong>Humidity:</strong> ${data.humidity} %<br>
        <strong>Other Metrics:</strong> ${data.otherMetrics}
      `;
    })
    .catch(err => console.error("Error fetching analytics data:", err));
}

// Call these functions on load to initialize
function initializeDashboard() {
  updateMotionStatus();  // Initial fetch for motion status
  updateAnalytics();  // Initial fetch for analytics data

  setInterval(updateMotionStatus, 5000);  // Update motion status every 5 seconds
  setInterval(updateAnalytics, 5000);  // Update analytics every 5 seconds
}

// Call initializeDashboard when the page is loaded
window.addEventListener("DOMContentLoaded", initializeDashboard);

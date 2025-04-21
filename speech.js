// speech.js - Updated with full Speech object and Quiet Mode check

const Speech = {
  synth: window.speechSynthesis,
  queue: [],
  speaking: false,
  muted: false,
  pitch: 1,
  rate: 1,
  voicePref: 'female',
  ready: false,
  quiet: false,

  init() {
    this.loadVoices(() => {
      this.ready = true;
      this.log("Speech system ready");
      this.populateVoiceSelector();
    });

    // Fetch quiet mode status on init
    fetch("/status")
      .then(res => res.json())
      .then(data => {
        this.quiet = !!data.quiet_mode_enabled;
      });
  },

  loadVoices(callback) {
    let voices = this.synth.getVoices();
    if (voices.length !== 0) {
      callback();
    } else {
      this.synth.onvoiceschanged = () => {
        voices = this.synth.getVoices();
        callback();
      };
    }
  },

  getVoice() {
    const voices = this.synth.getVoices();
    const selected = document.getElementById('voiceSelector')?.value || 'female';
    const match = voices.find(v =>
      selected === 'female'
        ? /(female|zira|samantha|karen)/i.test(v.name)
        : /(male|david|daniel|fred|george)/i.test(v.name)
    );
    this.log(`[Voice Picked] ${match?.name || voices[0]?.name}`);
    return match || voices[0];
  },

  speak(msg, options = {}) {
    if (this.muted || this.quiet || !msg || !this.ready) return;

    const utter = new SpeechSynthesisUtterance(msg);
    utter.pitch = options.pitch || this.pitch;
    utter.rate = options.rate || this.rate;
    utter.voice = this.getVoice();

    utter.onend = () => {
      this.speaking = false;
      this.processQueue();
    };

    this.queue.push(utter);
    this.processQueue();
  },

  processQueue() {
    if (this.speaking || this.queue.length === 0) return;
    this.speaking = true;
    const next = this.queue.shift();
    this.synth.speak(next);
  },

  mute(toggle) {
    this.muted = toggle;
    if (toggle) this.synth.cancel();
  },

  log(msg) {
    console.log('[Speech]', msg);
  },

  populateVoiceSelector() {
    const selector = document.getElementById('voiceSelector');
    if (!selector) return;

    selector.addEventListener('change', () => {
      this.voicePref = selector.value;
      this.log(`Voice preference changed to: ${this.voicePref}`);
      pushMessageToFlask(`Switched to ${this.voicePref} voice`);
    });
  }
};

// 🔊 Global speak function
function speak(msg, options) {
  Speech.speak(msg, options);
}

// 🔇 Mute checkbox
function toggleMute() {
  const toggle = document.getElementById('muteToggle');
  Speech.mute(toggle.checked);
}

// 🔊 Prevent duplicate messages within 1 second
let lastSpokenMessage = "";
let lastSpokenTime = 0;

// 🧠 On load + setup SocketIO voice listener
window.addEventListener('DOMContentLoaded', () => {
  Speech.init();

  document.body.addEventListener('click', () => {
    if (!Speech.ready && !Speech.speaking) {
      speak("Smart Hat ready");
    }
  }, { once: true });

  const socket = io();
  socket.on('connect', () => console.log("[SocketIO] Connected"));

  socket.on('speak', (data) => {
    if (!Speech.muted && !Speech.quiet && data?.message) {
      const now = Date.now();
      if (data.message === lastSpokenMessage && now - lastSpokenTime < 1000) {
        console.log("[Speech] Duplicate message skipped:", data.message);
        return;
      }
      lastSpokenMessage = data.message;
      lastSpokenTime = now;
      console.log("[SocketIO] Speaking:", data.message);
      speak(data.message);
    }
  });
});
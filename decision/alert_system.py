"""
alert_system.py — Decision layer
Escalating Alert Response: graduated visual/audio cues to prevent alarm fatigue.

Every system in existing drowsiness-detection literature is threshold → alarm.
Real driver-monitoring products that people actually keep turned on address
alarm fatigue/habituation through escalating response: a subtle ambient cue
first, a gentle prompt second, only a loud alert if the driver doesn't respond.

Alert escalation design:
    ALERT→WARNING:  Subtle amber pulsing border + soft 2-tone chime
    WARNING (held): Amber pulse intensifies + gentle TTS every 30s
    WARNING→CRITICAL: Full red border + flash + large "PULL OVER" overlay
    CRITICAL (held): Red flash + urgent TTS repeats every 15s
    Distraction:     Orange bottom bar with pulsing opacity + soft chime
    Face lost:       Red border + urgent voice prompt
    Drowsy nods:     Amber inner border + specific voice warning

The pulsing visual effect uses sine-wave modulated border thickness/opacity,
creating an organic, non-jarring visual cue that increases urgency without
the binary on/off harshness of a solid red border.
"""

import time
import math

import cv2
import queue
import threading

try:
    import winsound
    _WINSOUND_OK = True
except Exception:
    _WINSOUND_OK = False

try:
    import numpy as np
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    _PYGAME_OK = True
except Exception:
    _PYGAME_OK = False

# ── Non-blocking Threaded TTS Engine ───────────────────────────────
_speech_queue = queue.Queue(maxsize=3)

def _tts_worker():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        while True:
            text = _speech_queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
            finally:
                _speech_queue.task_done()
    except Exception:
        pass

_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()

def _speak(text):
    """Non-blocking text-to-speech dispatch."""
    try:
        if _speech_queue.qsize() < 2:
            _speech_queue.put_nowait(text)
    except Exception:
        pass


STATE_COLORS = {  # BGR
    "ALERT": (0, 200, 0),
    "WARNING": (0, 165, 255),
    "CRITICAL": (0, 0, 255),
}

DISTRACTION_COLOR = (255, 128, 0)
NOD_ALERT_COLOR = (0, 140, 255)  # deep amber, distinct from WARNING's orange


# ─── Audio helpers ───────────────────────────────────────────────

def _beep(frequency=440, duration_ms=300):
    if _PYGAME_OK:
        try:
            sample_rate = 44100
            n_samples = int(sample_rate * duration_ms / 1000)
            t = np.linspace(0, duration_ms / 1000, n_samples, False)
            wave = np.sin(frequency * t * 2 * np.pi)
            audio = (wave * 32767).astype(np.int16)
            stereo = np.column_stack([audio, audio])
            pygame.sndarray.make_sound(stereo).play()
            return
        except Exception:
            pass
    if _WINSOUND_OK:
        try:
            winsound.Beep(int(frequency), int(duration_ms))
        except Exception:
            pass


def _gentle_chime():
    """Soft two-tone chime for WARNING onset — less jarring than a beep."""
    if _PYGAME_OK:
        try:
            sample_rate = 44100
            duration = 0.16
            n = int(sample_rate * duration)
            t = np.linspace(0, duration, n, False)
            mid = n // 2
            wave = np.zeros(n)
            wave[:mid] = np.sin(400 * t[:mid] * 2 * np.pi)
            wave[mid:] = np.sin(500 * t[mid:] * 2 * np.pi)
            envelope = np.ones(n)
            fade_len = int(n * 0.2)
            envelope[:fade_len] = np.linspace(0, 1, fade_len)
            envelope[-fade_len:] = np.linspace(1, 0, fade_len)
            wave *= envelope * 0.6
            audio = (wave * 32767).astype(np.int16)
            stereo = np.column_stack([audio, audio])
            pygame.sndarray.make_sound(stereo).play()
            return
        except Exception:
            pass
    if _WINSOUND_OK:
        try:
            winsound.Beep(500, 100)
        except Exception:
            pass


def _urgent_alarm():
    """Urgent repeating alarm tone for CRITICAL state."""
    if _PYGAME_OK:
        try:
            sample_rate = 44100
            duration = 0.4
            n = int(sample_rate * duration)
            t = np.linspace(0, duration, n, False)
            carrier = np.sin(900 * t * 2 * np.pi)
            modulator = 0.5 + 0.5 * np.sin(8 * t * 2 * np.pi)
            wave = carrier * modulator
            audio = (wave * 32767).astype(np.int16)
            stereo = np.column_stack([audio, audio])
            pygame.sndarray.make_sound(stereo).play()
            return
        except Exception:
            pass
    if _WINSOUND_OK:
        try:
            winsound.Beep(900, 250)
        except Exception:
            pass


# ─── Visual effect helpers ───────────────────────────────────────

def _pulse_value(now, frequency=1.0, min_val=0.3, max_val=1.0):
    """
    Generate a smooth sine-wave pulse value between min_val and max_val.
    frequency: pulses per second
    """
    phase = math.sin(2 * math.pi * frequency * now)
    return min_val + (max_val - min_val) * (0.5 + 0.5 * phase)


def _draw_pulsing_border(frame, color, now, pulse_freq=1.0,
                          min_thickness=2, max_thickness=12):
    """Draw a border with pulsing thickness for organic urgency feel."""
    h, w = frame.shape[:2]
    pulse = _pulse_value(now, pulse_freq)
    thickness = int(min_thickness + (max_thickness - min_thickness) * pulse)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)


def _draw_flash_overlay(frame, color, now, flash_freq=2.0, max_alpha=0.15):
    """Draw a semi-transparent full-frame flash for CRITICAL state."""
    pulse = _pulse_value(now, flash_freq, 0.0, max_alpha)
    if pulse > 0.02:
        overlay = frame.copy()
        overlay[:] = color
        cv2.addWeighted(overlay, pulse, frame, 1 - pulse, 0, frame)


def _draw_pull_over_overlay(frame, now):
    """Draw the large PULL OVER text overlay for CRITICAL state."""
    h, w = frame.shape[:2]
    pulse = _pulse_value(now, 1.5, 0.6, 1.0)
    alpha = pulse * 0.5

    # Semi-transparent dark background band
    ov = frame.copy()
    band_y1 = h // 2 - 40
    band_y2 = h // 2 + 40
    cv2.rectangle(ov, (0, band_y1), (w, band_y2), (0, 0, 60), -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

    # Pulsing text
    text = "PULL OVER — FATIGUE DETECTED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(w / 640, 1.2)
    (tw, th), _ = cv2.getTextSize(text, font, scale, 3)
    tx = (w - tw) // 2
    ty = h // 2 + th // 2
    intensity = int(255 * pulse)
    cv2.putText(frame, text, (tx, ty), font, scale,
                (0, 0, intensity), 3, cv2.LINE_AA)


# ─── Main AlertSystem class ─────────────────────────────────────

class AlertSystem:
    """
    Escalating alert system with graduated visual/audio responses.

    Design principle: a monitoring system people will actually keep
    turned on must not produce alarm fatigue. Subtle cues first,
    escalating only when the driver doesn't respond.
    """

    def __init__(self, nod_alert_hold_seconds=7.0,
                 warning_tts_interval=30.0,
                 critical_tts_interval=15.0,
                 critical_alarm_interval=3.0):
        self._last_state = None
        self._last_distraction = False
        self._last_face_lost = False
        self._last_nod_alert_input = False
        self.nod_alert_hold_seconds = nod_alert_hold_seconds
        self._nod_alert_until = None

        # Escalating response timing
        self.warning_tts_interval = warning_tts_interval
        self.critical_tts_interval = critical_tts_interval
        self.critical_alarm_interval = critical_alarm_interval
        self._last_warning_tts = None
        self._last_critical_tts = None
        self._last_critical_alarm = None
        self._warning_enter_time = None
        self._critical_enter_time = None
        self._distraction_start = None

    def update(self, frame, state, distraction_flag=False, face_lost_alert=False,
               drowsy_nod_alert=False, gaze_distraction_flag=False, now=None):
        now = now if now is not None else time.time()
        h, w = frame.shape[:2]

        is_any_distraction = distraction_flag or gaze_distraction_flag
        alert_fired = False

        # ── State-dependent visual rendering ──────────────────────

        if face_lost_alert:
            # Face lost: urgent red border + text
            color = (0, 0, 255)
            _draw_pulsing_border(frame, color, now, pulse_freq=2.0,
                                 min_thickness=6, max_thickness=14)
            cv2.putText(frame, "FACE NOT DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        elif state == "CRITICAL":
            # CRITICAL: full red flash + pulsing border + PULL OVER overlay
            _draw_flash_overlay(frame, (0, 0, 180), now, flash_freq=2.0)
            _draw_pulsing_border(frame, (0, 0, 255), now, pulse_freq=2.0,
                                 min_thickness=8, max_thickness=16)
            _draw_pull_over_overlay(frame, now)
            cv2.putText(frame, "CRITICAL", (w - 160, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        elif state == "WARNING":
            # WARNING: amber pulsing border (subtle but noticeable)
            _draw_pulsing_border(frame, (0, 165, 255), now, pulse_freq=0.8,
                                 min_thickness=3, max_thickness=10)
            # Subtle warning text
            pulse_alpha = _pulse_value(now, 0.8, 0.5, 1.0)
            intensity = int(200 * pulse_alpha)
            cv2.putText(frame, "DROWSINESS BUILDING",
                        (w - 280, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, intensity, 255), 2)

        else:
            # ALERT: calm green border (solid, thin)
            color = STATE_COLORS.get(state, (255, 255, 255))
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 2)
            cv2.putText(frame, state, (w - 160, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # ── Distraction overlay ───────────────────────────────────

        if is_any_distraction:
            label = "DISTRACTION (GAZE)" if gaze_distraction_flag else "DISTRACTION"
            # Pulsing opacity for distraction label
            pulse = _pulse_value(now, 1.2, 0.5, 1.0)
            intensity = int(255 * pulse)
            cv2.putText(frame, label, (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (intensity, int(128 * pulse), 0), 2)

        # ── Audio escalation logic ────────────────────────────────

        # State transition alerts
        if state != self._last_state:
            alert_fired = True

            if state == "WARNING":
                self._warning_enter_time = now
                self._last_warning_tts = now
                _gentle_chime()

            elif state == "CRITICAL":
                self._critical_enter_time = now
                self._last_critical_tts = now
                self._last_critical_alarm = now
                _urgent_alarm()
                _speak("Please pull over, fatigue detected")

            elif state == "ALERT":
                # Recovery — reset escalation timers cleanly
                self._warning_enter_time = None
                self._critical_enter_time = None
                self._last_warning_tts = None
                self._last_critical_tts = None
                self._last_critical_alarm = None

            self._last_state = state

        # Sustained WARNING: gentle TTS reminders at intervals (only after warning_tts_interval)
        if state == "WARNING" and self._warning_enter_time is not None:
            if self._last_warning_tts is not None and (now - self._last_warning_tts) >= self.warning_tts_interval:
                _speak("You have been driving for a while, consider a break")
                self._last_warning_tts = now
                alert_fired = True

        # Sustained CRITICAL: repeated alarm + TTS at intervals
        if state == "CRITICAL":
            if self._last_critical_alarm is not None and (now - self._last_critical_alarm) >= self.critical_alarm_interval:
                _urgent_alarm()
                self._last_critical_alarm = now
                alert_fired = True

            if self._last_critical_tts is not None and (now - self._last_critical_tts) >= self.critical_tts_interval:
                _speak("Please pull over immediately")
                self._last_critical_tts = now
                alert_fired = True

        # Distraction onset (debounced by 0.5s to prevent audio jitter)
        if is_any_distraction:
            if self._distraction_start is None:
                self._distraction_start = now
            if (now - self._distraction_start) >= 0.5 and not self._last_distraction:
                _gentle_chime()
                alert_fired = True
                self._last_distraction = True
        else:
            self._distraction_start = None
            self._last_distraction = False

        # Face lost onset
        if face_lost_alert and not self._last_face_lost:
            alert_fired = True
            _beep(frequency=900, duration_ms=400)
            _speak("Please check your driving position")
        self._last_face_lost = face_lost_alert

        # --- sticky nod alert ---
        if drowsy_nod_alert and not self._last_nod_alert_input:
            alert_fired = True
            self._nod_alert_until = now + self.nod_alert_hold_seconds
            _beep(frequency=850, duration_ms=350)
            _speak("You seem drowsy, please take a break")
        self._last_nod_alert_input = drowsy_nod_alert

        if self._nod_alert_until is not None and now < self._nod_alert_until:
            _draw_pulsing_border(frame, NOD_ALERT_COLOR, now, pulse_freq=1.0,
                                 min_thickness=4, max_thickness=10)
            cv2.putText(frame, "DROWSY - REPEATED NODDING", (20, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, NOD_ALERT_COLOR, 2)
        elif self._nod_alert_until is not None and now >= self._nod_alert_until:
            self._nod_alert_until = None

        return frame, alert_fired

    def reset_alerts(self):
        """Manually clear any active sticky alerts or voice timers."""
        self._nod_alert_until = None
        self._distraction_start = None
        self._last_distraction = False
        self._last_face_lost = False
        # Cleanly reset escalation states/timers to silence alarms
        self._last_state = "ALERT"
        self._warning_enter_time = None
        self._critical_enter_time = None
        self._last_warning_tts = None
        self._last_critical_tts = None
        self._last_critical_alarm = None


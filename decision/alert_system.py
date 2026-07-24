import time

import cv2

try:
    import numpy as np
    import pygame
    pygame.mixer.init()
    _PYGAME_OK = True
except Exception:
    _PYGAME_OK = False

try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _PYTTSX3_OK = True
except Exception:
    _PYTTSX3_OK = False


STATE_COLORS = {  # BGR
    "ALERT": (0, 200, 0),
    "WARNING": (0, 165, 255),
    "CRITICAL": (0, 0, 255),
}

DISTRACTION_COLOR = (255, 128, 0)
NOD_ALERT_COLOR = (0, 140, 255)  # deep amber, distinct from WARNING's orange


def _beep(frequency=440, duration_ms=300):
    if not _PYGAME_OK:
        return
    try:
        sample_rate = 44100
        n_samples = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, n_samples, False)
        wave = np.sin(frequency * t * 2 * np.pi)
        audio = (wave * 32767).astype(np.int16)
        stereo = np.column_stack([audio, audio])
        pygame.sndarray.make_sound(stereo).play()
    except Exception:
        pass


def _speak(text):
    if not _PYTTSX3_OK:
        return
    try:
        _tts_engine.say(text)
        _tts_engine.runAndWait()
    except Exception:
        pass


class AlertSystem:
    """
    Draws a state-colored HUD border + labels, and fires audio/
    voice alerts on transitions (not every frame, so it doesn't
    spam a beep 30x/second).

    drowsy_nod_alert is different from the others: it's STICKY.
    Repeated nodding in a short window is a strong fatigue signal
    on its own (see nod_rate.py) that shouldn't get diluted into
    the composite score's hysteresis, and a single-frame flash is
    too easy to miss if the driver is genuinely dozing. Once
    triggered, the banner + voice line stay up for
    nod_alert_hold_seconds regardless of what happens next frame.
    """

    def __init__(self, nod_alert_hold_seconds=7.0):
        self._last_state = None
        self._last_distraction = False
        self._last_face_lost = False
        self._last_nod_alert_input = False
        self.nod_alert_hold_seconds = nod_alert_hold_seconds
        self._nod_alert_until = None

    def update(self, frame, state, distraction_flag=False, face_lost_alert=False,
               drowsy_nod_alert=False, now=None):
        now = now if now is not None else time.time()
        h, w = frame.shape[:2]

        # Face-lost overrides the normal state color -- if the
        # driver's face isn't visible at all, that matters more
        # right now than whatever the last computed score was.
        if face_lost_alert:
            color = (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 12)
            cv2.putText(frame, "FACE NOT DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            color = STATE_COLORS.get(state, (255, 255, 255))
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 12)
            cv2.putText(frame, state, (w - 160, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if distraction_flag:
            cv2.putText(frame, "DISTRACTION", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, DISTRACTION_COLOR, 2)

        alert_fired = False

        if state != self._last_state:
            alert_fired = True
            if state == "WARNING":
                _beep(frequency=600, duration_ms=200)
            elif state == "CRITICAL":
                _beep(frequency=900, duration_ms=400)
                _speak("Please pull over, fatigue detected")
            self._last_state = state

        if distraction_flag and not self._last_distraction:
            _beep(frequency=750, duration_ms=150)
        self._last_distraction = distraction_flag

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
            cv2.rectangle(frame, (4, 4), (w - 5, h - 5), NOD_ALERT_COLOR, 8)
            cv2.putText(frame, "DROWSY - REPEATED NODDING", (20, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, NOD_ALERT_COLOR, 2)
        elif self._nod_alert_until is not None and now >= self._nod_alert_until:
            self._nod_alert_until = None

        return frame, alert_fired

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
    voice alerts only on state transitions (not every frame, so it
    doesn't spam a beep 30x/second).
    """

    def __init__(self):
        self._last_state = None
        self._last_distraction = False

    def update(self, frame, state, distraction_flag=False):
        h, w = frame.shape[:2]
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

        return frame, alert_fired

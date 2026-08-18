# Data Contract

This is the agreed interface between modules. Nobody changes key names or types here without flagging it to the other two first, since both downstream modules depend on this exact shape.

Status: DRAFT -- finalize together in Week 1 before any integration code is written.

## 1. Perception -> Decision (per-frame dict)

Produced by Hafsa's module, once per processed frame.

```python
{
    "frame_id": int,           # sequential frame counter
    "timestamp": float,         # seconds since session start
    "EAR": float,                 # 0.0-0.4 range, averaged both eyes
    "MAR": float,                  # mouth aspect ratio
    "blink_state": str,             # "open" | "closed"
    "ear_confidence": float,          # 0.0-1.0, lower if glasses/poor lighting
    "landmarks_detected": bool,        # False if no face found this frame
}
```

Open questions to settle as a group:
- Left/right eye EAR reported separately, or only the average? (default: average only, unless the evaluation ablation study needs per-eye values)
- What value is reported when no face is detected? (proposal: `None` for EAR/MAR, `landmarks_detected: False`, rather than 0.0, so the Decision layer can distinguish "eyes closed" from "no face found")

## 2. Decision -> App / Evaluation (per-frame or per-window output)

Produced by Sheethal's module.

```python
{
    "frame_id": int,
    "drowsiness_score": float,     # 0-100
    "system_state": str,             # "ALERT" | "WARNING" | "CRITICAL"
    "PERCLOS": float,                  # 0.0-1.0, rolling 60s window
    "yawn_count": int,                   # cumulative this session
    "nod_count": int,                     # cumulative this session
    "head_pitch": float,                    # degrees
    "head_yaw": float,                       # degrees
    "head_roll": float,                       # degrees
    "distraction_flag": bool,
    "alert_fired": bool,
}
```

Open questions to settle as a group:
- Is this emitted every frame, or only on state change / every N frames? (affects logging volume and what the evaluation layer reads)
- Units for pitch/yaw/roll -- degrees confirmed, but define the zero point (straight ahead at camera = 0,0,0) and sign convention (e.g. positive yaw = looking right) so the angle-sensitivity experiment interprets values correctly.

## 3. Logging format (for evaluation)

Every state transition and alert gets logged to CSV, one row per event:

```
timestamp, frame_id, event_type, drowsiness_score, system_state, PERCLOS, notes
```

`event_type` examples: `state_change`, `alert_fired`, `calibration_complete`.

This is what `evaluate.py` and the metrics dashboard read from -- confirm this schema works for the ROC/confusion matrix/latency calculations before Week 4.

## 4. Versioning

If any key name, type, or range changes after this is finalized, note it here with a date, so nobody's module silently breaks against an old assumption:

| Date | Change | Changed by |
|---|---|---|
| | | |
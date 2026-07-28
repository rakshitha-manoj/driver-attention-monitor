"""
Assembles Decision's per-frame output dict, matching
shared/data_contract.md exactly:

    frame_id, drowsiness_score, system_state, PERCLOS, yawn_count,
    nod_count, head_pitch, head_yaw, head_roll, distraction_flag,
    alert_fired

This is what Raks (Evaluation) and app.py consume downstream.
Field names/types are locked per the contract -- don't rename or
retype these without flagging the team first, same rule that
applies to Perception's dict.

Extra signals built this session (distraction_axis, face_lost_alert)
are NOT in the original contract. They're included as additional
keys here so nothing is lost, but Raks's code won't expect them --
flag these two as proposed contract additions, don't assume she's
already handling them.
"""


def build_output_dict(frame_id, drowsiness_score, system_state,
                       perclos, yawn_count, nod_count,
                       head_pitch, head_yaw, head_roll,
                       distraction_flag, alert_fired,
                       distraction_axis=None, face_lost_alert=False,
                       pose_plausible=True):
    return {
        "frame_id": frame_id,
        "drowsiness_score": round(drowsiness_score, 2),
        "system_state": system_state,
        "PERCLOS": round(perclos, 4),
        "yawn_count": yawn_count,
        "nod_count": nod_count,
        "head_pitch": round(head_pitch, 2),
        "head_yaw": round(head_yaw, 2),
        "head_roll": round(head_roll, 2),
        "distraction_flag": distraction_flag,
        "alert_fired": alert_fired,
        # -- proposed additions, not yet in shared/data_contract.md --
        "distraction_axis": distraction_axis,
        "face_lost_alert": face_lost_alert,
        "pose_plausible": pose_plausible,
    }


def is_plausible(pitch, yaw, roll, max_pitch=70.0, max_yaw=90.0, max_roll=60.0):
    """
    Single sanity gate for a calibrated pose reading. A cropped or
    partially-out-of-frame face can make solvePnP return a
    mathematically valid but physically impossible rotation --
    this catches it before it reaches nod/distraction detection,
    regardless of which angle the corruption shows up in.

    Bounds are generous (a real distraction turn can legitimately
    approach ~90 deg yaw) -- the goal is filtering garbage, not
    restricting normal head movement.
    """
    return (abs(pitch) <= max_pitch and
            abs(yaw) <= max_yaw and
            abs(roll) <= max_roll)

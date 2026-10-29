"""
Module 3 - Agent-style teaching suggestions (rule-based reasoning, offline).

The agent reads the engagement timeline, spots patterns (drops, peaks, fading
attention, low interaction, participants leaving) and turns them into plain
language tips. It is decision SUPPORT only - the instructor decides.
"""
import numpy as np


def _runs(mask):
    """Group consecutive True positions into (start, end) index pairs."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append((start, i - 1)); start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def generate_suggestions(windows, students, window_s=30):
    tips = []
    if windows is None or windows.empty:
        return [("info", "Not enough data to analyse engagement. Try a longer video with clearly visible faces.")]

    eng = windows["engagement"].to_numpy()
    avg = float(np.mean(eng))
    if avg >= 65:
        tips.append(("good", f"Overall engagement was strong ({avg:.0f}/100). The current pacing and format are working well."))
    elif avg >= 40:
        tips.append(("info", f"Overall engagement was moderate ({avg:.0f}/100). A few interactive moments could lift it."))
    else:
        tips.append(("warn", f"Overall engagement was low ({avg:.0f}/100). Consider shorter explanation blocks and more learner activity."))

    # Attention drops
    low = (eng < 40) | (eng < avg - 15)
    for s, e in _runs(low):
        a, b = windows["label"].iloc[s].split("-")[0], windows["label"].iloc[e].split("-")[1]
        tips.append(("warn", f"Attention dropped between {a} and {b}. Try a quick poll, a question to the class, "
                             f"or a short hands-on activity at that point."))

    # Peak
    top = int(np.argmax(eng))
    tips.append(("good", f"Engagement peaked at {windows['label'].iloc[top]} ({eng[top]:.0f}/100). "
                         f"Note what you were doing then and reuse that approach."))

    # Fading attention
    if len(eng) >= 3:
        third = max(1, len(eng) // 3)
        first, last = float(np.mean(eng[:third])), float(np.mean(eng[-third:]))
        if first - last > 10:
            tips.append(("warn", "Engagement faded towards the end of the session. Add a short break, "
                                 "a recap quiz, or split the later content into smaller chunks."))
        elif last - first > 10:
            tips.append(("good", "Engagement grew over the session - the warm-up may be worth shortening."))

    # Audio / interaction
    if windows["audio"].notna().any():
        a_mean = float(windows["audio"].dropna().mean())
        if a_mean < 15:
            tips.append(("info", "Very little audio activity was detected. Encourage questions or "
                                 "discussion so learners participate verbally."))
        quiet_and_low = windows[(windows["audio"].fillna(100) < 15) & (windows["engagement"] < avg)]
        if len(quiet_and_low):
            tips.append(("info", f"Quiet, low-engagement stretches ({', '.join(quiet_and_low['label'].head(3))}) "
                                 f"may be good places for pair discussion."))
    else:
        tips.append(("info", "No audio track was found, so engagement was computed from video only."))

    # Participants leaving
    peak = windows["people"].max()
    dips = windows[windows["people"] < 0.7 * peak]
    if peak >= 3 and len(dips):
        tips.append(("warn", f"The number of visible participants dipped in {', '.join(dips['label'].head(3))}. "
                             f"Check pacing, breaks, or whether learners stepped out."))

    # Individuals (anonymous)
    if students is not None and not students.empty:
        needs = students[(students["Avg engagement"] < 35) | (students["Longest inactivity (s)"] > 120)]
        if len(needs):
            names = ", ".join(needs["Person"].head(5))
            tips.append(("info", f"{names}: low engagement or long inactivity. A friendly check-in may help "
                                 f"(IDs are anonymous; no identities are stored)."))

    tips.append(("info", "These suggestions are assistive. Use your judgement - you know your class best."))
    return tips

from numba.core.decorators import njit
import numpy as np


@njit(cache=True, boundscheck=False)
def extract_notes_start_stop_numba(sorted_note_events: np.ndarray, notes_mode: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract the notes from the sorted note events.
    The note events are assumed to be sorted lexigographically by pitch, tick the original midi order.

    We provide control to the user on how to handle overlapping note and zero length notes
    through the parameter `notes_mode` that allows choosing among multiple modes:

    | notes_mode | strategy              |
    |------|-----------------------|
    | 0    | no overlap            |
    | 1    | first-in-first-out    |
    | 2    | Note Off stops all    |

    """
    assert notes_mode in {0, 1, 2}, "mode must be between 1 and 6"
    note_start_ids: list[int] = []
    note_stop_ids: list[int] = []
    active_note_starts: list[int] = []
    new_active_note_starts: list[int] = []
    last_pitch = -1
    last_channel = -1
    for k in range(len(sorted_note_events)):
        if not last_pitch == sorted_note_events[k]["value1"] or not last_channel == sorted_note_events[k]["channel"]:
            # remove unfinished notes for the previous pitch and channel
            active_note_starts.clear()
            last_pitch = sorted_note_events[k]["value1"]
            last_channel = sorted_note_events[k]["channel"]

        if sorted_note_events[k]["event_type"] == 0 and sorted_note_events[k]["value2"] > 0:
            # Note on event
            if notes_mode == 0:
                # stop the all active notes
                for note in active_note_starts:
                    note_duration = sorted_note_events[k]["tick"] - sorted_note_events[note]["tick"]
                    if note_duration > 0:
                        note_start_ids.append(note)
                        note_stop_ids.append(k)
                active_note_starts.clear()
            active_note_starts.append(k)
        # Note off event
        elif notes_mode in {0, 2}:
            # stop all the active notes whose duration is greater than 0
            new_active_note_starts.clear()
            for note in active_note_starts:
                note_duration = sorted_note_events[k]["tick"] - sorted_note_events[note]["tick"]
                if note_duration > 0:
                    note_start_ids.append(note)
                    note_stop_ids.append(k)
                else:
                    new_active_note_starts.append(note)
            active_note_starts.clear()
            for note in new_active_note_starts:
                active_note_starts.append(note)

        elif notes_mode == 1:
            # stop the first active
            if len(active_note_starts) > 0:
                note = active_note_starts.pop(0)
                note_duration = sorted_note_events[k]["tick"] - sorted_note_events[note]["tick"]
                if note_duration > 0:
                    note_start_ids.append(note)
                    note_stop_ids.append(k)
        else:
            raise ValueError(f"Unknown mode {notes_mode}")
    return np.array(note_start_ids), np.array(note_stop_ids)


@njit(cache=True, boundscheck=False)
def get_events_program(events: np.ndarray) -> np.ndarray:
    channel_to_program = np.zeros((16), dtype=np.int32)
    program = np.zeros((len(events)), dtype=np.int32)

    for i in range(len(events)):
        if events[i]["event_type"] == 4:
            channel_to_program[events[i]["channel"]] = events[i]["value1"]
        program[i] = channel_to_program[events[i]["channel"]]
    return program


@njit(cache=True, boundscheck=False)
def get_pedals_from_controls_jit(channel_controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # remove heading pedal off events appearing any pedal on event
    active_pedal = False
    pedal_start = 0
    pedals_starts = []
    pedals_ends = []

    for k in range(len(channel_controls)):
        if channel_controls["number"][k] != 64:
            continue
        if channel_controls[k]["value"] == 127 and not active_pedal:
            active_pedal = True
            pedal_start = k
        if channel_controls[k]["value"] == 0 and active_pedal:
            active_pedal = False
            pedals_starts.append(pedal_start)
            pedals_ends.append(k)

    return np.array(pedals_starts), np.array(pedals_ends)


@njit(cache=True, boundscheck=False)
def _get_overlapping_notes_pairs_jit(
    start: np.ndarray, duration: np.ndarray, pitch: np.ndarray, order: np.ndarray
) -> np.ndarray:
    """Get the pairs of overlapping notes in the score.

    the order array should be the order of the notes by pitch
    and then by start time within each pitch.
    i.e np.lexsort(notes["start"], notes["pitch"])
    but lexsort is does not seem supported by numba so keeping it as an argument.
    """
    n = len(start)
    if n == 0:
        return np.empty((0, 2), dtype=np.int64)

    # sort the notes by pitch and then by start time
    start = start[order]
    duration = duration[order]
    pitch = pitch[order]

    min_pitch = pitch.min()
    max_pitch = pitch.max()
    num_pitches = max_pitch - min_pitch + 1

    # for each pitch, get the start and end index in the sorted array
    pitch_start_indices = np.full(num_pitches, n, dtype=np.int64)
    pitch_end_indices = np.zeros(num_pitches, dtype=np.int64)
    for i in range(n):
        p = pitch[i] - min_pitch
        if pitch_start_indices[p] == n:
            pitch_start_indices[p] = i
        pitch_end_indices[p] = i + 1

    # Process each pitch independently
    overlapping_notes = []
    for k in range(num_pitches):
        # Check overlaps within this pitch
        for i in range(pitch_start_indices[k], pitch_end_indices[k]):
            for j in range(i + 1, pitch_end_indices[k]):
                # Check overlap condition
                if start[i] + duration[i] > start[j]:
                    assert pitch[i] == pitch[j], "Pitch mismatch"
                    overlapping_notes.append((order[i], order[j]))
                else:
                    # Break early since notes are sorted by start time
                    break

    if len(overlapping_notes) == 0:
        result = np.empty((0, 2), dtype=np.int64)
    else:
        result = np.array(overlapping_notes, dtype=np.int64)
    return result


@njit(cache=True, boundscheck=False)
def recompute_tempo_times(tempo: np.ndarray, ticks_per_quarter: int) -> None:
    """Get the time of each event in ticks and seconds."""
    tick = np.uint32(0)
    time = 0.0
    second_per_tick = 0.0

    ref_tick = 0
    ref_time = 0.0
    last_tempo_event = -1

    for i in range(len(tempo)):
        delta_tick = tempo[i]["tick"] - tick
        tick += delta_tick
        while last_tempo_event + 1 < len(tempo) and tick >= tempo[last_tempo_event + 1]["tick"]:
            # tempo change event
            last_tempo_event += 1
            tempo_event = tempo[last_tempo_event]
            ref_time = ref_time + (tempo_event["tick"] - ref_tick) * second_per_tick
            ref_tick = tempo_event["tick"]
            bpm = tempo_event["bpm"]
            second_per_tick = 60.0 / (bpm * ticks_per_quarter)  # seconds per tick

        time = ref_time + (tick - ref_tick) * second_per_tick
        tempo[i]["time"] = time

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
    channel_to_program = np.full((16), -1, dtype=np.int32)
    program = np.zeros((len(events)), dtype=np.int32)

    for i in range(len(events)):
        if events[i]["event_type"] == 4:
            channel_to_program[events[i]["channel"]] = events[i]["value1"]
        program[i] = channel_to_program[events[i]["channel"]]

    # walk backward to replace -1 by the nearest following valid program
    # in the channel
    # this is to deal with events before the first program change
    # FIXME maybe we should take into account the notes and attribute use the channel
    # associate to the next note?
    channel_to_program[channel_to_program == -1] = 0
    for i in range(len(events) - 1, -1, -1):
        if program[i] == -1:
            program[i] = channel_to_program[events[i]["channel"]]
        else:
            channel_to_program[events[i]["channel"]] = program[i]
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
    overlapping_notes: list[tuple[int, int]] = []
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
        num_ovrlapping_notes = len(overlapping_notes)
        result = np.empty((num_ovrlapping_notes, 2), dtype=np.int64)
        for i in range(num_ovrlapping_notes):
            result[i, 0] = overlapping_notes[i][0]
            result[i, 1] = overlapping_notes[i][1]

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
            quarter_notes_per_minute = tempo_event["quarter_notes_per_minute"]
            second_per_tick = 60.0 / (quarter_notes_per_minute * ticks_per_quarter)  # seconds per tick

        time = ref_time + (tick - ref_tick) * second_per_tick
        tempo[i]["time"] = time


@njit(cache=True, boundscheck=False)
def get_beats_per_bar(time_signature: np.ndarray) -> np.ndarray:
    """Get the number of beats per bar from the time signature."""
    compound_meter = is_compound_meter(time_signature)
    out = np.where(compound_meter, time_signature["numerator"] // 3, time_signature["numerator"])
    return out


@njit(cache=True, boundscheck=False)
def is_compound_meter(time_signature: np.ndarray) -> np.ndarray:
    """Check if the time signature is a compound meter.
    a signature is a compound meter if all applies:
    * the numerator is divisible by 3
    * the numerator is greater than 3
    * the denominator is 8 or 16.
    """
    return (
        (time_signature["numerator"] % 3 == 0)
        & (time_signature["numerator"] > 3)
        & ((time_signature["denominator"] == 8) | (time_signature["denominator"] == 16))
    )


@njit(cache=True, boundscheck=False)
def get_subdivision_per_beat(time_signature: np.ndarray) -> np.ndarray:
    """Get the subdivision per beat from the time signature."""
    compound_meter = is_compound_meter(time_signature)
    out = np.where(compound_meter, 3, 1)
    return out


@njit(cache=True, boundscheck=False)
def get_tick_per_subdivision(ticks_per_quarter: int, time_signature: np.ndarray) -> np.ndarray:
    """Get the tick per subdivision from the time signature."""
    return ticks_per_quarter * 4 / time_signature["denominator"]


@njit(cache=True, boundscheck=False)
def get_subdivision_beat_and_bar_ticks_jit(
    ticks_per_quarter: int, last_tick: int, time_signature: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the beat and bar ticks from the time signature."""
    beat = 0
    tick = 0
    bar = 0
    i_signature = 0

    subdivision_ticks = [0]
    beat_ticks = [0]
    bar_ticks = [0]

    tick_per_subdivision_array = get_tick_per_subdivision(ticks_per_quarter, time_signature)
    subdivision_per_beat_array = get_subdivision_per_beat(time_signature)
    beat_per_bar_array = get_beats_per_bar(time_signature)
    subdivision = 0
    while True:
        if tick >= last_tick:
            break

        tick += tick_per_subdivision_array[i_signature]
        subdivision += 1
        subdivision_ticks.append(tick)

        if subdivision >= subdivision_per_beat_array[i_signature]:
            beat += 1
            subdivision = 0
            # Add the tick to the beat ticks
            beat_ticks.append(tick)

        if beat >= beat_per_bar_array[i_signature]:
            bar += 1
            beat = 0
            bar_ticks.append(tick)

        if i_signature + 1 < len(time_signature["tick"]) and tick >= time_signature["tick"][i_signature + 1]:
            i_signature += 1

    subdivision_ticks_np = np.array(subdivision_ticks)
    bar_ticks_np = np.array(bar_ticks)
    beat_ticks_np = np.array(beat_ticks)
    return subdivision_ticks_np, beat_ticks_np, bar_ticks_np

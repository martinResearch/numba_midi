"""
Pure Python Cython implementation of score processing functions.
This uses standard Python with Cython decorators for optimization.
"""

import cython
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def extract_notes_start_stop_soa(tick, event_type, channel, value1, value2, notes_mode):
    """Extract the notes from sorted note events using SoA structure."""
    num_events = len(tick)
    
    # Pre-allocate arrays with maximum possible size
    note_start_ids = np.empty(num_events, dtype=np.int32)
    note_stop_ids = np.empty(num_events, dtype=np.int32)
    active_note_starts = np.empty(num_events, dtype=np.int32)
    
    note_pair_count = 0
    active_count = 0
    last_pitch = -1
    last_channel = -1
    
    for k in range(num_events):
        current_pitch = value1[k]
        current_channel = channel[k]
        
        if last_pitch != current_pitch or last_channel != current_channel:
            # Clear active notes for the previous pitch and channel
            active_count = 0
            last_pitch = current_pitch
            last_channel = current_channel
            
        et = event_type[k]
        v2 = value2[k]
        current_tick = tick[k]
        
        if et == 0 and v2 > 0:
            # Note on event
            if notes_mode == 0:
                # Stop all active notes
                for i in range(active_count):
                    note = active_note_starts[i]
                    note_duration = current_tick - tick[note]
                    if note_duration > 0:
                        note_start_ids[note_pair_count] = note
                        note_stop_ids[note_pair_count] = k
                        note_pair_count += 1
                active_count = 0
            
            # Add new active note
            active_note_starts[active_count] = k
            active_count += 1
            
        elif notes_mode in (0, 2):
            # Note off event - stop all active notes whose duration is > 0
            new_active_count = 0
            for i in range(active_count):
                note = active_note_starts[i]
                note_duration = current_tick - tick[note]
                if note_duration > 0:
                    note_start_ids[note_pair_count] = note
                    note_stop_ids[note_pair_count] = k
                    note_pair_count += 1
                else:
                    # Keep notes with zero duration active
                    active_note_starts[new_active_count] = note
                    new_active_count += 1
            active_count = new_active_count
            
        elif notes_mode == 1:
            # Stop the first active note (FIFO)
            if active_count > 0:
                note = active_note_starts[0]
                note_duration = current_tick - tick[note]
                if note_duration > 0:
                    note_start_ids[note_pair_count] = note
                    note_stop_ids[note_pair_count] = k
                    note_pair_count += 1
                
                # Shift remaining active notes down
                for i in range(1, active_count):
                    active_note_starts[i - 1] = active_note_starts[i]
                active_count -= 1
        else:
            raise ValueError(f"Unknown mode {notes_mode}")
    
    # Return trimmed arrays with actual size
    return note_start_ids[:note_pair_count].copy(), note_stop_ids[:note_pair_count].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
def get_events_program_soa(event_type, channel, value1):
    """Get program changes for events using SoA structure."""
    num_events = len(event_type)
    channel_to_program = np.full(16, -1, dtype=np.int32)
    program = np.zeros(num_events, dtype=np.int32)
    
    # Forward pass
    for i in range(num_events):
        et = event_type[i]
        ch = channel[i]
        
        if et == 4:  # Program change
            v1 = value1[i]
            channel_to_program[ch] = v1
        program[i] = channel_to_program[ch]
    
    # Replace -1 with 0 for channels without program changes
    for i in range(16):
        if channel_to_program[i] == -1:
            channel_to_program[i] = 0
    
    # Backward pass to handle events before first program change
    for i in range(num_events - 1, -1, -1):
        ch = channel[i]
        if program[i] == -1:
            program[i] = channel_to_program[ch]
        else:
            channel_to_program[ch] = program[i]
            
    return program


@cython.boundscheck(False)
@cython.wraparound(False)
def get_overlapping_notes_pairs_soa(start_tick, duration_tick, pitch, order):
    """Get the pairs of overlapping notes using SoA structure."""
    n = len(start_tick)
    if n == 0:
        return np.empty((0, 2), dtype=np.int64)
        
    # Sort the notes by pitch and then by start time using the order array
    start_sorted = np.empty(n, dtype=np.int32)
    duration_sorted = np.empty(n, dtype=np.int32)
    pitch_sorted = np.empty(n, dtype=np.int32)
    
    for i in range(n):
        idx = order[i]
        start_sorted[i] = start_tick[idx]
        duration_sorted[i] = duration_tick[idx]
        pitch_sorted[i] = pitch[idx]
    
    min_pitch = pitch_sorted.min()
    max_pitch = pitch_sorted.max()
    num_pitches = max_pitch - min_pitch + 1
    
    # For each pitch, get the start and end index in the sorted array
    pitch_start_indices = np.full(num_pitches, n, dtype=np.int32)
    pitch_end_indices = np.zeros(num_pitches, dtype=np.int32)
    
    for i in range(n):
        p = pitch_sorted[i] - min_pitch
        if pitch_start_indices[p] == n:
            pitch_start_indices[p] = i
        pitch_end_indices[p] = i + 1
    
    # Pre-allocate array for overlapping pairs (worst case: n*(n-1)/2 pairs)
    max_pairs = n * (n - 1) // 2
    overlapping_pairs = np.empty((max_pairs, 2), dtype=np.int32)
    pair_count = 0
    
    # Process each pitch independently
    for k in range(num_pitches):
        # Check overlaps within this pitch
        for i in range(pitch_start_indices[k], pitch_end_indices[k]):
            for j in range(i + 1, pitch_end_indices[k]):
                # Check overlap condition
                if start_sorted[i] + duration_sorted[i] > start_sorted[j]:
                    overlapping_pairs[pair_count, 0] = order[i]
                    overlapping_pairs[pair_count, 1] = order[j]
                    pair_count += 1
                else:
                    # Break early since notes are sorted by start time
                    break
    
    if pair_count == 0:
        return np.empty((0, 2), dtype=np.int64)
    else:
        # Convert to int64 and return trimmed array
        result = np.empty((pair_count, 2), dtype=np.int64)
        for i in range(pair_count):
            result[i, 0] = overlapping_pairs[i, 0]
            result[i, 1] = overlapping_pairs[i, 1]
        return result


@cython.boundscheck(False)
@cython.wraparound(False)
def recompute_tempo_times_soa(time, tick, quarter_notes_per_minute, ticks_per_quarter):
    """Recompute tempo times using SoA structure."""
    num_tempo = len(time)
    current_tick = 0
    current_time = 0.0
    second_per_tick = 0.0
    ref_tick = 0
    ref_time = 0.0
    last_tempo_event = -1
    
    for i in range(num_tempo):
        current_tick = tick[i]
        delta_tick = current_tick - current_tick
        current_tick += delta_tick
        
        while (last_tempo_event + 1 < num_tempo and 
               current_tick >= tick[last_tempo_event + 1]):
            # Tempo change event
            last_tempo_event += 1
            tempo_event_tick = tick[last_tempo_event]
            ref_time = ref_time + (tempo_event_tick - ref_tick) * second_per_tick
            ref_tick = tempo_event_tick
            qnpm = quarter_notes_per_minute[last_tempo_event]
            second_per_tick = 60.0 / (qnpm * ticks_per_quarter)
            
        current_time = ref_time + (current_tick - ref_tick) * second_per_tick
        time[i] = current_time


@cython.boundscheck(False)
@cython.wraparound(False)
def get_pedals_from_controls_soa(number, value):
    """Extract pedals from controls using SoA structure."""
    num_controls = len(number)
    
    # Pre-allocate arrays with maximum possible size
    pedals_starts = np.empty(num_controls, dtype=np.int32)
    pedals_ends = np.empty(num_controls, dtype=np.int32)
    
    active_pedal = False
    pedal_start = 0
    pedal_count = 0
    
    for k in range(num_controls):
        num = number[k]
        if num != 64:  # Sustain pedal only
            continue
            
        val = value[k]
        if val == 127 and not active_pedal:
            # Pedal on
            active_pedal = True
            pedal_start = k
        elif val == 0 and active_pedal:
            # Pedal off
            active_pedal = False
            pedals_starts[pedal_count] = pedal_start
            pedals_ends[pedal_count] = k
            pedal_count += 1
            
    # Return trimmed arrays with actual size
    return pedals_starts[:pedal_count].copy(), pedals_ends[:pedal_count].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
def get_subdivision_beat_and_bar_ticks_soa(ticks_per_quarter, last_tick, time, tick, numerator, denominator):
    """Get the beat and bar ticks using SoA structure."""
    beat = 0
    current_tick = 0
    bar = 0
    i_signature = 0
    subdivision = 0
    
    # Estimate maximum size (conservative upper bound)
    max_ticks = last_tick // (ticks_per_quarter // 16) + 1000
    
    subdivision_ticks = np.empty(max_ticks, dtype=np.int32)
    beat_ticks = np.empty(max_ticks, dtype=np.int32)
    bar_ticks = np.empty(max_ticks, dtype=np.int32)
    
    subdiv_count = 1
    beat_count = 1
    bar_count = 1
    
    # Initialize with tick 0
    subdivision_ticks[0] = 0
    beat_ticks[0] = 0
    bar_ticks[0] = 0
    
    num_signatures = len(time)
    
    while True:
        if current_tick >= last_tick:
            break
        
        # Calculate tick per subdivision    
        tick_per_subdivision = ticks_per_quarter * 4 // denominator[i_signature]
        
        # Calculate subdivision per beat (compound vs simple meter)
        is_compound = ((numerator[i_signature] % 3 == 0) and 
                      (numerator[i_signature] > 3) and 
                      ((denominator[i_signature] == 8) or (denominator[i_signature] == 16)))
        subdivision_per_beat = 3 if is_compound else 1
        
        # Calculate beats per bar
        beat_per_bar = numerator[i_signature] // 3 if is_compound else numerator[i_signature]
            
        current_tick += tick_per_subdivision
        subdivision += 1
        
        if subdiv_count < max_ticks:
            subdivision_ticks[subdiv_count] = current_tick
            subdiv_count += 1
        
        if subdivision >= subdivision_per_beat:
            beat += 1
            subdivision = 0
            if beat_count < max_ticks:
                beat_ticks[beat_count] = current_tick
                beat_count += 1
            
        if beat >= beat_per_bar:
            bar += 1
            beat = 0
            if bar_count < max_ticks:
                bar_ticks[bar_count] = current_tick
                bar_count += 1
            
        if (i_signature + 1 < num_signatures and 
            current_tick >= tick[i_signature + 1]):
            i_signature += 1
            
    return (
        subdivision_ticks[:subdiv_count].copy(),
        beat_ticks[:beat_count].copy(),
        bar_ticks[:bar_count].copy()
    )
    
    k: cython.int
    note: cython.int
    note_duration: cython.int
    i: cython.int
    new_active_count: cython.int
    current_pitch: cython.int
    current_channel: cython.int
    et: cython.uchar
    v2: cython.short
    current_tick: cython.int
    
    for k in range(num_events):
        current_pitch = value1_view[k]
        current_channel = channel_view[k]
        
        if last_pitch != current_pitch or last_channel != current_channel:
            # Clear active notes for the previous pitch and channel
            active_count = 0
            last_pitch = current_pitch
            last_channel = current_channel
            
        et = event_type_view[k]
        v2 = value2_view[k]
        current_tick = tick_view[k]
        
        if et == 0 and v2 > 0:
            # Note on event
            if notes_mode == 0:
                # Stop all active notes
                for i in range(active_count):
                    note = active_starts_view[i]
                    note_duration = current_tick - tick_view[note]
                    if note_duration > 0:
                        note_start_view[note_pair_count] = note
                        note_stop_view[note_pair_count] = k
                        note_pair_count += 1
                active_count = 0
            
            # Add new active note
            active_starts_view[active_count] = k
            active_count += 1
            
        elif notes_mode in (0, 2):
            # Note off event - stop all active notes whose duration is > 0
            new_active_count = 0
            for i in range(active_count):
                note = active_starts_view[i]
                note_duration = current_tick - tick_view[note]
                if note_duration > 0:
                    note_start_view[note_pair_count] = note
                    note_stop_view[note_pair_count] = k
                    note_pair_count += 1
                else:
                    # Keep notes with zero duration active
                    active_starts_view[new_active_count] = note
                    new_active_count += 1
            active_count = new_active_count
            
        elif notes_mode == 1:
            # Stop the first active note (FIFO)
            if active_count > 0:
                note = active_starts_view[0]
                note_duration = current_tick - tick_view[note]
                if note_duration > 0:
                    note_start_view[note_pair_count] = note
                    note_stop_view[note_pair_count] = k
                    note_pair_count += 1
                
                # Shift remaining active notes down
                for i in range(1, active_count):
                    active_starts_view[i - 1] = active_starts_view[i]
                active_count -= 1
        else:
            raise ValueError(f"Unknown mode {notes_mode}")
    
    # Return trimmed arrays with actual size
    return note_start_ids[:note_pair_count].copy(), note_stop_ids[:note_pair_count].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
def get_events_program_soa(
    event_type: ndarray,
    channel: ndarray,
    value1: ndarray
) -> ndarray:
    """Get program changes for events using SoA structure."""
    num_events: cython.int = len(event_type)
    channel_to_program: ndarray = np.full(16, -1, dtype=np.int32)
    program: ndarray = np.zeros(num_events, dtype=np.int32)
    
    channel_to_program_view = cython.cast(cython.int[:], channel_to_program)
    program_view = cython.cast(cython.int[:], program)
    event_type_view = cython.cast(cython.uchar[:], event_type)
    channel_view = cython.cast(cython.uchar[:], channel)
    value1_view = cython.cast(cython.int[:], value1)
    
    i: cython.int
    ch: cython.int
    et: cython.uchar
    v1: cython.int
    
    # Forward pass
    for i in range(num_events):
        et = event_type_view[i]
        ch = channel_view[i]
        
        if et == 4:  # Program change
            v1 = value1_view[i]
            channel_to_program_view[ch] = v1
        program_view[i] = channel_to_program_view[ch]
    
    # Replace -1 with 0 for channels without program changes
    for i in range(16):
        if channel_to_program_view[i] == -1:
            channel_to_program_view[i] = 0
    
    # Backward pass to handle events before first program change
    for i in range(num_events - 1, -1, -1):
        ch = channel_view[i]
        if program_view[i] == -1:
            program_view[i] = channel_to_program_view[ch]
        else:
            channel_to_program_view[ch] = program_view[i]
            
    return program


@cython.boundscheck(False)
@cython.wraparound(False)
def get_overlapping_notes_pairs_soa(
    start_tick: ndarray,
    duration_tick: ndarray,
    pitch: ndarray,
    order: ndarray
) -> ndarray:
    """Get the pairs of overlapping notes using SoA structure."""
    n: cython.int = len(start_tick)
    if n == 0:
        return np.empty((0, 2), dtype=np.int64)
        
    # Sort the notes by pitch and then by start time using the order array
    start_view = cython.cast(cython.int[:], start_tick)
    duration_view = cython.cast(cython.int[:], duration_tick)
    pitch_view = cython.cast(cython.int[:], pitch)
    order_view = cython.cast(cython.int[:], order)
    
    start_sorted: ndarray = np.empty(n, dtype=np.int32)
    duration_sorted: ndarray = np.empty(n, dtype=np.int32)
    pitch_sorted: ndarray = np.empty(n, dtype=np.int32)
    
    start_sorted_view = cython.cast(cython.int[:], start_sorted)
    duration_sorted_view = cython.cast(cython.int[:], duration_sorted)
    pitch_sorted_view = cython.cast(cython.int[:], pitch_sorted)
    
    i: cython.int
    idx: cython.int
    for i in range(n):
        idx = order_view[i]
        start_sorted_view[i] = start_view[idx]
        duration_sorted_view[i] = duration_view[idx]
        pitch_sorted_view[i] = pitch_view[idx]
    
    min_pitch: cython.int = pitch_sorted.min()
    max_pitch: cython.int = pitch_sorted.max()
    num_pitches: cython.int = max_pitch - min_pitch + 1
    
    # For each pitch, get the start and end index in the sorted array
    pitch_start_indices: ndarray = np.full(num_pitches, n, dtype=np.int32)
    pitch_end_indices: ndarray = np.zeros(num_pitches, dtype=np.int32)
    
    pitch_start_view = cython.cast(cython.int[:], pitch_start_indices)
    pitch_end_view = cython.cast(cython.int[:], pitch_end_indices)
    
    p: cython.int
    for i in range(n):
        p = pitch_sorted_view[i] - min_pitch
        if pitch_start_view[p] == n:
            pitch_start_view[p] = i
        pitch_end_view[p] = i + 1
    
    # Pre-allocate array for overlapping pairs (worst case: n*(n-1)/2 pairs)
    max_pairs: cython.int = n * (n - 1) // 2
    overlapping_pairs: ndarray = np.empty((max_pairs, 2), dtype=np.int32)
    pairs_view = cython.cast(cython.int[:, :], overlapping_pairs)
    pair_count: cython.int = 0
    
    # Process each pitch independently
    k: cython.int
    j: cython.int
    result: ndarray
    result_view: cython.long[:, :]
    
    for k in range(num_pitches):
        # Check overlaps within this pitch
        for i in range(pitch_start_view[k], pitch_end_view[k]):
            for j in range(i + 1, pitch_end_view[k]):
                # Check overlap condition
                if start_sorted_view[i] + duration_sorted_view[i] > start_sorted_view[j]:
                    pairs_view[pair_count, 0] = order_view[i]
                    pairs_view[pair_count, 1] = order_view[j]
                    pair_count += 1
                else:
                    # Break early since notes are sorted by start time
                    break
    
    if pair_count == 0:
        return np.empty((0, 2), dtype=np.int64)
    else:
        # Convert to int64 and return trimmed array
        result = np.empty((pair_count, 2), dtype=np.int64)
        result_view = cython.cast(cython.long[:, :], result)
        for i in range(pair_count):
            result_view[i, 0] = pairs_view[i, 0]
            result_view[i, 1] = pairs_view[i, 1]
        return result


@cython.boundscheck(False)
@cython.wraparound(False)
def recompute_tempo_times_soa(
    time: ndarray,
    tick: ndarray,
    quarter_notes_per_minute: ndarray,
    ticks_per_quarter: int
) -> None:
    """Recompute tempo times using SoA structure."""
    num_tempo: cython.int = len(time)
    current_tick: c_uint32_t = 0
    current_time: cython.double = 0.0
    second_per_tick: cython.double = 0.0
    ref_tick: cython.int = 0
    ref_time: cython.double = 0.0
    last_tempo_event: cython.int = -1
    
    time_view = cython.cast(cython.double[:], time)
    tick_view = cython.cast(cython.int[:], tick)
    qnpm_view = cython.cast(cython.double[:], quarter_notes_per_minute)
    
    i: cython.int
    delta_tick: c_uint32_t
    qnpm: cython.double
    tempo_event_tick: cython.int
    
    for i in range(num_tempo):
        current_tick = tick_view[i]
        delta_tick = current_tick - current_tick
        current_tick += delta_tick
        
        while (last_tempo_event + 1 < num_tempo and 
               current_tick >= tick_view[last_tempo_event + 1]):
            # Tempo change event
            last_tempo_event += 1
            tempo_event_tick = tick_view[last_tempo_event]
            ref_time = ref_time + (tempo_event_tick - ref_tick) * second_per_tick
            ref_tick = tempo_event_tick
            qnpm = qnpm_view[last_tempo_event]
            second_per_tick = 60.0 / (qnpm * ticks_per_quarter)
            
        current_time = ref_time + (current_tick - ref_tick) * second_per_tick
        time_view[i] = current_time


@cython.boundscheck(False)
@cython.wraparound(False)
def get_pedals_from_controls_soa(
    number: ndarray,
    value: ndarray
) -> Tuple[ndarray, ndarray]:
    """Extract pedals from controls using SoA structure."""
    num_controls: cython.int = len(number)
    
    # Pre-allocate arrays with maximum possible size
    pedals_starts: ndarray = np.empty(num_controls, dtype=np.int32)
    pedals_ends: ndarray = np.empty(num_controls, dtype=np.int32)
    
    starts_view = cython.cast(cython.int[:], pedals_starts)
    ends_view = cython.cast(cython.int[:], pedals_ends)
    
    active_pedal: cython.bint = False
    pedal_start: cython.int = 0
    pedal_count: cython.int = 0
    
    number_view = cython.cast(cython.int[:], number)
    value_view = cython.cast(cython.int[:], value)
    
    k: cython.int
    num: cython.int
    val: cython.int
    
    for k in range(num_controls):
        num = number_view[k]
        if num != 64:  # Sustain pedal only
            continue
            
        val = value_view[k]
        if val == 127 and not active_pedal:
            # Pedal on
            active_pedal = True
            pedal_start = k
        elif val == 0 and active_pedal:
            # Pedal off
            active_pedal = False
            starts_view[pedal_count] = pedal_start
            ends_view[pedal_count] = k
            pedal_count += 1
            
    # Return trimmed arrays with actual size
    return pedals_starts[:pedal_count].copy(), pedals_ends[:pedal_count].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
def get_subdivision_beat_and_bar_ticks_soa(
    ticks_per_quarter: int,
    last_tick: int,
    time: ndarray,
    tick: ndarray,
    numerator: ndarray,
    denominator: ndarray
) -> Tuple[ndarray, ndarray, ndarray]:
    """Get the beat and bar ticks using SoA structure."""
    beat: cython.int = 0
    current_tick: cython.int = 0
    bar: cython.int = 0
    i_signature: cython.int = 0
    subdivision: cython.int = 0
    
    # Estimate maximum size (conservative upper bound)
    max_ticks: cython.int = last_tick // (ticks_per_quarter // 16) + 1000
    
    subdivision_ticks: ndarray = np.empty(max_ticks, dtype=np.int32)
    beat_ticks: ndarray = np.empty(max_ticks, dtype=np.int32)
    bar_ticks: ndarray = np.empty(max_ticks, dtype=np.int32)
    
    subdiv_view = cython.cast(cython.int[:], subdivision_ticks)
    beat_view = cython.cast(cython.int[:], beat_ticks)
    bar_view = cython.cast(cython.int[:], bar_ticks)
    
    subdiv_count: cython.int = 1
    beat_count: cython.int = 1
    bar_count: cython.int = 1
    
    # Initialize with tick 0
    subdiv_view[0] = 0
    beat_view[0] = 0
    bar_view[0] = 0
    
    numerator_view = cython.cast(cython.int[:], numerator)
    denominator_view = cython.cast(cython.int[:], denominator)
    tick_view = cython.cast(cython.int[:], tick)
    
    num_signatures: cython.int = len(time)
    tick_per_subdivision: cython.int
    subdivision_per_beat: cython.int
    beat_per_bar: cython.int
    is_compound: cython.bint
    
    while True:
        if current_tick >= last_tick:
            break
        
        # Calculate tick per subdivision    
        tick_per_subdivision = ticks_per_quarter * 4 // denominator_view[i_signature]
        
        # Calculate subdivision per beat (compound vs simple meter)
        is_compound = ((numerator_view[i_signature] % 3 == 0) and 
                      (numerator_view[i_signature] > 3) and 
                      ((denominator_view[i_signature] == 8) or (denominator_view[i_signature] == 16)))
        subdivision_per_beat = 3 if is_compound else 1
        
        # Calculate beats per bar
        beat_per_bar = numerator_view[i_signature] // 3 if is_compound else numerator_view[i_signature]
            
        current_tick += tick_per_subdivision
        subdivision += 1
        
        if subdiv_count < max_ticks:
            subdiv_view[subdiv_count] = current_tick
            subdiv_count += 1
        
        if subdivision >= subdivision_per_beat:
            beat += 1
            subdivision = 0
            if beat_count < max_ticks:
                beat_view[beat_count] = current_tick
                beat_count += 1
            
        if beat >= beat_per_bar:
            bar += 1
            beat = 0
            if bar_count < max_ticks:
                bar_view[bar_count] = current_tick
                bar_count += 1
            
        if (i_signature + 1 < num_signatures and 
            current_tick >= tick_view[i_signature + 1]):
            i_signature += 1
            
    return (
        subdivision_ticks[:subdiv_count].copy(),
        beat_ticks[:beat_count].copy(),
        bar_ticks[:bar_count].copy()
    )

#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint8_t, int32_t, int64_t, uint32_t
cimport cython

ctypedef cnp.uint8_t DTYPE_uint8_t
ctypedef cnp.int32_t DTYPE_int32_t
ctypedef cnp.int64_t DTYPE_int64_t
ctypedef cnp.uint32_t DTYPE_uint32_t
ctypedef cnp.float64_t DTYPE_float64_t
ctypedef cnp.float32_t DTYPE_float32_t

def extract_notes_start_stop(
    cnp.ndarray sorted_note_events,
    int notes_mode
):
    """Extract the notes from the sorted note events."""
    cdef int num_events = len(sorted_note_events)
    cdef list note_start_ids = []
    cdef list note_stop_ids = []
    cdef list active_note_starts = []
    cdef list new_active_note_starts = []
    cdef int last_pitch = -1
    cdef int last_channel = -1
    cdef int k, note, note_duration
    cdef int current_pitch, current_channel, event_type, value2, tick
    
    for k in range(num_events):
        current_pitch = sorted_note_events[k]['value1']
        current_channel = sorted_note_events[k]['channel']
        
        if last_pitch != current_pitch or last_channel != current_channel:
            # Remove unfinished notes for the previous pitch and channel
            active_note_starts.clear()
            last_pitch = current_pitch
            last_channel = current_channel
            
        event_type = sorted_note_events[k]['event_type']
        value2 = sorted_note_events[k]['value2']
        tick = sorted_note_events[k]['tick']
        
        if event_type == 0 and value2 > 0:
            # Note on event
            if notes_mode == 0:
                # Stop all active notes
                for note in active_note_starts:
                    note_duration = tick - sorted_note_events[note]['tick']
                    if note_duration > 0:
                        note_start_ids.append(note)
                        note_stop_ids.append(k)
                active_note_starts.clear()
            active_note_starts.append(k)
        elif notes_mode in (0, 2):
            # Note off event - stop all active notes whose duration is > 0
            new_active_note_starts.clear()
            for note in active_note_starts:
                note_duration = tick - sorted_note_events[note]['tick']
                if note_duration > 0:
                    note_start_ids.append(note)
                    note_stop_ids.append(k)
                else:
                    new_active_note_starts.append(note)
            active_note_starts.clear()
            active_note_starts.extend(new_active_note_starts)
        elif notes_mode == 1:
            # Stop the first active note
            if len(active_note_starts) > 0:
                note = active_note_starts.pop(0)
                note_duration = tick - sorted_note_events[note]['tick']
                if note_duration > 0:
                    note_start_ids.append(note)
                    note_stop_ids.append(k)
        else:
            raise ValueError(f"Unknown mode {notes_mode}")
            
    return np.array(note_start_ids), np.array(note_stop_ids)

def get_events_program(cnp.ndarray events):
    """Get program changes for events."""
    cdef int num_events = len(events)
    cdef cnp.int32_t[:] channel_to_program = np.full(16, -1, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] program = np.zeros(num_events, dtype=np.int32)
    cdef cnp.int32_t[:] program_view = program
    cdef int i, channel, event_type, value1
    
    # Forward pass
    for i in range(num_events):
        event_type = events[i]['event_type']
        channel = events[i]['channel']
        
        if event_type == 4:  # Program change
            value1 = events[i]['value1']
            channel_to_program[channel] = value1
        program_view[i] = channel_to_program[channel]
    
    # Replace -1 with 0 for channels without program changes
    for i in range(16):
        if channel_to_program[i] == -1:
            channel_to_program[i] = 0
    
    # Backward pass to handle events before first program change
    for i in range(num_events - 1, -1, -1):
        channel = events[i]['channel']
        if program_view[i] == -1:
            program_view[i] = channel_to_program[channel]
        else:
            channel_to_program[channel] = program_view[i]
            
    return program

def get_pedals_from_controls(cnp.ndarray channel_controls):
    """Remove heading pedal off events appearing before any pedal on event."""
    cdef int num_controls = len(channel_controls)
    cdef bint active_pedal = False
    cdef int pedal_start = 0
    cdef list pedals_starts = []
    cdef list pedals_ends = []
    cdef int k, number, value
    
    for k in range(num_controls):
        number = channel_controls[k]['number']
        if number != 64:  # Sustain pedal
            continue
            
        value = channel_controls[k]['value']
        if value == 127 and not active_pedal:
            active_pedal = True
            pedal_start = k
        elif value == 0 and active_pedal:
            active_pedal = False
            pedals_starts.append(pedal_start)
            pedals_ends.append(k)
            
    return np.array(pedals_starts), np.array(pedals_ends)

def get_overlapping_notes_pairs(
    cnp.ndarray[cnp.int32_t, ndim=1] start,
    cnp.ndarray[cnp.int32_t, ndim=1] duration,
    cnp.ndarray[cnp.int32_t, ndim=1] pitch,
    cnp.ndarray[cnp.int32_t, ndim=1] order
):
    """Get the pairs of overlapping notes in the score."""
    cdef int n = len(start)
    if n == 0:
        return np.empty((0, 2), dtype=np.int64)
        
    # Sort the notes by pitch and then by start time using the order array
    cdef cnp.ndarray[cnp.int32_t, ndim=1] start_sorted = start[order]
    cdef cnp.ndarray[cnp.int32_t, ndim=1] duration_sorted = duration[order]
    cdef cnp.ndarray[cnp.int32_t, ndim=1] pitch_sorted = pitch[order]
    
    cdef int min_pitch = pitch_sorted.min()
    cdef int max_pitch = pitch_sorted.max()
    cdef int num_pitches = max_pitch - min_pitch + 1
    
    # For each pitch, get the start and end index in the sorted array
    cdef cnp.ndarray[cnp.int32_t, ndim=1] pitch_start_indices = np.full(num_pitches, n, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] pitch_end_indices = np.zeros(num_pitches, dtype=np.int32)
    cdef cnp.int32_t[:] pitch_start_view = pitch_start_indices
    cdef cnp.int32_t[:] pitch_end_view = pitch_end_indices
    cdef cnp.int32_t[:] start_view = start_sorted
    cdef cnp.int32_t[:] duration_view = duration_sorted
    cdef cnp.int32_t[:] pitch_view = pitch_sorted
    cdef cnp.int32_t[:] order_view = order
    
    cdef int i, p
    for i in range(n):
        p = pitch_view[i] - min_pitch
        if pitch_start_view[p] == n:
            pitch_start_view[p] = i
        pitch_end_view[p] = i + 1
    
    # Process each pitch independently
    cdef list overlapping_notes = []
    cdef int k, j
    
    for k in range(num_pitches):
        # Check overlaps within this pitch
        for i in range(pitch_start_view[k], pitch_end_view[k]):
            for j in range(i + 1, pitch_end_view[k]):
                # Check overlap condition
                if start_view[i] + duration_view[i] > start_view[j]:
                    overlapping_notes.append((order_view[i], order_view[j]))
                else:
                    # Break early since notes are sorted by start time
                    break
    
    # Declare variables outside conditional blocks
    cdef int num_overlapping
    cdef cnp.ndarray[cnp.int32_t, ndim=2] result
    
    if len(overlapping_notes) == 0:
        return np.empty((0, 2), dtype=np.int64)
    else:
        num_overlapping = len(overlapping_notes)
        result = np.empty((num_overlapping, 2), dtype=np.int64)
        for i in range(num_overlapping):
            result[i, 0] = overlapping_notes[i][0]
            result[i, 1] = overlapping_notes[i][1]
        return result

def recompute_tempo_times(cnp.ndarray tempo, int ticks_per_quarter):
    """Get the time of each event in ticks and seconds."""
    cdef int num_tempo = len(tempo)
    cdef uint32_t tick = 0
    cdef double time = 0.0
    cdef double second_per_tick = 0.0
    cdef int ref_tick = 0
    cdef double ref_time = 0.0
    cdef int last_tempo_event = -1
    cdef int i
    cdef uint32_t delta_tick, current_tick
    cdef double quarter_notes_per_minute
    
    for i in range(num_tempo):
        current_tick = tempo[i]['tick']
        delta_tick = current_tick - tick
        tick += delta_tick
        
        while (last_tempo_event + 1 < num_tempo and 
               tick >= tempo[last_tempo_event + 1]['tick']):
            # Tempo change event
            last_tempo_event += 1
            tempo_event_tick = tempo[last_tempo_event]['tick']
            ref_time = ref_time + (tempo_event_tick - ref_tick) * second_per_tick
            ref_tick = tempo_event_tick
            quarter_notes_per_minute = tempo[last_tempo_event]['quarter_notes_per_minute']
            second_per_tick = 60.0 / (quarter_notes_per_minute * ticks_per_quarter)
            
        time = ref_time + (tick - ref_tick) * second_per_tick
        tempo[i]['time'] = time

def get_beats_per_bar(cnp.ndarray time_signature):
    """Get the number of beats per bar from the time signature."""
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] compound_meter = is_compound_meter(time_signature)
    cdef cnp.ndarray result = np.where(
        compound_meter, 
        time_signature['numerator'] // 3, 
        time_signature['numerator']
    )
    return result

def is_compound_meter(cnp.ndarray time_signature):
    """Check if the time signature is a compound meter."""
    cdef cnp.ndarray numerator = time_signature['numerator']
    cdef cnp.ndarray denominator = time_signature['denominator']
    
    return (
        (numerator % 3 == 0) &
        (numerator > 3) &
        ((denominator == 8) | (denominator == 16))
    )

def get_subdivision_per_beat(cnp.ndarray time_signature):
    """Get the subdivision per beat from the time signature."""
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] compound_meter = is_compound_meter(time_signature)
    return np.where(compound_meter, 3, 1)

def get_tick_per_subdivision(int ticks_per_quarter, cnp.ndarray time_signature):
    """Get the tick per subdivision from the time signature."""
    return ticks_per_quarter * 4 / time_signature['denominator']

def get_subdivision_beat_and_bar_ticks(
    int ticks_per_quarter,
    int last_tick,
    cnp.ndarray time_signature
):
    """Get the beat and bar ticks from the time signature."""
    cdef int beat = 0
    cdef int tick = 0
    cdef int bar = 0
    cdef int i_signature = 0
    cdef int subdivision = 0
    
    cdef list subdivision_ticks = [0]
    cdef list beat_ticks = [0]
    cdef list bar_ticks = [0]
    
    cdef cnp.ndarray tick_per_subdivision_array = get_tick_per_subdivision(ticks_per_quarter, time_signature)
    cdef cnp.ndarray subdivision_per_beat_array = get_subdivision_per_beat(time_signature)
    cdef cnp.ndarray beat_per_bar_array = get_beats_per_bar(time_signature)
    
    cdef int num_signatures = len(time_signature)
    
    while True:
        if tick >= last_tick:
            break
            
        tick += int(tick_per_subdivision_array[i_signature])
        subdivision += 1
        subdivision_ticks.append(tick)
        
        if subdivision >= subdivision_per_beat_array[i_signature]:
            beat += 1
            subdivision = 0
            beat_ticks.append(tick)
            
        if beat >= beat_per_bar_array[i_signature]:
            bar += 1
            beat = 0
            bar_ticks.append(tick)
            
        if (i_signature + 1 < num_signatures and 
            tick >= time_signature['tick'][i_signature + 1]):
            i_signature += 1
            
    return (
        np.array(subdivision_ticks),
        np.array(beat_ticks),
        np.array(bar_ticks)
    )

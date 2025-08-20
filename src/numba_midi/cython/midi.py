"""
Pure Python Cython implementation of MIDI parsing and encoding functions.
This uses Cython's pure Python mode with type annotations.
"""


import cython
import numpy as np

# Cython imports for types
if cython.compiled:
    from cython.cimports.libc.stdint import int32_t as c_int32_t
    from cython.cimports.libc.stdint import uint8_t as c_uint8_t
    from cython.cimports.numpy import float32_t, int16_t, int32_t, ndarray, uint8_t, uint16_t, uint32_t
else:
    # Fallback types for non-compiled mode
    ndarray = np.ndarray
    int32_t = np.int32
    uint8_t = np.uint8
    uint16_t = np.uint16
    uint32_t = np.uint32
    int16_t = np.int16
    float32_t = np.float32
    c_int32_t = int
    c_uint8_t = int

# Define the event dtype structure that matches the numpy definition
event_dtype = np.dtype(
    [
        ("tick", np.int32),
        ("event_type", np.uint8),
        ("channel", np.uint8),
        ("value1", np.int32),
        ("value2", np.int16),
        ("value3", np.uint8),
        ("value4", np.uint8),
    ]
)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_event_times(midi_events: ndarray, tempo_events: ndarray, ticks_per_quarter: int) -> ndarray:
    """Get the time of each event in ticks and seconds."""
    i: cython.Py_ssize_t
    tick: c_int32_t = 0
    time: cython.double = 0.0
    quarter_notes_per_minute_init: cython.double = 120.0
    second_per_tick: cython.double = quarter_notes_per_minute_init / ticks_per_quarter / 60.0
    num_events: cython.Py_ssize_t = midi_events.shape[0]
    num_tempo_events: cython.Py_ssize_t = tempo_events.shape[0]

    events_times: ndarray = np.zeros(num_events, dtype=np.float32)
    events_times_mv = cython.cast(cython.float[:], events_times)

    ref_tick: cython.int = 0
    ref_time: cython.double = 0.0
    last_tempo_event: cython.int = -1
    delta_tick: c_int32_t
    microseconds_per_quarter_note: cython.double

    for i in range(num_events):
        delta_tick = midi_events[i]["tick"] - tick
        tick += delta_tick

        while last_tempo_event + 1 < num_tempo_events and tick >= tempo_events[last_tempo_event + 1]["tick"]:
            last_tempo_event += 1
            ref_time = ref_time + (tempo_events[last_tempo_event]["tick"] - ref_tick) * second_per_tick
            ref_tick = tempo_events[last_tempo_event]["tick"]
            microseconds_per_quarter_note = tempo_events[last_tempo_event]["value1"]
            second_per_tick = microseconds_per_quarter_note / ticks_per_quarter / 1_000_000

        time = ref_time + (tick - ref_tick) * second_per_tick
        events_times_mv[i] = time

    return events_times


@cython.boundscheck(False)
@cython.wraparound(False)
def get_event_times_soa_fast(
    midi_tick: ndarray, midi_event_type: ndarray, tempo_tick: ndarray, tempo_value1: ndarray, ticks_per_quarter: int
) -> ndarray:
    """Optimized version using struct of arrays."""
    num_events: cython.Py_ssize_t = midi_tick.shape[0]
    num_tempo_events: cython.Py_ssize_t = tempo_tick.shape[0]
    events_times: ndarray = np.zeros(num_events, dtype=np.float32)

    # Type the memoryviews
    midi_tick_mv = cython.cast(cython.int[:], midi_tick)
    tempo_tick_mv = cython.cast(cython.int[:], tempo_tick)
    tempo_value1_mv = cython.cast(cython.int[:], tempo_value1)
    events_times_mv = cython.cast(cython.float[:], events_times)

    tick: c_int32_t = 0
    time: cython.double = 0.0
    quarter_notes_per_minute_init: cython.double = 120.0
    second_per_tick: cython.double = quarter_notes_per_minute_init / ticks_per_quarter / 60.0
    ref_tick: c_int32_t = 0
    ref_time: cython.double = 0.0
    last_tempo_event: cython.Py_ssize_t = -1
    delta_tick: c_int32_t
    microseconds_per_quarter_note: cython.double
    i: cython.Py_ssize_t

    for i in range(num_events):
        delta_tick = midi_tick_mv[i] - tick
        tick += delta_tick

        # Check for tempo changes
        while last_tempo_event + 1 < num_tempo_events and tick >= tempo_tick_mv[last_tempo_event + 1]:
            last_tempo_event += 1
            ref_time = ref_time + (tempo_tick_mv[last_tempo_event] - ref_tick) * second_per_tick
            ref_tick = tempo_tick_mv[last_tempo_event]
            microseconds_per_quarter_note = tempo_value1_mv[last_tempo_event]
            second_per_tick = microseconds_per_quarter_note / ticks_per_quarter / 1_000_000

        time = ref_time + (tick - ref_tick) * second_per_tick
        events_times_mv[i] = time

    return events_times


@cython.boundscheck(False)
@cython.wraparound(False)
def read_var_length_fast(data: bytes, offset: int) -> tuple[int, int]:
    """Reads a variable-length quantity from the MIDI file.

    Returns (value, new_offset)
    """
    data_mv = cython.cast(cython.uchar[:], data)
    result: c_int32_t = 0
    byte: c_uint8_t
    current_offset: cython.int = offset

    while True:
        byte = data_mv[current_offset]
        result = (result << 7) | (byte & 0x7F)
        current_offset += 1
        if byte & 0x80 == 0:
            break

    return result, current_offset


@cython.boundscheck(False)
@cython.wraparound(False)
def write_var_length(value: int) -> bytes:
    """Writes a variable-length quantity to bytes."""
    if value == 0:
        return bytes([0])

    bytes_list: list = []
    temp_value: cython.int = value

    # Extract the 7-bit groups
    while temp_value > 0:
        bytes_list.insert(0, temp_value & 0x7F)
        temp_value >>= 7

    # Set the continuation bit for all but the last byte
    for i in range(len(bytes_list) - 1):
        bytes_list[i] |= 0x80

    return bytes(bytes_list)


@cython.boundscheck(False)
@cython.wraparound(False)
def parse_midi_header(data: bytes) -> tuple[int, int, int]:
    """Parse MIDI header chunk.

    Returns (format_type, num_tracks, ticks_per_quarter)
    """
    data_mv = cython.cast(cython.uchar[:], data)

    # Check for MThd header
    if data_mv[0] != ord("M") or data_mv[1] != ord("T") or data_mv[2] != ord("h") or data_mv[3] != ord("d"):
        raise ValueError("Invalid MIDI header")

    # Header length should be 6
    header_length: cython.int = (data_mv[4] << 24) | (data_mv[5] << 16) | (data_mv[6] << 8) | data_mv[7]
    if header_length != 6:
        raise ValueError(f"Invalid header length: {header_length}")

    # Parse header data
    format_type: cython.int = (data_mv[8] << 8) | data_mv[9]
    num_tracks: cython.int = (data_mv[10] << 8) | data_mv[11]
    ticks_per_quarter: cython.int = (data_mv[12] << 8) | data_mv[13]

    return format_type, num_tracks, ticks_per_quarter


@cython.boundscheck(False)
@cython.wraparound(False)
def parse_midi_events(track_data: bytes) -> tuple[ndarray, int]:
    """Parse MIDI events from track data.

    Returns (events_array, ticks_per_quarter)
    """
    data_mv = cython.cast(cython.uchar[:], track_data)
    data_length: cython.int = len(track_data)

    # Pre-allocate events list (we'll convert to array later)
    events: list = []

    offset: cython.int = 0
    current_tick: c_int32_t = 0
    running_status: c_uint8_t = 0

    while offset < data_length:
        # Read delta time
        delta_time: cython.int
        delta_time, offset = read_var_length_fast(track_data, offset)
        current_tick += delta_time

        if offset >= data_length:
            break

        # Read event
        event_byte: c_uint8_t = data_mv[offset]
        offset += 1

        if event_byte >= 0x80:
            # Status byte
            running_status = event_byte
        else:
            # Use running status
            offset -= 1  # Backup to re-read as data byte
            event_byte = running_status

        # Parse the event based on type
        if event_byte >= 0x80 and event_byte <= 0xEF:
            # Channel message
            channel: c_uint8_t = event_byte & 0x0F
            event_type: c_uint8_t = (event_byte >> 4) & 0x07

            value1: c_int32_t = data_mv[offset]
            offset += 1

            value2: c_int32_t = 0
            if event_type != 4 and event_type != 5:  # Not program change or channel pressure
                value2 = data_mv[offset]
                offset += 1

            events.append(
                {
                    "tick": current_tick,
                    "event_type": event_type,
                    "channel": channel,
                    "value1": value1,
                    "value2": value2,
                    "value3": 0,
                    "value4": 0,
                }
            )

        elif event_byte == 0xFF:
            # Meta event
            meta_type: c_uint8_t = data_mv[offset]
            offset += 1

            length: cython.int
            length, offset = read_var_length_fast(track_data, offset)

            if meta_type == 0x51:  # Set tempo
                if length == 3:
                    microseconds_per_quarter: c_int32_t = (
                        (data_mv[offset] << 16) | (data_mv[offset + 1] << 8) | data_mv[offset + 2]
                    )
                    events.append(
                        {
                            "tick": current_tick,
                            "event_type": 6,  # Tempo change
                            "channel": 0,
                            "value1": microseconds_per_quarter,
                            "value2": 0,
                            "value3": 0,
                            "value4": 0,
                        }
                    )

            offset += length

        # System exclusive or other
        elif event_byte == 0xF0 or event_byte == 0xF7:
            length: cython.int
            length, offset = read_var_length_fast(track_data, offset)
            offset += length
        else:
            offset += 1  # Skip unknown event

    # Convert to structured array
    if events:
        events_array = np.array(
            [
                (e["tick"], e["event_type"], e["channel"], e["value1"], e["value2"], e["value3"], e["value4"])
                for e in events
            ],
            dtype=event_dtype,
        )
    else:
        events_array = np.array([], dtype=event_dtype)

    return events_array, current_tick


@cython.boundscheck(False)
@cython.wraparound(False)
def parse_midi_track_soa_fast(track_data: bytes, offset: int):
    """
    Optimized pure Python Cython implementation using pre-allocated arrays instead of Python lists.
    Returns struct of arrays format for better performance.
    """
    data_mv = cython.cast(cython.uchar[:], track_data)
    data_length: cython.int = len(track_data)
    
    # Check track header 
    if len(track_data) < offset + 8:
        raise ValueError("Track data too short")
        
    # Read MTrk header (skip for now, assume valid)
    track_length: cython.int = (
        (track_data[offset + 4] << 24) |
        (track_data[offset + 5] << 16) | 
        (track_data[offset + 6] << 8) |
        track_data[offset + 7]
    )
    offset += 8
    
    if track_length <= 0:
        raise ValueError("Track length must be positive")
    
    track_end: cython.int = offset + track_length
    if track_end > data_length:
        raise ValueError("Track length too large")
    
    # Pre-allocate arrays - estimate max events as track_length / 2 (conservative)
    max_events: cython.int = track_length // 2
    if max_events < 100:
        max_events = 100
        
    tick_array = np.zeros(max_events, dtype=np.int32)
    event_type_array = np.zeros(max_events, dtype=np.uint8) 
    channel_array = np.zeros(max_events, dtype=np.uint8)
    value1_array = np.zeros(max_events, dtype=np.int32)
    value2_array = np.zeros(max_events, dtype=np.int16)
    value3_array = np.zeros(max_events, dtype=np.uint8)
    value4_array = np.zeros(max_events, dtype=np.uint8)
    
    track_name = b""
    lyrics = []
    
    tick: cython.int = 0
    delta_ticks: cython.int
    status_byte: cython.uchar
    meta_type: cython.uchar
    meta_length: cython.int
    current_tempo: cython.int
    numerator: cython.uchar
    denominator_power_of_2: cython.uchar
    clocks_per_click: cython.uchar
    notated_32nd_notes_per_beat: cython.uchar
    sysex_length: cython.int
    channel: cython.uchar = 0
    message_type: cython.uchar = 0  # Initialize for running status
    pitch: cython.uchar
    velocity: cython.uchar
    number: cython.uchar
    value: cython.uchar
    pressure: cython.uchar
    program: cython.uchar
    pitch_bend_value: cython.int
    num_events: cython.int = 0
    
    while offset < track_end:
        # Read delta time
        delta_ticks, offset = read_var_length_fast(track_data, offset)
        tick += delta_ticks
        status_byte = data_mv[offset]
        offset += 1
        
        if status_byte == 0xFF:  # Meta event
            meta_type = data_mv[offset]
            offset += 1
            meta_length, offset = read_var_length_fast(track_data, offset)
            meta_data = track_data[offset:offset + meta_length]
            offset += meta_length
            
            if meta_type == 0x51:  # Set Tempo event
                current_tempo = ((meta_data[0] << 16) | (meta_data[1] << 8) | meta_data[2])
                # Add tempo event
                tick_array[num_events] = tick
                event_type_array[num_events] = 5
                channel_array[num_events] = 0
                value1_array[num_events] = current_tempo
                value2_array[num_events] = 0
                value3_array[num_events] = 0
                value4_array[num_events] = 0
                num_events += 1
            
            elif meta_type == 0x58:  # Time signature
                if meta_length != 4:
                    raise ValueError("Time signature meta event has wrong length")
                numerator = meta_data[0]
                denominator_power_of_2 = meta_data[1] 
                clocks_per_click = meta_data[2]
                notated_32nd_notes_per_beat = meta_data[3]
                # Add time signature event
                tick_array[num_events] = tick
                event_type_array[num_events] = 10
                channel_array[num_events] = 0
                value1_array[num_events] = numerator
                value2_array[num_events] = denominator_power_of_2
                value3_array[num_events] = clocks_per_click
                value4_array[num_events] = notated_32nd_notes_per_beat
                num_events += 1
            
            elif meta_type == 0x59:  # Key signature
                pass  # Ignore for now
            
            elif meta_type == 0x03:  # Track name
                track_name = meta_data
            
            elif meta_type == 0x01:  # Text event
                text = meta_data
                if not text.startswith(b"@") and not text.startswith(b"%") and tick > 0:
                    lyrics.append((tick, text))
            
            elif meta_type == 0x04:  # Lyric event
                pass
            
            elif meta_type == 0x2F:  # End of track
                # Add end of track event
                tick_array[num_events] = tick
                event_type_array[num_events] = 9
                channel_array[num_events] = 0
                value1_array[num_events] = 0
                value2_array[num_events] = 0
                value3_array[num_events] = 0
                value4_array[num_events] = 0
                num_events += 1
        
        elif status_byte == 0xF0:  # SysEx event
            sysex_length, offset = read_var_length_fast(track_data, offset)
            offset += sysex_length
        
        elif status_byte in (0xF1, 0xF3):  # 1-byte messages
            offset += 1
        
        elif status_byte == 0xF2:  # 2-byte message (Song Position Pointer)
            offset += 2
        
        elif status_byte in (0xF8, 0xFA, 0xFC):  # Clock, Start, Continue
            pass  # No additional data
        
        elif status_byte <= 0xEF:  # MIDI channel messages
            if status_byte >= 0x80:
                channel = status_byte & 0x0F
                message_type = (status_byte & 0xF0) >> 4
            else:
                # Running status: use the last event type and channel
                offset -= 1
            
            if message_type == 0x9:  # Note On
                pitch = data_mv[offset]
                velocity = data_mv[offset + 1]
                offset += 2
                # Add note on event
                tick_array[num_events] = tick
                event_type_array[num_events] = 0
                channel_array[num_events] = channel
                value1_array[num_events] = pitch
                value2_array[num_events] = velocity
                value3_array[num_events] = 0
                value4_array[num_events] = 0
                num_events += 1
            
            elif message_type == 0x8:  # Note Off
                pitch = data_mv[offset]
                velocity = data_mv[offset + 1]
                offset += 2
                # Add note off event
                tick_array[num_events] = tick
                event_type_array[num_events] = 1
                channel_array[num_events] = channel
                value1_array[num_events] = pitch
                value2_array[num_events] = velocity
                value3_array[num_events] = 0
                value4_array[num_events] = 0
                num_events += 1
            
            elif message_type == 0xB:  # Control Change
                number = data_mv[offset]
                value = data_mv[offset + 1]
                offset += 2
                # Add control change event
                tick_array[num_events] = tick
                event_type_array[num_events] = 3
                channel_array[num_events] = channel
                value1_array[num_events] = number
                value2_array[num_events] = value
                value3_array[num_events] = 0
                value4_array[num_events] = 0
                num_events += 1
            
            elif message_type == 0xC:  # Program change
                program = data_mv[offset]
                # Add program change event
                tick_array[num_events] = tick
                event_type_array[num_events] = 4
                channel_array[num_events] = channel
                value1_array[num_events] = program
                value2_array[num_events] = 0
                value3_array[num_events] = 0
                value4_array[num_events] = 0
                num_events += 1
                offset += 1
            
            elif message_type == 0xE:  # Pitch Bend
                # Decode pitch bend from two bytes
                byte1 = data_mv[offset]
                byte2 = data_mv[offset + 1]
                if not (0 <= byte1 <= 127 and 0 <= byte2 <= 127):
                    raise ValueError("Invalid pitch bend bytes")
                unsigned = (byte2 << 7) | byte1
                pitch_bend_value = unsigned - 8192
                if not (-8192 <= pitch_bend_value <= 8191):
                    raise ValueError("Pitch bend value out of range")
                # Add pitch bend event
                tick_array[num_events] = tick
                event_type_array[num_events] = 2
                channel_array[num_events] = channel
                value1_array[num_events] = pitch_bend_value
                value2_array[num_events] = 0
                value3_array[num_events] = 0
                value4_array[num_events] = 0
                num_events += 1
                offset += 2
            
            elif message_type == 0xA:  # Polyphonic Aftertouch
                pitch = data_mv[offset]
                pressure = data_mv[offset + 1]
                offset += 2
                # Add polyphonic aftertouch event
                tick_array[num_events] = tick
                event_type_array[num_events] = 7
                channel_array[num_events] = channel
                value1_array[num_events] = pitch
                value2_array[num_events] = pressure
                value3_array[num_events] = 0
                value4_array[num_events] = 0
                num_events += 1
            
            elif message_type == 0xD:  # Channel Aftertouch
                pressure = data_mv[offset]
                offset += 1
                # Add channel aftertouch event
                tick_array[num_events] = tick
                event_type_array[num_events] = 6
                channel_array[num_events] = channel
                value1_array[num_events] = pressure
                value2_array[num_events] = 0
                value3_array[num_events] = 0
                value4_array[num_events] = 0
                num_events += 1
            else:
                offset += 1
        
        else:
            raise ValueError(f"Invalid status byte: {status_byte}")
    
    # Resize arrays to actual event count and return as SoA tuple
    return offset, (
        tick_array[:num_events].copy(),
        event_type_array[:num_events].copy(), 
        channel_array[:num_events].copy(),
        value1_array[:num_events].copy(),
        value2_array[:num_events].copy(),
        value3_array[:num_events].copy(),
        value4_array[:num_events].copy()
    ), track_name, lyrics

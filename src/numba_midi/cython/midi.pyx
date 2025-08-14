# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Cython implementation of MIDI parsing and encoding functions.
This replaces the numba-accelerated functions in midi.py for better distribution.
"""

import numpy as np
import cython
cimport numpy as cnp
from libc.stdint cimport uint8_t, uint16_t, uint32_t, int32_t, int16_t

# Initialize numpy
cnp.import_array()

# Define the event dtype structure that matches the numpy definition
event_dtype = np.dtype([
    ("tick", np.uint32),
    ("event_type", np.uint8),
    ("channel", np.uint8),
    ("value1", np.int32),
    ("value2", np.int16),
    ("value3", np.uint8),
    ("value4", np.uint8),
])


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray get_event_times(
    cnp.ndarray midi_events,
    cnp.ndarray tempo_events,
    int ticks_per_quarter
):
    """Get the time of each event in ticks and seconds."""
    cdef Py_ssize_t i
    cdef uint32_t tick = 0
    cdef double time = 0.0
    cdef double quarter_notes_per_minute_init = 120.0
    cdef double second_per_tick = quarter_notes_per_minute_init / ticks_per_quarter / 60.0
    cdef Py_ssize_t num_events = midi_events.shape[0]
    cdef Py_ssize_t num_tempo_events = tempo_events.shape[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] events_times = np.zeros(num_events, dtype=np.float32)
    cdef cnp.float32_t[:] events_times_mv = events_times
    cdef int ref_tick = 0
    cdef double ref_time = 0.0
    cdef int last_tempo_event = -1
    cdef uint32_t delta_tick
    cdef double microseconds_per_quarter_note
    for i in range(num_events):
        delta_tick = <uint32_t>(midi_events[i]['tick']) - tick
        tick += delta_tick
        while (last_tempo_event + 1 < num_tempo_events and 
               tick >= <uint32_t>(tempo_events[last_tempo_event + 1]['tick'])):
            last_tempo_event += 1
            ref_time = ref_time + (<uint32_t>(tempo_events[last_tempo_event]['tick']) - ref_tick) * second_per_tick
            ref_tick = <uint32_t>(tempo_events[last_tempo_event]['tick'])
            microseconds_per_quarter_note = <double>(tempo_events[last_tempo_event]['value1'])
            second_per_tick = microseconds_per_quarter_note / ticks_per_quarter / 1_000_000
        time = ref_time + (tick - ref_tick) * second_per_tick
        events_times_mv[i] = <cnp.float32_t>time
    return events_times


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline tuple read_var_length(bytes data_bytes, int offset):
    """Reads a variable-length quantity from the MIDI file."""
    cdef const uint8_t[::1] data = data_bytes
    cdef int value = 0
    cdef uint8_t byte
    while True:
        byte = data[offset]
        value = (value << 7) | (byte & 0x7F)
        offset += 1
        if byte & 0x80 == 0:
            break
    return value, offset


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline uint32_t unpack_uint32(bytes data_bytes):
    """Unpacks a 4-byte unsigned integer (big-endian)."""
    cdef const uint8_t[::1] data = data_bytes
    return ((<uint32_t>data[0]) << 24) | ((<uint32_t>data[1]) << 16) | ((<uint32_t>data[2]) << 8) | (<uint32_t>data[3])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline tuple unpack_uint8_pair(bytes data_bytes):
    """Unpacks two 1-byte unsigned integers."""
    cdef const uint8_t[::1] data = data_bytes
    return data[0], data[1]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline tuple unpack_uint16_triplet(bytes data_bytes):
    """Unpacks three 2-byte unsigned integers (big-endian)."""
    cdef const uint8_t[::1] data = data_bytes
    cdef uint16_t val1 = ((<uint16_t>data[0]) << 8) | (<uint16_t>data[1])
    cdef uint16_t val2 = ((<uint16_t>data[2]) << 8) | (<uint16_t>data[3])
    cdef uint16_t val3 = ((<uint16_t>data[4]) << 8) | (<uint16_t>data[5])
    return val1, val2, val3


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int32_t decode_pitch_bend(bytes data_bytes):
    """Decode pitch bend from two bytes."""
    cdef const uint8_t[::1] data = data_bytes
    cdef uint8_t byte1 = data[0]
    cdef uint8_t byte2 = data[1]
    assert 0 <= byte1 <= 127
    assert 0 <= byte2 <= 127
    cdef uint16_t unsigned = ((<uint16_t>byte2) << 7) | (<uint16_t>byte1)
    return <int32_t>(unsigned - 8192)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline tuple encode_pitchbend(int value):
    """Encodes a pitch bend value to two bytes."""
    assert -8192 <= value <= 8191, "Pitch bend value out of range"
    cdef uint16_t unsigned = <uint16_t>(value + 8192)
    cdef uint8_t byte1 = <uint8_t>(unsigned & 0x7F)
    cdef uint8_t byte2 = <uint8_t>((unsigned >> 7) & 0x7F)
    return byte1, byte2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline list encode_delta_time_cython(int delta_time):
    """Encodes delta time as a variable-length quantity."""
    if delta_time == 0:
        return [0]
    cdef list result = []
    cdef uint8_t byte
    while delta_time > 0:
        byte = <uint8_t>(delta_time & 0x7F)
        delta_time >>= 7
        if len(result) > 0:
            byte |= 0x80
        result.insert(0, byte)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list encode_delta_time(int delta_time):
    """Public wrapper for encode_delta_time."""
    return encode_delta_time_cython(delta_time)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple parse_midi_track(bytes data_bytes, int offset):
    """
    Cython implementation of MIDI track parsing.
    This replaces the numba-accelerated _parse_midi_track function.
    """
    cdef const uint8_t[::1] data = data_bytes
    cdef int data_len = len(data_bytes)
    
    # Check track header
    if unpack_uint32(data_bytes[offset:offset+4]) != unpack_uint32(b"MTrk"):
        raise ValueError("Invalid track chunk")
    
    cdef uint32_t track_length = unpack_uint32(data_bytes[offset + 4:offset + 8])
    offset += 8
    assert track_length > 0, "Track length must be positive"
    cdef int track_end = offset + <int>track_length
    assert track_end <= data_len, "Track length too large."
    
    # Use Python lists for dynamic growth
    midi_events = []
    track_name = b""
    lyrics = []
    
    cdef uint32_t tick = 0
    cdef uint32_t delta_ticks
    cdef uint8_t status_byte, meta_type
    cdef int meta_length, current_tempo
    cdef uint8_t numerator, denominator_power_of_2, clocks_per_click, notated_32nd_notes_per_beat
    cdef int sysex_length
    cdef uint8_t channel = 0, message_type = 0  # Initialize for running status
    cdef uint8_t pitch, velocity, number, value, pressure, program
    cdef int32_t pitch_bend_value
    cdef bytes meta_data, text
    
    while offset < track_end:
        delta_ticks, offset = read_var_length(data_bytes, offset)
        tick += delta_ticks
        status_byte = data[offset]
        offset += 1
        
        if status_byte == 0xFF:  # Meta event
            meta_type = data[offset]
            offset += 1
            meta_length, offset = read_var_length(data_bytes, offset)
            meta_data = data_bytes[offset:offset + meta_length]
            offset += meta_length
            
            if meta_type == 0x51:  # Set Tempo event
                current_tempo = ((<int>meta_data[0]) << 16) | ((<int>meta_data[1]) << 8) | (<int>meta_data[2])
                midi_events.append((tick, 5, 0, current_tempo, 0, 0, 0))
            
            elif meta_type == 0x58:  # Time signature
                assert meta_length == 4, "Time signature meta event has wrong length"
                numerator = meta_data[0]
                denominator_power_of_2 = meta_data[1]
                clocks_per_click = meta_data[2]
                notated_32nd_notes_per_beat = meta_data[3]
                midi_events.append((tick, 10, 0, numerator, denominator_power_of_2, clocks_per_click, notated_32nd_notes_per_beat))
            
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
                midi_events.append((tick, 9, 0, 0, 0, 0, 0))
        
        elif status_byte == 0xF0:  # SysEx event
            sysex_length, offset = read_var_length(data_bytes, offset)
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
                pitch, velocity = unpack_uint8_pair(data_bytes[offset:offset + 2])
                offset += 2
                midi_events.append((tick, 0, channel, pitch, velocity, 0, 0))
            
            elif message_type == 0x8:  # Note Off
                pitch, velocity = unpack_uint8_pair(data_bytes[offset:offset + 2])
                offset += 2
                midi_events.append((tick, 1, channel, pitch, velocity, 0, 0))
            
            elif message_type == 0xB:  # Control Change
                number, value = unpack_uint8_pair(data_bytes[offset:offset + 2])
                offset += 2
                midi_events.append((tick, 3, channel, number, value, 0, 0))
            
            elif message_type == 0xC:  # Program change
                program = data[offset]
                midi_events.append((tick, 4, channel, program, 0, 0, 0))
                offset += 1
            
            elif message_type == 0xE:  # Pitch Bend
                pitch_bend_value = decode_pitch_bend(data_bytes[offset:offset + 2])
                assert pitch_bend_value >= -8192 and pitch_bend_value <= 8191, "Pitch bend value out of range"
                midi_events.append((tick, 2, channel, pitch_bend_value, 0, 0, 0))
                offset += 2
            
            elif message_type == 0xA:  # Polyphonic Aftertouch
                pitch, pressure = unpack_uint8_pair(data_bytes[offset:offset + 2])
                offset += 2
                midi_events.append((tick, 7, channel, pitch, pressure, 0, 0))
            
            elif message_type == 0xD:  # Channel Aftertouch
                pressure = data[offset]
                offset += 1
                midi_events.append((tick, 6, channel, pressure, 0, 0, 0))
            else:
                offset += 1
        
        else:
            raise ValueError(f"Invalid status byte: {status_byte}")
    
    # Convert list to numpy structured array
    cdef int num_events = len(midi_events)
    cdef cnp.ndarray midi_events_np = np.zeros(num_events, dtype=event_dtype)
    
    # Fill the structured array
    cdef int i
    for i in range(num_events):
        event = midi_events[i]
        midi_events_np[i]['tick'] = event[0]
        midi_events_np[i]['event_type'] = event[1]
        midi_events_np[i]['channel'] = event[2]
        midi_events_np[i]['value1'] = event[3]
        midi_events_np[i]['value2'] = event[4]
        midi_events_np[i]['value3'] = event[5]
        midi_events_np[i]['value4'] = event[6]
    
    return offset, midi_events_np, track_name, lyrics


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray encode_midi_track(
    bytes name,
    cnp.ndarray events
):
    """Encodes a MIDI track to bytes using Cython."""
    cdef list data = []
    
    # Add track name
    data.extend(encode_delta_time(0))
    data.extend([0xFF, 0x03, len(name)])
    data.extend(name)
    
    cdef uint32_t tick = 0
    cdef uint32_t event_tick, delta_time
    cdef uint8_t event_type, channel
    cdef int32_t value1
    cdef int16_t value2
    cdef uint8_t value3, value4
    cdef tuple pitch_bend_bytes
    
    cdef int i
    cdef int num_events = len(events)
    
    for i in range(num_events):
        # Access structured array fields
        event_tick = events[i]['tick']
        delta_time = event_tick - tick
        tick = event_tick
        
        event_type = events[i]['event_type']
        channel = events[i]['channel']
        value1 = events[i]['value1']
        value2 = events[i]['value2']
        value3 = events[i]['value3']
        value4 = events[i]['value4']
        
        data.extend(encode_delta_time(delta_time))
        
        if event_type == 0:  # Note On
            data.extend([0x90 | channel, value1, value2])
        elif event_type == 1:  # Note Off
            data.extend([0x80 | channel, value1, value2])
        elif event_type == 2:  # Pitch Bend
            pitch_bend_bytes = encode_pitchbend(value1)
            data.extend([0xE0 | channel, pitch_bend_bytes[0], pitch_bend_bytes[1]])
        elif event_type == 3:  # Control Change
            data.extend([0xB0 | channel, value1, value2])
        elif event_type == 4:  # Program Change
            data.extend([0xC0 | channel, value1])
        elif event_type == 5:  # Tempo Change
            data.extend([0xFF, 0x51, 3, value1 >> 16, (value1 >> 8) & 0xFF, value1 & 0xFF])
        elif event_type == 6:  # Channel Aftertouch
            data.extend([0xD0 | channel, value1])
        elif event_type == 9:  # End of track
            data.extend([0xFF, 0x2F, 0])
        elif event_type == 10:  # Time Signature Change
            data.extend([0xFF, 0x58, 4, value1, value2, value3, value4])
        else:
            raise ValueError(f"Invalid event type: {event_type}")
    
    return np.array(data, dtype=np.uint8)
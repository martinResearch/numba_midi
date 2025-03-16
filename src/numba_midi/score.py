"""Music score representtion based on structured numpy arrays."""

from copy import copy
from dataclasses import dataclass

from numba.core.decorators import njit
import numpy as np

from numba_midi.instruments import instrument_to_program
from numba_midi.midi import event_dtype, get_event_times, load_midi_score, Midi, MidiTrack, save_midi_file

note_dtype = np.dtype(
    [
        ("start", np.float32),
        ("start_tick", np.int32),
        ("duration", np.float32),
        ("duration_tick", np.int32),
        ("pitch", np.int32),
        ("velocity_on", np.uint8),
    ]
)

control_dtype = np.dtype([("time", np.float32), ("tick", np.int32), ("number", np.int32), ("value", np.int32)])
pedal_dtype = np.dtype([("time", np.float32), ("tick", np.int32), ("duration", np.float32)])
pitch_bend_dtype = np.dtype([("time", np.float32), ("tick", np.int32), ("value", np.int32)])
tempo_dtype = np.dtype([("time", np.float32), ("tick", np.int32), ("bpm", np.float32)])


@dataclass
class Track:
    """MIDI track representation."""

    channel: int
    program: int
    midi_track_id: int
    is_drum: bool
    name: str
    notes: np.ndarray  # 1D structured numpy array with note_dtype elements
    controls: np.ndarray  # 1D structured numpy array with control_dtype elements
    pedals: np.ndarray  # 1D structured numpy array with pedal_dtype elements
    pitch_bends: np.ndarray  # 1D structured numpy array with pitch_bend_dtype elements

    def __post_init__(self) -> None:
        assert self.notes.dtype == note_dtype, "Notes must be a structured numpy array with note_dtype elements"
        assert self.controls.dtype == control_dtype, (
            "Controls must be a structured numpy array with control_dtype elements"
        )
        assert self.pedals.dtype == pedal_dtype, "Pedals must be a structured numpy array with pedal_dtype elements"
        assert self.pitch_bends.dtype == pitch_bend_dtype, (
            "Pitch bends must be a structured numpy array with pitch_bend_dtype elements"
        )

    def last_tick(self) -> int:
        """Get the last tick of the track."""
        return np.max(self.notes["start_tick"] + self.notes["duration_tick"])

    def __repr__(self) -> str:
        return (
            f"Track {self.name} with {len(self.notes)} notes, program={self.program}, "
            f"{len(self.controls)} controls, {len(self.pedals)} pedals, {len(self.pitch_bends)} pitch bends"
        )


@dataclass
class Score:
    """MIDI score representation."""

    tracks: list[Track]
    duration: float
    numerator: int
    denominator: int
    clocks_per_click: int
    ticks_per_quarter: int
    notated_32nd_notes_per_beat: int
    tempo: np.ndarray  # 1D structured numpy array with tempo_dtype elements
    lyrics: list[tuple[int, str]] | None = None

    def __repr__(self) -> str:
        num_notes = sum(len(track.notes) for track in self.tracks)
        last_tick = max(track.last_tick() for track in self.tracks)
        return (
            f"Score with {len(self.tracks)} tracks, {num_notes} notes, ",
            f"last tick {last_tick} and duration {self.duration}",
        )


@njit(cache=True)
def extract_notes_start_stop_numba(sorted_note_events: np.ndarray) -> np.ndarray:
    """Extract the notes from the sorted note events.
    The note events are assumed to be sorted by pitch and then by start time.
    """
    note_start_ids = []
    note_stop_ids = []
    last_pitch = -1
    num_active_notes = 0
    last_channel = -1
    for k in range(len(sorted_note_events)):
        if not last_pitch == sorted_note_events[k]["value1"] or not last_channel == sorted_note_events[k]["channel"]:
            # remove  unfinished notes
            while num_active_notes > 0:
                note_start_ids.pop()
                num_active_notes -= 1
            assert num_active_notes == 0
            # assert len(note_start_ids)==len(note_stop_ids)
            # assert np.all(sorted_note_events[note_start_ids]["value1"] == sorted_note_events[note_stop_ids]["value1"])
        last_pitch = sorted_note_events[k]["value1"]
        last_channel = sorted_note_events[k]["channel"]

        if sorted_note_events[k]["event_type"] == 0 and sorted_note_events[k]["value2"] > 0:
            note_start_ids.append(k)
            num_active_notes += 1
        elif num_active_notes > 0:
            note_stop_ids.append(k)
            num_active_notes -= 1
        assert num_active_notes >= 0, "num_active_notes should be non negative"

    # assert len(note_start_ids) == len(note_stop_ids) + num_active_notes
    while num_active_notes > 0:
        # remove unfinished notes
        note_start_ids.pop()
        num_active_notes -= 1
    assert len(note_start_ids) == len(note_stop_ids)
    # assert np.all(sorted_note_events[note_start_ids]["value1"] == sorted_note_events[note_stop_ids]["value1"])
    return note_start_ids, note_stop_ids


def get_events_program(events: np.ndarray) -> np.ndarray:
    channel_to_program = np.zeros((16), dtype=np.int32)
    program = np.zeros((len(events)), dtype=np.int32)

    for i in range(len(events)):
        if events[i]["event_type"] == 4:
            channel_to_program[events[i]["channel"]] = events[i]["value1"]
        program[i] = channel_to_program[events[i]["channel"]]
    return program


def group_data(keys: list[np.ndarray]) -> dict[tuple, np.ndarray]:
    order = np.lexsort(keys)
    # make sure the keys have the same length
    for k in keys:
        assert len(k) == len(keys[0]), "Keys must have the same length"
    is_boundary = np.zeros((len(keys[0]) - 1), dtype=bool)
    for i in range(len(keys)):
        is_boundary |= np.diff(keys[i][order]) != 0
    boundaries = np.nonzero(is_boundary)[0]
    starts = np.concatenate(([0], boundaries + 1))
    ends = np.concatenate((boundaries + 1, [len(keys[0])]))
    groups = {}
    for i in range(len(starts)):
        key = tuple([k[order[starts[i]]] for k in keys])
        groups[key] = order[starts[i] : ends[i]]
        for k in range(len(keys)):
            assert np.all(keys[k][groups[key]] == key[k])
    return groups


def extract_notes_start_stop(note_events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # midi_order = np.arange(len(note_events), dtype=np.int32)
    notes_order = np.lexsort(
        (note_events["event_type"], note_events["tick"], note_events["value1"], note_events["channel"])
    )
    sorted_note_events = note_events[notes_order]
    # sorted_notes_times = notes_times[notes_order]
    # deal with the case there are multiple consecutive note on events on the same note
    # if allow_overlap:

    ordered_note_start_ids, ordered_note_stop_ids = extract_notes_start_stop_numba(sorted_note_events)
    note_start_ids = notes_order[ordered_note_start_ids]
    note_stop_ids = notes_order[ordered_note_stop_ids]

    # restore order to mach the one in the original midi file
    order = np.argsort(note_start_ids)
    note_start_ids = note_start_ids[order]
    note_stop_ids = note_stop_ids[order]

    return note_start_ids, note_stop_ids


def midi_to_score(midi_score: Midi) -> Score:
    """Convert a MidiScore to a Score.

    Convert from event-based representation notes with durations
    """
    tracks = []
    duration = 0.0
    # assert len(midi_score.tracks) == 1, "Only one track is supported for now"
    ticks_per_quarter = midi_score.ticks_per_quarter
    all_tempo_events = []

    for midi_track in midi_score.tracks:
        tempo_change_mask = midi_track.events["event_type"] == 5
        num_tempo_change = np.sum(tempo_change_mask)
        if num_tempo_change > 0:
            tempo_change = midi_track.events[tempo_change_mask]
            # keep only the last tempo change for each tick
            keep = np.hstack((np.diff(tempo_change["tick"]) > 0, [True]))
            all_tempo_events.append(tempo_change[keep])
    if len(all_tempo_events) > 0:
        tempo_events = np.concatenate(all_tempo_events, axis=0)
        # sort by tick
        tempo_events = tempo_events[np.argsort(tempo_events["tick"])]
        # keep only the last tempo change for each tick
        # TODO should it be the first?
        tempo_events = tempo_events[np.hstack((np.diff(tempo_events["tick"]) > 0, [True]))]
    else:
        # if no tempo events are found, we create a default one
        tempo_events = np.zeros(1, dtype=event_dtype)
        tempo_events["event_type"] = 5
        tempo_events["channel"] = 0
        tempo_events["value1"] = 120 * 1000000.0 / 60.0
        tempo_events["value2"] = 0
        tempo_events["tick"] = 0

    tempo_events_times = get_event_times(tempo_events, tempo_events, midi_score.ticks_per_quarter)
    tempo = np.zeros(len(tempo_events), dtype=tempo_dtype)
    tempo["time"] = tempo_events_times
    tempo["tick"] = tempo_events["tick"]
    tempo["bpm"] = tempo_events["value1"] * 60 / 1000000.0

    lyrics = []
    for midi_track_id, midi_track in enumerate(midi_score.tracks):
        if midi_track.lyrics is not None:
            lyrics.extend(midi_track.lyrics)

    lyrics = np.array(lyrics, dtype=[("tick", np.int32), ("text", "U256")])
    # sort the lyrics by tick
    lyrics = lyrics[np.argsort(lyrics["tick"])]

    for midi_track_id, midi_track in enumerate(midi_score.tracks):
        numerator = midi_track.numerator
        denominator = midi_track.denominator
        clocks_per_click = midi_track.clocks_per_click
        notated_32nd_notes_per_beat = midi_track.notated_32nd_notes_per_beat

        # get the program for each channel
        events_programs = get_events_program(midi_track.events)
        # program_change = midi_track.events[midi_track.events["event_type"] == 4]

        # channel_to_program = np.zeros((16), dtype=np.int32)
        # channel_to_program[program_change["channel"]] = program_change["value1"]
        # # assert np.all(channel_to_program[program_change["channel"]] == program_change["value1"]), (
        # #     "Multiple program changes for the same channel"
        # # )

        events = midi_track.events
        # compute the tick and time of each event
        events_ticks = events["tick"]

        events_times = get_event_times(events, tempo_events, midi_score.ticks_per_quarter)

        # sort in lexicographic order by pitch first and then by tick, then even type
        # this allows to have a order for the events that simplifies the
        # extracting matching note starts and stops
        # we sort by inverse of event type in order to deal with the case there is no gap
        # between two consecutive notes
        # extract the event of type note on or note off
        notes_events_ids = np.nonzero((events["event_type"] == 0) | (events["event_type"] == 1))[0]
        if len(notes_events_ids) > 0:
            note_events = events[notes_events_ids]

            note_start_ids, note_stop_ids = extract_notes_start_stop(note_events)
            assert np.all(np.diff(note_start_ids) >= 0), "note start ids should be sorted"
            if note_start_ids.size == 0:
                continue
            note_start_ids = notes_events_ids[note_start_ids]
            note_stop_ids = notes_events_ids[note_stop_ids]

            # the note stop need to be give the same program as the note start
            note_programs = events_programs[note_start_ids]
            note_channels = events["channel"][note_start_ids]
            note_start_events = events[note_start_ids]
            note_stop_events = events[note_stop_ids]
            note_starts_time = events_times[note_start_ids]
            note_stops_time = events_times[note_stop_ids]

            note_groups = group_data([note_programs, note_channels])
        else:
            continue
        # sort all the events in lexicographic order by channel and tick
        # this allows to have a order for the events that simplifies the code to process them
        events_groups = group_data([events_programs, events["channel"]])

        for group_keys, track_events_ids in events_groups.items():
            if group_keys not in note_groups:
                continue

            track_notes_ids = note_groups[group_keys]
            track_notes_ids = np.sort(track_notes_ids)  # to keep the original order of the notes in the midi
            # print(f"{group_keys} {len(track_notes_ids)}")
            track_program, track_channel = group_keys
            assert track_program >= 0 and track_program < 128, "program should be between 0 and 127"
            assert track_channel >= 0 and track_channel < 16, "channel should be between 0 and 15"
            track_events = events[track_events_ids]
            track_events_times = events_times[track_events_ids]
            track_events_ticks = events_ticks[track_events_ids]
            assert np.all(np.diff(track_events_ticks) >= 0)
            #  control change events
            control_change_mask = track_events["event_type"] == 3
            control_change_events = track_events[control_change_mask]
            controls = np.zeros(len(control_change_events), dtype=control_dtype)
            controls["time"] = track_events_times[control_change_mask]
            controls["tick"] = track_events_ticks[control_change_mask]
            controls["number"] = control_change_events["value1"]
            controls["value"] = control_change_events["value2"]

            pitch_bends_mask = track_events["event_type"] == 2
            pitch_bends_events = track_events[pitch_bends_mask]
            pitch_bends = np.zeros(len(pitch_bends_events), dtype=pitch_bend_dtype)
            pitch_bends["time"] = track_events_times[pitch_bends_mask]
            pitch_bends["tick"] = track_events_ticks[pitch_bends_mask]
            pitch_bends["value"] = pitch_bends_events["value1"]

            # extract the event of type note on or note off

            notes_np = np.zeros(len(track_notes_ids), dtype=note_dtype)
            track_note_start_events = note_start_events[track_notes_ids]
            track_note_stop_events = note_stop_events[track_notes_ids]
            notes_np["start"] = note_starts_time[track_notes_ids]
            notes_np["start_tick"] = track_note_start_events["tick"]
            notes_np["duration"] = note_stops_time[track_notes_ids] - note_starts_time[track_notes_ids]
            notes_np["duration_tick"] = np.uint32(
                np.int64(track_note_stop_events["tick"]) - np.int64(track_note_start_events["tick"])
            )
            # assert np.all(notes_np["duration_tick"] > 0), "duration_tick should be strictly positive"
            notes_np["pitch"] = track_note_start_events["value1"]
            notes_np["velocity_on"] = track_note_start_events["value2"]
            if len(notes_np) > 0:
                duration = max(duration, np.max(notes_np["start"] + notes_np["duration"]))

            # reorder using the original midi order
            notes_np = notes_np[np.argsort(note_start_ids[track_notes_ids])]
            if notes_np.size > 0:
                track = Track(
                    channel=int(track_channel),
                    midi_track_id=midi_track_id,
                    program=int(track_program),
                    is_drum=False,  # FIXME
                    name=midi_track.name,
                    notes=notes_np,
                    controls=controls,
                    pedals=np.zeros((0,), dtype=pedal_dtype),  # FIXME not supported yet
                    pitch_bends=pitch_bends,
                )
                tracks.append(track)

    # sort the tracks by channel and program
    # tracks.sort(key=lambda x: (x.channel, x.program))
    return Score(
        tracks=tracks,
        lyrics=lyrics,
        duration=duration,
        numerator=numerator,
        denominator=denominator,
        clocks_per_click=clocks_per_click,
        ticks_per_quarter=ticks_per_quarter,
        notated_32nd_notes_per_beat=notated_32nd_notes_per_beat,
        tempo=tempo,
    )


def has_duplicate_values(values: list) -> bool:
    list_values = list(values)
    return len(list_values) != len(set(list_values))


def score_to_midi(score: Score) -> Midi:
    """Convert a Score to a Midi file and save it."""
    midi_tracks = []
    # if two tracks use the same channel, we use multiple midi tracks
    use_multiple_tracks = len(set(track.midi_track_id for track in score.tracks)) > 1
    if has_duplicate_values((track.channel for track in score.tracks)):
        # multiple tracks with the same channel not supported because
        # it requires carefull program changes each time the instrument changes in the channel
        # using multiple tracks instead
        use_multiple_tracks = True

    if not use_multiple_tracks:
        num_events = 0
        for track in score.tracks:
            num_events += len(track.notes) * 2 + len(track.controls) + len(track.pitch_bends) + 1
        num_events += len(score.tempo)
        events = np.zeros(num_events, dtype=event_dtype)

        id_start = 0

        tempo = score.tempo
        events["tick"][id_start : id_start + len(tempo)] = tempo["tick"]
        events["event_type"][id_start : id_start + len(tempo)] = 5
        events["channel"][id_start : id_start + len(tempo)] = 0
        events["value1"][id_start : id_start + len(tempo)] = tempo["bpm"] * 1000000.0 / 60.0
        events["value2"][id_start : id_start + len(tempo)] = 0
        id_start += len(tempo)

        lyrics = score.lyrics
    else:
        # create track with the tempo changes
        tempo_events = np.zeros(len(score.tempo), dtype=event_dtype)
        tempo_events["event_type"] = 5
        tempo_events["channel"] = 0
        tempo_events["value1"] = score.tempo["bpm"] * 1000000.0 / 60.0
        tempo_events["value2"] = 0
        tempo_events["tick"] = score.tempo["tick"]
        midi_tracks.append(
            MidiTrack(
                name="tempo",
                lyrics=None,
                events=tempo_events,
                numerator=score.numerator,
                denominator=score.denominator,
                clocks_per_click=score.clocks_per_click,
                notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
            )
        )
        # create a track for the lyrics
        if score.lyrics is not None:
            midi_tracks.append(
                MidiTrack(
                    name="lyrics",
                    events=np.zeros(0, dtype=event_dtype),
                    lyrics=score.lyrics,
                    numerator=score.numerator,
                    denominator=score.denominator,
                    clocks_per_click=score.clocks_per_click,
                    notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
                )
            )
        lyrics = None

    for track in score.tracks:
        if use_multiple_tracks:
            num_events = len(track.notes) * 2 + len(track.controls) + len(track.pitch_bends) + 1
            events = np.zeros(num_events, dtype=event_dtype)
            id_start = 0

        num_track_events = len(track.notes) * 2 + len(track.controls) + len(track.pitch_bends) + 1
        events["channel"][id_start : id_start + num_track_events] = track.channel

        # add the program change event
        events["event_type"][id_start] = 4
        events["value1"][id_start] = track.program
        events["value2"][id_start] = 0
        events["tick"][id_start] = 0
        # print(events[id_start] )
        id_start += 1

        # add the notes on events
        events["event_type"][id_start : id_start + len(track.notes)] = 0
        events["value1"][id_start : id_start + len(track.notes)] = track.notes["pitch"]
        events["value2"][id_start : id_start + len(track.notes)] = track.notes["velocity_on"]
        events["tick"][id_start : id_start + len(track.notes)] = track.notes["start_tick"]
        id_start += len(track.notes)

        # add the notes off events

        events["event_type"][id_start : id_start + len(track.notes)] = 1
        events["value1"][id_start : id_start + len(track.notes)] = track.notes["pitch"]
        events["value2"][id_start : id_start + len(track.notes)] = 0
        events["tick"][id_start : id_start + len(track.notes)] = (
            track.notes["start_tick"] + track.notes["duration_tick"]
        )
        # assert np.all(track.notes["duration_tick"] > 0), "duration_tick should be strictly positive"
        id_start += len(track.notes)

        # add the control change events

        events["event_type"][id_start : id_start + len(track.controls)] = 3
        events["value1"][id_start : id_start + len(track.controls)] = track.controls["number"]
        events["value2"][id_start : id_start + len(track.controls)] = track.controls["value"]
        events["tick"][id_start : id_start + len(track.controls)] = track.controls["tick"]
        id_start += len(track.controls)

        # add the pitch bend events

        events["event_type"][id_start : id_start + len(track.pitch_bends)] = 2
        events["value1"][id_start : id_start + len(track.pitch_bends)] = track.pitch_bends["value"]
        events["value2"][id_start : id_start + len(track.pitch_bends)] = 0
        events["tick"][id_start : id_start + len(track.pitch_bends)] = track.pitch_bends["tick"]
        id_start += len(track.pitch_bends)

        if use_multiple_tracks:
            order = np.lexsort((np.arange(len(events)), events["tick"]))
            events = events[order]
            midi_track = MidiTrack(
                name=track.name,
                events=events,
                lyrics=lyrics,
                numerator=score.numerator,
                denominator=score.denominator,
                clocks_per_click=score.clocks_per_click,
                notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
            )
            midi_tracks.append(midi_track)

    if not use_multiple_tracks:
        # sort by tick and keep the original order of the events
        # for event with the same tick,
        order = np.lexsort((np.arange(len(events)), events["tick"]))
        events = events[order]

        midi_track = MidiTrack(
            name=track.name,
            events=events,
            lyrics=lyrics,
            numerator=score.numerator,
            denominator=score.denominator,
            clocks_per_click=score.clocks_per_click,
            notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
        )
        midi_tracks = [midi_track]
    midi_score = Midi(tracks=midi_tracks, ticks_per_quarter=score.ticks_per_quarter)
    return midi_score


def load_score(file_path: str) -> Score:
    """Loads a MIDI file and converts it to a Score."""
    midi_raw = load_midi_score(file_path)
    return midi_to_score(midi_raw)


def save_score_to_midi(score: Score, file_path: str) -> None:
    """Saves a Score to a MIDI file."""
    midi_score = score_to_midi(score)
    save_midi_file(midi_score, file_path)


def merge_tracks_with_same_program(score: Score) -> Score:
    # merge tracks with the same program
    tracks_dict: dict[int, Track] = {}
    for track in score.tracks:
        if track.program not in tracks_dict:
            tracks_dict[track.program] = track
        else:
            tracks_dict[track.program].notes = np.concatenate((tracks_dict[track.program].notes, track.notes))
            tracks_dict[track.program].controls = np.concatenate((tracks_dict[track.program].controls, track.controls))
            tracks_dict[track.program].pedals = np.concatenate((tracks_dict[track.program].pedals, track.pedals))
            tracks_dict[track.program].pitch_bends = np.concatenate(
                (tracks_dict[track.program].pitch_bends, track.pitch_bends)
            )
    # sort the note, control, pedal and pitch_bend arrays
    for _, track in tracks_dict.items():
        track.notes = np.sort(track.notes, order="start")
        track.controls = np.sort(track.controls, order="time")
        track.pedals = np.sort(track.pedals, order="time")
        track.pitch_bends = np.sort(track.pitch_bends, order="time")
    # sort tracks by program
    tracks = list(tracks_dict.values())
    tracks.sort(key=lambda x: x.program)

    new_score = Score(
        tracks=tracks,
        duration=score.duration,
        numerator=score.numerator,
        denominator=score.denominator,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
    )
    return new_score


def filter_instruments(score: Score, instrument_names: list[str]) -> Score:
    """Filter the tracks of the score to keep only the ones with the specified instrument names."""
    tracks = []

    programs = set([instrument_to_program[instrument_name] for instrument_name in instrument_names])
    for track in score.tracks:
        if track.is_drum:
            continue
        if track.program in programs:
            tracks.append(track)
    return Score(
        tracks=tracks,
        duration=score.duration,
        numerator=score.numerator,
        denominator=score.denominator,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
    )


def remove_empty_tracks(score: Score) -> Score:
    """Remove the tracks of the score that have no notes."""
    tracks = []
    for track in score.tracks:
        if track.notes.size > 0:
            tracks.append(track)
    return Score(
        tracks=tracks,
        duration=score.duration,
        numerator=score.numerator,
        denominator=score.denominator,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
    )


def remove_pitch_bends(score: Score) -> Score:
    """Remove the pitch bends from the score."""
    tracks = []
    for track in score.tracks:
        new_track = copy(track)
        new_track.pitch_bends = np.array([], dtype=track.pitch_bends.dtype)
        tracks.append(new_track)

    return Score(
        tracks=tracks,
        duration=score.duration,
        numerator=score.numerator,
        denominator=score.denominator,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
    )


def remove_control_changes(score: Score) -> Score:
    """Remove the control changes from the score."""
    tracks = []
    for track in score.tracks:
        new_track = copy(track)
        new_track.controls = np.array([], dtype=track.controls.dtype)
        tracks.append(new_track)

    return Score(
        tracks=tracks,
        duration=score.duration,
        numerator=score.numerator,
        denominator=score.denominator,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
    )


def filter_pitch(score: Score, pitch_min: int, pitch_max: int) -> Score:
    tracks = []
    for track in score.tracks:
        new_track = copy(track)
        keep = (track.notes["pitch"] >= pitch_min) & (track.notes["pitch"] < pitch_max)
        new_track.notes = track.notes[keep]
        tracks.append(new_track)
    return Score(
        tracks=tracks,
        duration=score.duration,
        numerator=score.numerator,
        denominator=score.denominator,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
    )


@njit(cache=True)
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


def get_overlapping_notes(notes: np.ndarray) -> np.ndarray:
    order = np.lexsort((notes["start"], notes["pitch"]))
    overlapping_notes = _get_overlapping_notes_pairs_jit(notes["start"], notes["duration"], notes["pitch"], order)
    return overlapping_notes


def get_overlapping_notes_ticks(notes: np.ndarray) -> np.ndarray:
    order = np.lexsort((notes["start_tick"], notes["pitch"]))
    overlapping_notes = _get_overlapping_notes_pairs_jit(
        notes["start_tick"], notes["duration_tick"], notes["pitch"], order
    )
    return overlapping_notes


def check_no_overlapping_notes(notes: np.ndarray, use_ticks: bool = True) -> None:
    """Check that there are no overlapping notes at the same pitch."""
    if use_ticks:
        overlapping_notes = get_overlapping_notes_ticks(notes)
    else:
        overlapping_notes = get_overlapping_notes(notes)
    if len(overlapping_notes) > 0:
        raise ValueError("Overlapping notes found")


def check_no_overlapping_notes_in_score(score: Score) -> None:
    for track in score.tracks:
        check_no_overlapping_notes(track.notes)


def time_to_ticks(time: float, tempo: np.ndarray, ticks_per_quarter: int) -> int:
    """Convert a time in seconds to ticks."""
    # get the tempo at the start of the time range
    tempo_idx = np.searchsorted(tempo["time"], time, side="right") - 1
    tempo_idx = np.clip(tempo_idx, 0, len(tempo) - 1)
    bpm = tempo["bpm"][tempo_idx]
    ref_ticks = tempo["tick"][tempo_idx]
    ref_time = tempo["time"][tempo_idx]
    ticks_per_second = bpm / 60.0 * ticks_per_quarter / 1_000_000.0
    return (ref_ticks + (time - ref_time) * ticks_per_second).astype(np.int32)


def crop_score(score: Score, start: float, duration: float) -> Score:
    """Crop a MIDI score to a specific time range.

    Note: the NoteOn events from before the start time are not kept
    and thus the sound may not be the same as the cropped original sound.
    """
    end = start + duration
    tracks = []

    for track in score.tracks:
        tick_start = time_to_ticks(start, track.tempo, score.ticks_per_quarter)
        tick_end = time_to_ticks(end, track.tempo, score.ticks_per_quarter)
        notes = track.notes
        notes_end = notes["start"] + notes["duration"]
        notes_keep = (notes["start"] < end) & (notes_end > start)
        new_notes = notes[notes_keep]
        if len(new_notes) == 0:
            continue
        new_notes["start"] = np.maximum(new_notes["start"] - start, 0)
        new_notes_end = np.minimum(notes_end[notes_keep] - start, end - start)
        new_notes["duration"] = new_notes_end - new_notes["start"]
        new_notes["start_tick"] = np.maximum(new_notes["start_tick"] - tick_start, 0)
        new_notes_end_tick = np.minimum(new_notes["start_tick"] + new_notes["duration_tick"], tick_end - tick_start)
        new_notes["duration_tick"] = new_notes_end_tick - new_notes["start_tick"]

        assert np.all(new_notes_end <= end - start), "Note end time exceeds score duration"

        check_no_overlapping_notes(new_notes)

        pedals_end = track.pedals["time"] + track.pedals["duration"]
        pedals_keep = (track.pedals["time"] < end) & (pedals_end > start)
        new_pedals = track.pedals[pedals_keep]
        new_pedals_end = np.minimum(pedals_end[pedals_keep], end) - start
        new_pedals["duration"] = new_pedals_end - new_pedals["time"]
        new_pedals["time"] = np.maximum(new_pedals["time"] - start, 0)
        new_pedals["tick"] = np.maximum(new_pedals["tick"] - tick_start, 0)

        controls_keep = (track.controls["time"] < end) & (track.controls["time"] >= start)
        new_controls = track.controls[controls_keep]
        new_controls["time"] = np.maximum(new_controls["time"] - start, 0)
        new_controls["tick"] = np.maximum(new_controls["tick"] - tick_start, 0)

        pitch_bends_keep = (track.pitch_bends["time"] < end) & (track.pitch_bends["time"] >= start)
        new_pitch_bends = track.pitch_bends[pitch_bends_keep]
        new_pitch_bends["time"] = np.maximum(new_pitch_bends["time"] - start, 0)
        new_pitch_bends["tick"] = np.maximum(new_pitch_bends["tick"] - tick_start, 0)

        tempo_keep = (track.tempo["time"] < end) & (track.tempo["time"] >= start)
        new_tempo = track.tempo[tempo_keep]
        new_tempo["time"] = np.maximum(new_tempo["time"] - start, 0)
        new_tempo["tick"] = np.maximum(new_tempo["tick"] - tick_start, 0)

        new_track = Track(
            channel=track.channel,
            program=track.program,
            is_drum=track.is_drum,
            name=track.name,
            notes=new_notes,
            controls=new_controls,
            pedals=new_pedals,
            pitch_bends=new_pitch_bends,
            tempo=track.tempo,
        )
        tracks.append(new_track)
    return Score(
        tracks=tracks,
        duration=duration,
        numerator=score.numerator,
        denominator=score.denominator,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
    )


def select_tracks(score: Score, track_ids: list[int]) -> Score:
    """Select only the tracks with the specified programs."""
    tracks = [score.tracks[track_id] for track_id in track_ids]
    return Score(
        tracks=tracks,
        duration=score.duration,
        numerator=score.numerator,
        denominator=score.denominator,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
    )


def distance(score1: Score, score2: Score, sort_tracks_with_programs: bool = False) -> float:
    assert len(score1.tracks) == len(score2.tracks), "The scores have different number of tracks"
    max_diff = 0
    tracks_1 = score1.tracks
    tracks_2 = score2.tracks
    if sort_tracks_with_programs:
        tracks_1 = sorted(tracks_1, key=lambda x: x.program)
        tracks_2 = sorted(tracks_2, key=lambda x: x.program)
    for track1, track2 in zip(tracks_1, tracks_2):
        print("Programs", track1.program, track2.program)
        print("Notes", track1.notes.shape, track2.notes.shape)
        print("Controls", track1.controls.shape, track2.controls.shape)
        print("Pedals", track1.pedals.shape, track2.pedals.shape)
        print("Pitch bends", track1.pitch_bends.shape, track2.pitch_bends.shape)

        # max note time difference
        max_start_time_diff = max(abs(track1.notes["start"] - track2.notes["start"]))
        max_duration_diff = max(abs(track1.notes["duration"] - track2.notes["duration"]))
        print("Max note start time difference", max_start_time_diff)
        print("Max note duration difference", max_duration_diff)
        max_diff = max(max_diff, max_start_time_diff, max_duration_diff)
        # max control time difference
        max_control_time_diff = max(abs(track1.controls["time"] - track2.controls["time"]))
        print("Max control time difference", max_control_time_diff)
        max_diff = max(max_diff, max_control_time_diff)
        # max pedal time difference
        if track1.pedals.size > 0:
            max_pedal_time_diff = max(abs(track1.pedals["time"] - track2.pedals["time"]))
            print("Max pedal time difference", max_pedal_time_diff)
            max_diff = max(max_diff, max_pedal_time_diff)
        # max pitch bend time difference
        if track1.pitch_bends.size > 0:
            max_pitch_bend_time_diff = max(abs(track1.pitch_bends["time"] - track2.pitch_bends["time"]))
            print("Max pitch bend time difference", max_pitch_bend_time_diff)
            max_diff = max(max_diff, max_pitch_bend_time_diff)
    return max_diff


def assert_scores_equal(
    score1: Score,
    score2: Score,
    sort_tracks_with_channel: bool = True,
    time_tol: float = 1e-3,
    value_tol: float = 1e-2,
) -> None:
    assert len(score1.tracks) == len(score2.tracks), "The scores have different number of tracks"
    max_diff = 0
    tracks_1 = score1.tracks
    tracks_2 = score2.tracks
    if sort_tracks_with_channel:
        tracks_1 = sorted(tracks_1, key=lambda x: x.channel)
        tracks_2 = sorted(tracks_2, key=lambda x: x.channel)

    for track1, track2 in zip(tracks_1, tracks_2):
        assert track1.name == track2.name, "Track names are different"
        assert track1.channel == track2.channel, "Track channels are different"
        assert track1.program == track2.program, "Track programs are different"
        # sort not by pitch then tick
        # notes1= track1.notes
        # notes2= track2.notes
        order1 = np.lexsort((range(len(track1.notes)), track1.notes["start_tick"], track1.notes["pitch"]))
        notes1 = track1.notes[order1]
        order2 = np.lexsort((range(len(track2.notes)), track2.notes["start_tick"], track2.notes["pitch"]))
        notes2 = track2.notes[order2]
        assert np.all(notes1["pitch"] == notes2["pitch"]), "Pitches are different"
        max_tick_diff = max(abs(notes1["start_tick"] - notes2["start_tick"]))
        assert max_tick_diff == 0, "Tick difference is not zero"
        max_duration_tick_diff = max(abs(notes1["duration_tick"] - notes2["duration_tick"]))
        assert max_duration_tick_diff == 0, "Duration tick difference is not zero"
        # max note time difference
        max_start_time_diff = max(abs(notes1["start"] - notes2["start"]))
        assert max_start_time_diff <= time_tol, f"Max note start time difference {max_start_time_diff}>{time_tol}"
        notes_stop_1 = notes1["start"] + notes1["duration"]
        notes_stop_2 = notes2["start"] + notes2["duration"]
        max_stop_diff = max(abs(notes_stop_1 - notes_stop_2))
        assert max_stop_diff <= time_tol, f"Max note end difference {max_stop_diff}>{time_tol}"
        # max note velocity difference
        velocify_abs_diff = abs(notes1["velocity_on"].astype(np.int16) - notes2["velocity_on"].astype(np.int16))
        max_velocity_diff = max(velocify_abs_diff)
        assert max_velocity_diff <= value_tol, f"Max note velocity difference {max_velocity_diff}>{value_tol}"
        # max note duration difference
        max_duration_diff = max(abs(notes1["duration"] - notes2["duration"]))
        assert max_duration_diff <= time_tol, f"Max note duration difference {max_duration_diff}>{time_tol}"

        # max control time difference
        assert track1.controls.shape == track2.controls.shape, "Different number of control events"
        if track1.controls.size > 0:
            max_control_time_diff = max(abs(track1.controls["time"] - track2.controls["time"]))
            assert max_control_time_diff <= time_tol, f"Max control time difference {max_control_time_diff}>{time_tol}"
            max_diff = max(max_diff, max_control_time_diff)
        # max pedal time difference
        assert track1.pedals.shape == track2.pedals.shape, "Different number of pedal events"
        if track1.pedals.size > 0:
            max_pedal_time_diff = max(abs(track1.pedals["time"] - track2.pedals["time"]))
            max_diff = max(max_diff, max_pedal_time_diff)
            assert max_pedal_time_diff <= time_tol, f"Max pedal time difference {max_pedal_time_diff}>{time_tol}"
        # max pitch bend time difference
        assert track1.pitch_bends.shape == track2.pitch_bends.shape, "Different number of pitch bend events"
        if track1.pitch_bends.size > 0:
            max_pitch_bend_time_diff = max(abs(track1.pitch_bends["time"] - track2.pitch_bends["time"]))
            max_diff = max(max_diff, max_pitch_bend_time_diff)
            assert max_pitch_bend_time_diff <= time_tol, (
                f"Max pitch bend time difference {max_pitch_bend_time_diff}>{time_tol}"
            )

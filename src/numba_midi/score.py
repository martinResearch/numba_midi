"""Music score represention based on structured numpy arrays."""

from copy import copy
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

from numba_midi._score_numba import (
    _get_overlapping_notes_pairs_jit,
    extract_notes_start_stop_numba,
    get_events_program,
    get_pedals_from_controls_jit,
)
from numba_midi.instruments import (
    instrument_to_program,
    program_to_instrument,
    program_to_instrument_group,
)
from numba_midi.midi import event_dtype, get_event_times, load_midi_score, Midi, MidiTrack, save_midi_file

NotesMode = Literal["no_overlap", "first_in_first_out", "note_off_stops_all"]

notes_mode_mapping: dict[NotesMode, int] = {
    "no_overlap": 0,
    "first_in_first_out": 1,
    "note_off_stops_all": 2,
}

note_dtype = np.dtype(
    [
        ("start", np.float64),
        ("start_tick", np.int32),
        ("duration", np.float64),
        ("duration_tick", np.int32),
        ("pitch", np.int32),
        ("velocity_on", np.uint8),
    ]
)

control_dtype = np.dtype([("time", np.float64), ("tick", np.int32), ("number", np.int32), ("value", np.int32)])
pedal_dtype = np.dtype(
    [("time", np.float64), ("tick", np.int32), ("duration", np.float64), ("duration_tick", np.int32)]
)
pitch_bend_dtype = np.dtype([("time", np.float64), ("tick", np.int32), ("value", np.int32)])
tempo_dtype = np.dtype([("time", np.float64), ("tick", np.int32), ("bpm", np.float64)])


@dataclass
class Track:
    """MIDI track representation."""

    program: int

    is_drum: bool
    name: str
    notes: np.ndarray  # 1D structured numpy array with note_dtype elements
    controls: np.ndarray  # 1D structured numpy array with control_dtype elements
    pedals: np.ndarray  # 1D structured numpy array with pedal_dtype elements
    pitch_bends: np.ndarray  # 1D structured numpy array with pitch_bend_dtype elements
    channel: Optional[int] = None
    midi_track_id: Optional[int] = None

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
        last_tick = 0
        if len(self.notes) > 0:
            last_tick = np.max(self.notes["start_tick"] + self.notes["duration_tick"])
        if len(self.controls) > 0:
            last_tick = max(last_tick, np.max(self.controls["tick"]))
        if len(self.pedals) > 0:
            last_tick = max(last_tick, np.max(self.pedals["tick"]))
        if len(self.pitch_bends) > 0:
            last_tick = max(last_tick, np.max(self.pitch_bends["tick"]))

        return last_tick

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
    tempo: np.ndarray  # 1D structured numpy array with tempo_dtype elements
    lyrics: list[tuple[int, str]] | None = None

    ticks_per_quarter: int = 480
    time_signature: tuple[int, int] = (4, 4)
    notated_32nd_notes_per_beat: int = 8
    clocks_per_click: int = 24

    def __repr__(self) -> str:
        num_notes = sum(len(track.notes) for track in self.tracks)
        last_tick = max(track.last_tick() for track in self.tracks)
        return (
            f"Score with {len(self.tracks)} tracks, {num_notes} notes, "
            f"last tick {last_tick} and duration {self.duration}"
        )

    def last_tick(self) -> int:
        """Get the last tick of the score."""
        last_tick = max(track.last_tick() for track in self.tracks)
        return last_tick

    def __post_init__(self) -> None:
        assert self.tempo.dtype == tempo_dtype, "Tempo must be a structured numpy array with tempo_dtype elements"
        assert self.tracks is not None, "Tracks must be a list of Track objects"
        # assert len(self.tracks) > 0, "Tracks must be a non-empty list of Track objects"
        assert self.duration > 0, "Duration must be a positive float"
        assert len(self.tempo) > 0, "Tempo must a non-empty"


def group_data(keys: list[np.ndarray], data: Optional[np.ndarray] = None) -> dict[Any, np.ndarray]:
    order = np.lexsort(keys)
    # make sure the keys have the same length
    for key_array in keys:
        assert len(key_array) > 0, "Keys must have at least one element"
        assert len(key_array) == len(keys[0]), "Keys must have the same length"
    is_boundary = np.zeros((len(keys[0]) - 1), dtype=bool)
    for i in range(len(keys)):
        is_boundary |= np.diff(keys[i][order]) != 0
    boundaries = np.nonzero(is_boundary)[0]
    starts = np.concatenate(([0], boundaries + 1))
    ends = np.concatenate((boundaries + 1, [len(keys[0])]))
    output: dict[tuple, np.ndarray] = {}
    for i in range(len(starts)):
        key = tuple([key_array[order[starts[i]]] for key_array in keys])
        if len(keys) == 1:
            key = key[0]
        group = order[starts[i] : ends[i]]
        if data is not None:
            output[key] = data[order[starts[i] : ends[i]]]
        else:
            output[key] = order[starts[i] : ends[i]]
        if len(keys) == 1:
            assert np.all(keys[0][group] == key), f"keys[0][group] {keys[0][group]} != key {key}"
        else:
            for j in range(len(keys)):
                assert np.all(keys[j][group] == key[j])
    return output


def extract_notes_start_stop(note_events: np.ndarray, notes_mode: NotesMode) -> tuple[np.ndarray, np.ndarray]:
    notes_order = np.lexsort(
        (note_events["event_type"], note_events["tick"], note_events["value1"], note_events["channel"])
    )
    sorted_note_events = note_events[notes_order]
    ordered_note_start_ids, ordered_note_stop_ids = extract_notes_start_stop_numba(
        sorted_note_events, notes_mode_mapping[notes_mode]
    )
    if len(ordered_note_start_ids) > 0:
        note_start_ids = notes_order[ordered_note_start_ids]
        note_stop_ids = notes_order[ordered_note_stop_ids]
        # restore order to mach the one in the original midi file
        order = np.argsort(note_start_ids)
        note_start_ids = note_start_ids[order]
        note_stop_ids = note_stop_ids[order]
    else:
        note_start_ids = np.zeros((0,), dtype=np.int32)
        note_stop_ids = np.zeros((0,), dtype=np.int32)

    return note_start_ids, note_stop_ids


def get_pedals_from_controls(channel_controls: np.ndarray) -> np.ndarray:
    pedals_start, pedals_end = get_pedals_from_controls_jit(channel_controls)
    if len(pedals_start) > 0:
        pedals = np.zeros(len(pedals_start), dtype=pedal_dtype)
        pedals["time"] = channel_controls[pedals_start]["time"]
        pedals["tick"] = channel_controls[pedals_start]["tick"]
        pedals["duration"] = channel_controls[pedals_end]["time"] - channel_controls[pedals_end]["time"]
        pedals["duration_tick"] = channel_controls[pedals_end]["tick"] - channel_controls[pedals_end]["tick"]

    else:
        pedals = np.zeros((0,), dtype=pedal_dtype)
    return pedals


def midi_to_score(midi_score: Midi, minimize_tempo: bool = True, notes_mode: NotesMode = "note_off_stops_all") -> Score:
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
    tempo["bpm"] = 60000000 / tempo_events["value1"]

    # remove unnecessary tempo events
    if minimize_tempo:
        tempo = tempo[np.hstack(([True], (np.diff(tempo["bpm"]) != 0)))]

    lyrics = []
    for _, midi_track in enumerate(midi_score.tracks):
        if midi_track.lyrics is not None:
            lyrics.extend(midi_track.lyrics)
    # sort the lyrics by tick
    lyrics = sorted(lyrics, key=lambda x: x[0])

    for midi_track_id, midi_track in enumerate(midi_score.tracks):
        if midi_track.events.size == 0:
            continue
        numerator, denominator = midi_track.time_signature

        clocks_per_click = midi_track.clocks_per_click
        notated_32nd_notes_per_beat = midi_track.notated_32nd_notes_per_beat

        # get the program for each channel
        events_programs = get_events_program(midi_track.events)

        events = midi_track.events
        # compute the tick and time of each event
        events_ticks = events["tick"]
        events_times = get_event_times(events, tempo_events, midi_score.ticks_per_quarter)

        # sort all the events in lexicographic order by channel and tick
        # this allows to have a order for the events that simplifies the code to process them
        events_groups = group_data([events_programs, events["channel"]])
        # sort in lexicographic order by pitch first and then by tick, then even type
        # this allows to have a order for the events that simplifies the
        # extracting matching note starts and stops
        # we sort by inverse of event type in order to deal with the case there is no gap
        # between two consecutive notes
        # extract the event of type note on or note off
        notes_events_ids = np.nonzero((events["event_type"] == 0) | (events["event_type"] == 1))[0]
        if len(notes_events_ids) > 0:
            note_events = events[notes_events_ids]

            note_start_ids, note_stop_ids = extract_notes_start_stop(note_events, notes_mode)
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

        control_change_events_ids = np.nonzero(events["event_type"] == 3)[0]

        control_change_events = events[control_change_events_ids]
        channels_controls = {channel: np.zeros((0,), dtype=control_dtype) for channel in range(16)}
        if len(control_change_events) > 0:
            channels_control_change_events_ids = group_data(
                [control_change_events["channel"]], control_change_events_ids
            )

            for channel, channel_control_change_events_ids in channels_control_change_events_ids.items():
                channel_control_change_events = events[channel_control_change_events_ids]
                controls = np.zeros(len(channel_control_change_events_ids), dtype=control_dtype)
                controls["time"] = events_times[channel_control_change_events_ids]
                controls["tick"] = events_ticks[channel_control_change_events_ids]
                controls["number"] = channel_control_change_events["value1"]
                controls["value"] = channel_control_change_events["value2"]
                channels_controls[channel] = controls

        channels_pedals = {}
        for channel, channel_controls in channels_controls.items():
            if len(channel_controls) > 0:
                pedals = get_pedals_from_controls(channel_controls)
            else:
                pedals = np.zeros((0,), dtype=pedal_dtype)
            channels_pedals[channel] = pedals

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

            pitch_bends_mask = track_events["event_type"] == 2
            pitch_bends_events = track_events[pitch_bends_mask]
            pitch_bends = np.zeros(len(pitch_bends_events), dtype=pitch_bend_dtype)
            pitch_bends["time"] = track_events_times[pitch_bends_mask]
            pitch_bends["tick"] = track_events_ticks[pitch_bends_mask]
            pitch_bends["value"] = pitch_bends_events["value1"]

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
                    controls=channels_controls[track_channel],
                    pedals=channels_pedals[track_channel],
                    pitch_bends=pitch_bends,
                )
                tracks.append(track)

    # sort the tracks by channel and program
    # tracks.sort(key=lambda x: (x.channel, x.program))
    return Score(
        tracks=tracks,
        lyrics=lyrics,
        duration=duration,
        time_signature=(numerator, denominator),
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
    if has_duplicate_values([track.channel for track in score.tracks]):
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
        events["value1"][id_start : id_start + len(tempo)] = 60000000 / tempo["bpm"]
        events["value2"][id_start : id_start + len(tempo)] = 0
        id_start += len(tempo)

        lyrics = score.lyrics
    else:
        # create track with the tempo changes
        tempo_events = np.zeros(len(score.tempo), dtype=event_dtype)
        tempo_events["event_type"] = 5
        tempo_events["channel"] = 0
        tempo_events["value1"] = 60000000 / score.tempo["bpm"]
        tempo_events["value2"] = 0
        tempo_events["tick"] = score.tempo["tick"]
        midi_tracks.append(
            MidiTrack(
                name="tempo",
                lyrics=[],
                events=tempo_events,
                time_signature=score.time_signature,
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
                    time_signature=score.time_signature,
                    clocks_per_click=score.clocks_per_click,
                    notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
                )
            )
        lyrics = []

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

        # TODO check that the pedals are consistent with the controls

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
                time_signature=score.time_signature,
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
            time_signature=score.time_signature,
            clocks_per_click=score.clocks_per_click,
            notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
        )
        midi_tracks = [midi_track]
    midi_score = Midi(tracks=midi_tracks, ticks_per_quarter=score.ticks_per_quarter)
    return midi_score


def load_score(
    file_path: str,
    notes_mode: NotesMode = "note_off_stops_all",
    minimize_tempo: bool = True,
    check_round_trip: bool = False,
) -> Score:
    """Loads a MIDI file and converts it to a Score."""
    midi_raw = load_midi_score(file_path)
    score = midi_to_score(midi_raw, minimize_tempo=minimize_tempo, notes_mode=notes_mode)

    if check_round_trip:
        # check if the two score can be converted back and forth
        midi_raw2 = score_to_midi(score)
        score2 = midi_to_score(midi_raw2, minimize_tempo=minimize_tempo, notes_mode=notes_mode)
        assert_scores_equal(score, score2)

    return score


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
        time_signature=score.time_signature,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
        tempo=score.tempo,
        lyrics=score.lyrics,
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
        time_signature=score.time_signature,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
        tempo=score.tempo,
        lyrics=score.lyrics,
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
        time_signature=score.time_signature,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
        tempo=score.tempo,
        lyrics=score.lyrics,
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
        time_signature=score.time_signature,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
        tempo=score.tempo,
        lyrics=score.lyrics,
    )


def remove_pedals(score: Score) -> Score:
    """Remove the pedals from the score."""
    tracks = []
    for track in score.tracks:
        new_track = copy(track)
        new_track.pedals = np.array([], dtype=track.pedals.dtype)
        tracks.append(new_track)

    return Score(
        tracks=tracks,
        duration=score.duration,
        time_signature=score.time_signature,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
        tempo=score.tempo,
        lyrics=score.lyrics,
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
        time_signature=score.time_signature,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
        tempo=score.tempo,
        lyrics=score.lyrics,
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
        time_signature=score.time_signature,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
        tempo=score.tempo,
        lyrics=score.lyrics,
    )


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


def time_to_tick(time: float, tempo: np.ndarray, ticks_per_quarter: int) -> int:
    """Convert a time in seconds to tick."""
    # get the tempo at the start of the time range
    tempo_idx: int = 0
    if tempo.size > 1:
        tempo_idx = int(np.searchsorted(tempo["time"], time, side="right") - 1)

    tempo_idx = np.clip(tempo_idx, 0, tempo.size - 1)
    bpm = tempo["bpm"][tempo_idx]
    ref_ticks = tempo["tick"][tempo_idx]
    ref_time = tempo["time"][tempo_idx]
    quarter_per_second = bpm / 60.0
    ticks_per_second = ticks_per_quarter * quarter_per_second
    return int(ref_ticks + (time - ref_time) * ticks_per_second)


def times_to_ticks(time: np.ndarray, tempo: np.ndarray, ticks_per_quarter: int) -> np.ndarray:
    """Convert a time in seconds to ticks."""
    # get the tempo at the start of the time range
    if tempo.size > 1:
        tempo_idx = np.searchsorted(tempo["time"], time, side="right") - 1
    else:
        tempo_idx = np.zeros(len(time), dtype=np.int32)
    tempo_idx = np.clip(tempo_idx, 0, tempo.size - 1)
    bpm = tempo["bpm"][tempo_idx]
    ref_ticks = tempo["tick"][tempo_idx]
    ref_time = tempo["time"][tempo_idx]
    quarter_per_second = bpm / 60.0
    ticks_per_second = ticks_per_quarter * quarter_per_second
    return (ref_ticks + (time - ref_time) * ticks_per_second).astype(np.int32)


def update_ticks(score: Score, tempo: np.ndarray) -> Score:
    """Update the ticks of the score according to the tempo."""
    tracks = []
    for track in score.tracks:
        new_track = copy(track)
        new_track.notes["start_tick"] = times_to_ticks(new_track.notes["start"], tempo, score.ticks_per_quarter)
        new_track.notes["duration_tick"] = (
            times_to_ticks(new_track.notes["start"] + new_track.notes["duration"], tempo, score.ticks_per_quarter)
            - new_track.notes["start_tick"]
        )
        new_track.pedals["tick"] = times_to_ticks(new_track.pedals["time"], tempo, score.ticks_per_quarter)
        new_track.controls["tick"] = times_to_ticks(new_track.controls["time"], tempo, score.ticks_per_quarter)
        new_track.pitch_bends["tick"] = times_to_ticks(new_track.pitch_bends["time"], tempo, score.ticks_per_quarter)
        tracks.append(new_track)

    return Score(
        tracks=tracks,
        duration=score.duration,
        time_signature=score.time_signature,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
        tempo=tempo,
    )


def crop_score(score: Score, start: float, duration: float) -> Score:
    """Crop a MIDI score to a specific time range.

    Note: the NoteOn events from before the start time are not kept
    and thus the sound may not be the same as the cropped original sound.
    """
    end = start + duration
    tracks = []

    previous_tempos = np.nonzero(score.tempo["time"] < start)[0]
    tempo_keep = (score.tempo["time"] < end) & (score.tempo["time"] >= start)
    if len(previous_tempos) > 0:
        tempo_keep[previous_tempos[-1]] = True
    new_tempo = score.tempo[tempo_keep]
    new_tempo["time"] = np.maximum(new_tempo["time"] - start, 0)
    tick_end = time_to_tick(end, score.tempo, score.ticks_per_quarter)
    tick_start = time_to_tick(start, score.tempo, score.ticks_per_quarter)
    new_tempo["tick"] = np.maximum(new_tempo["tick"] - tick_start, 0)

    for track in score.tracks:
        notes = track.notes
        notes_end = notes["start"] + notes["duration"]
        notes_end_tick = notes["start_tick"] + notes["duration_tick"]
        notes_keep = (notes["start"] < end) & (notes_end > start)
        new_notes = notes[notes_keep]
        if len(new_notes) == 0:
            continue
        new_notes["start"] = np.maximum(new_notes["start"] - start, 0)
        new_notes_end = np.minimum(notes_end[notes_keep] - start, end - start)
        new_notes["duration"] = new_notes_end - new_notes["start"]
        new_notes["start_tick"] = np.maximum(new_notes["start_tick"] - tick_start, 0)
        new_notes_end_tick = np.minimum(notes_end_tick[notes_keep] - tick_start, tick_end - tick_start)
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

        new_track = Track(
            channel=track.channel,
            program=track.program,
            is_drum=track.is_drum,
            name=track.name,
            notes=new_notes,
            controls=new_controls,
            pedals=new_pedals,
            pitch_bends=new_pitch_bends,
            midi_track_id=track.midi_track_id,
        )
        tracks.append(new_track)
    return Score(
        tracks=tracks,
        duration=duration,
        time_signature=score.time_signature,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
        tempo=new_tempo,
    )


def select_tracks(score: Score, track_ids: list[int]) -> Score:
    """Select only the tracks with the specified programs."""
    tracks = [score.tracks[track_id] for track_id in track_ids]
    return Score(
        tracks=tracks,
        duration=score.duration,
        time_signature=score.time_signature,
        clocks_per_click=score.clocks_per_click,
        ticks_per_quarter=score.ticks_per_quarter,
        notated_32nd_notes_per_beat=score.notated_32nd_notes_per_beat,
        tempo=score.tempo,
        lyrics=score.lyrics,
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
    time_tol: float = 1e-3,
    value_tol: float = 1e-2,
    tick_tol: int = 0,
    compare_channels: bool = True,
) -> None:
    assert len(score1.tracks) == len(score2.tracks), "The scores have different number of tracks"
    max_diff = 0
    tracks_1 = score1.tracks
    tracks_2 = score2.tracks

    # sort by nme, program and then by number of notes to try to have the same order
    if compare_channels:
        tracks_1 = sorted(tracks_1, key=lambda x: (x.name, x.program, x.channel, len(x.notes), x.notes["pitch"].sum()))
        tracks_2 = sorted(tracks_2, key=lambda x: (x.name, x.program, x.channel, len(x.notes), x.notes["pitch"].sum()))
    else:
        tracks_1 = sorted(tracks_1, key=lambda x: (x.name, x.program, len(x.notes), x.notes["pitch"].sum()))
        tracks_2 = sorted(tracks_2, key=lambda x: (x.name, x.program, len(x.notes), x.notes["pitch"].sum()))
    assert score1.tempo.shape == score2.tempo.shape, "Different number of tempo events"
    assert np.all(score1.tempo["tick"] == score2.tempo["tick"]), "Different tick values for tempo events"

    assert np.allclose(score1.tempo["bpm"], score2.tempo["bpm"], atol=1e-3), "Different bpm values for tempo events"
    assert np.allclose(score1.tempo["time"], score2.tempo["time"], atol=1e-3), "Different time values for tempo events"
    for track_id, (track1, track2) in enumerate(zip(tracks_1, tracks_2)):
        assert track1.name == track2.name, "Track names are different"
        assert track1.program == track2.program, "Track programs are different"
        if compare_channels:
            assert track1.channel == track2.channel, "Track channels are different"
        # sort not by pitch then tick
        # notes1= track1.notes
        # notes2= track2.notes
        order1 = np.lexsort((np.arange(len(track1.notes)), track1.notes["start_tick"], track1.notes["pitch"]))
        notes1 = track1.notes[order1]
        order2 = np.lexsort((np.arange(len(track2.notes)), track2.notes["start_tick"], track2.notes["pitch"]))
        notes2 = track2.notes[order2]

        min_len = min(len(notes1), len(notes2))
        np.nonzero(notes1[:min_len]["start_tick"] != notes2[:min_len]["start_tick"])
        assert len(notes1) == len(notes2), f"Different number of notes in track {track_id}"
        assert np.all(notes1["pitch"] == notes2["pitch"]), f"Pitches are different in track {track_id}"
        max_tick_diff = max(abs(notes1["start_tick"] - notes2["start_tick"]))
        assert max_tick_diff <= tick_tol, f"Tick difference larger than {tick_tol} in track {track_id}"
        max_duration_tick_diff = max(abs(notes1["duration_tick"] - notes2["duration_tick"]))
        assert max_duration_tick_diff <= tick_tol, (
            f"Duration tick difference {max_duration_tick_diff} greater than {tick_tol} in track {track_id}"
        )
        # max note time difference
        max_start_time_diff = max(abs(notes1["start"] - notes2["start"]))
        assert max_start_time_diff <= time_tol, (
            f"Max note start time difference {max_start_time_diff}>{time_tol} in track {track_id}"
        )
        notes_stop_1 = notes1["start"] + notes1["duration"]
        notes_stop_2 = notes2["start"] + notes2["duration"]
        max_stop_diff = max(abs(notes_stop_1 - notes_stop_2))
        assert max_stop_diff <= time_tol, f"Max note end difference {max_stop_diff}>{time_tol} in track {track_id}"
        # max note velocity difference
        velocify_abs_diff = abs(notes1["velocity_on"].astype(np.int16) - notes2["velocity_on"].astype(np.int16))
        max_velocity_diff = max(velocify_abs_diff)
        assert max_velocity_diff <= value_tol, (
            f"Max note velocity difference {max_velocity_diff}>{value_tol} in track {track_id}"
        )
        # max note duration difference
        max_duration_diff = max(abs(notes1["duration"] - notes2["duration"]))
        assert max_duration_diff <= time_tol, (
            f"Max note duration difference {max_duration_diff}>{time_tol} in track {track_id}"
        )

        # max control time difference
        assert track1.controls.shape == track2.controls.shape, f"Different number of control events in track {track_id}"
        if track1.controls.size > 0:
            max_control_time_diff = max(abs(track1.controls["time"] - track2.controls["time"]))
            assert max_control_time_diff <= time_tol, (
                f"Max control time difference {max_control_time_diff}>{time_tol} in track {track_id}"
            )
            max_diff = max(max_diff, max_control_time_diff)
        # max pedal time difference
        assert track1.pedals.shape == track2.pedals.shape, f"Different number of pedal events in track {track_id}"
        if track1.pedals.size > 0:
            max_pedal_time_diff = max(abs(track1.pedals["time"] - track2.pedals["time"]))
            max_diff = max(max_diff, max_pedal_time_diff)
            assert max_pedal_time_diff <= time_tol, (
                f"Max pedal time difference {max_pedal_time_diff}>{time_tol} in track {track_id}"
            )
        # max pitch bend time difference
        assert track1.pitch_bends.shape == track2.pitch_bends.shape, (
            f"Different number of pitch bend events in track {track_id}"
        )
        if track1.pitch_bends.size > 0:
            max_pitch_bend_time_diff = max(abs(track1.pitch_bends["time"] - track2.pitch_bends["time"]))
            max_diff = max(max_diff, max_pitch_bend_time_diff)
            assert max_pitch_bend_time_diff <= time_tol, (
                f"Max pitch bend time difference {max_pitch_bend_time_diff}>{time_tol} in track {track_id}"
            )


def get_score_instrumnets(score: Score) -> list[str]:
    """Get the instruments from a score."""
    instruments = set()
    for track in score.tracks:
        instrument_name = program_to_instrument[track.program]
        instruments.add(instrument_name)
    return list(instruments)


def get_score_instrument_groups(score: Score) -> list[str]:
    """Get the instrument groups from a score."""
    instrument_groups = set()
    for track in score.tracks:
        instrument_group_name = program_to_instrument_group[track.program]
        instrument_groups.add(instrument_group_name)
    return list(instrument_groups)


def get_num_notes_per_group(score: Score) -> dict[str, int]:
    """Get the number of notes per instrument group."""
    num_notes_per_group = {}
    for track in score.tracks:
        instrument_group_name = program_to_instrument_group[track.program]
        if instrument_group_name not in num_notes_per_group:
            num_notes_per_group[instrument_group_name] = 0
        num_notes_per_group[instrument_group_name] += len(track.notes)
    return num_notes_per_group

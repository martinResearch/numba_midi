"""Music score representtion based on structured numpy arrays."""

from copy import copy
from dataclasses import dataclass

from numba import njit
import numpy as np

from numba_midi.instruments import instrument_to_program
from numba_midi.midi import get_even_ticks_and_times, load_midi_raw, Midi

note_dtype = np.dtype(
    [
        ("start", np.float32),
        ("duration", np.float32),
        ("pitch", np.int32),
        ("velocity_on", np.uint8),
    ]
)

control_dtype = np.dtype([("time", np.float32), ("number", np.int32), ("value", np.int32)])
pedal_dtype = np.dtype([("time", np.float32), ("duration", np.float32)])
pitch_bend_dtype = np.dtype([("time", np.float32), ("value", np.int32)])


@dataclass
class Track:
    """MIDI track representation."""

    channel: int
    program: int
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


@dataclass
class Score:
    """MIDI score representation."""

    tracks: list[Track]
    duration: float

    def __repr__(self) -> str:
        return f"Score with {len(self.tracks)} tracks and duration {self.duration}"


def midi_to_score(midi_score: Midi) -> Score:
    """Convert a MidiScore to a Score.

    Convert from event-based representation notes with durations
    """
    tracks = []
    duration = 0.0
    for midi_track in midi_score.tracks:
        program = 0

        # get the program for each channel
        program_change = midi_track.events[midi_track.events["event_type"] == 4]
        channel_to_program = np.zeros((16), dtype=np.int32)
        channel_to_program[program_change["channel"]] = program_change["value1"]
        assert np.all(channel_to_program[program_change["channel"]] == program_change["value1"]), (
            "Multiple program changes for the same channel"
        )

        events = midi_track.events
        # compute the tick and time of each event
        events_ticks, events_times = get_even_ticks_and_times(events, midi_track.ticks_per_quarter)

        # sort all the events in lexicographic order by channel and tick
        # this allows to have a order for the events that simplifies the code to process them
        events_order = np.lexsort((events_ticks, events["channel"]))
        events = events[events_order]
        events_times = events_times[events_order]
        channel_change = np.nonzero(np.diff(events["channel"]))[0] + 1
        channel_starts = np.concatenate(([0], channel_change))
        channel_ends = np.concatenate((channel_change, [len(events)]))
        channels = events["channel"][channel_starts]
        for channel, channel_start, channel_end in zip(channels, channel_starts, channel_ends, strict=True):
            # extract the events for the channel
            channel_events = events[channel_start:channel_end]
            channel_events_times = events_times[channel_start:channel_end]

            #  control change events
            control_change_mask = channel_events["event_type"] == 3
            control_change_events = channel_events[control_change_mask]
            controls = np.zeros(len(control_change_events), dtype=control_dtype)
            controls["time"] = channel_events_times[control_change_mask]
            controls["number"] = control_change_events["value1"]
            controls["value"] = control_change_events["value2"]

            pitch_bends_mask = channel_events["event_type"] == 2
            pitch_bends_events = channel_events[pitch_bends_mask]
            pitch_bends = np.zeros(len(pitch_bends_events), dtype=pitch_bend_dtype)
            pitch_bends["time"] = channel_events_times[pitch_bends_mask]
            pitch_bends["value"] = pitch_bends_events["value1"]

            # extract the event of type note on or note off
            notes_mask = (channel_events["event_type"] == 0) | (channel_events["event_type"] == 1)
            note_events = channel_events[notes_mask]
            notes_times = channel_events_times[notes_mask]

            # sort in lexicographic order by pitch first and then by tick
            # this allows to have a order for the events that simplifies the
            # extracting matching note starts and stops
            notes_order = np.lexsort((notes_times, note_events["value1"]))
            sorted_note_events = note_events[notes_order]

            note_starts = sorted_note_events[::2]
            note_stops = sorted_note_events[1::2]
            assert np.all(note_starts["event_type"] == 0)
            assert np.all((note_stops["event_type"] == 1) | (note_stops["value2"] == 0))
            assert np.all(note_stops["value1"] == note_starts["value1"])

            notes_np = np.zeros(len(note_starts), dtype=note_dtype)
            notes_times_ordered = notes_times[notes_order]
            notes_np["start"] = notes_times_ordered[::2]
            notes_np["duration"] = notes_times_ordered[1::2] - notes_times_ordered[::2]
            notes_np["pitch"] = note_starts["value1"]
            notes_np["velocity_on"] = note_starts["value2"]
            duration = max(duration, np.max(notes_np["start"] + notes_np["duration"]))

            program = channel_to_program[channel]
            notes_np = notes_np[np.argsort(notes_np["start"])]
            track = Track(
                channel=int(channel),
                program=int(program),
                is_drum=False,  # FIXME
                name=midi_track.name,
                notes=notes_np,
                controls=controls,
                pedals=np.zeros((0,), dtype=pedal_dtype),  # FIXME not supported yet
                pitch_bends=pitch_bends,
            )
            tracks.append(track)

    return Score(tracks=tracks, duration=duration)


def load_score(file_path: str) -> Score:
    """Loads a MIDI file and converts it to a Score."""
    midi_raw = load_midi_raw(file_path)
    return midi_to_score(midi_raw)


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

    new_score = Score(tracks=tracks, duration=score.duration)
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
    return Score(tracks=tracks, duration=score.duration)


def remove_empty_tracks(score: Score) -> Score:
    """Remove the tracks of the score that have no notes."""
    tracks = []
    for track in score.tracks:
        if track.notes.size > 0:
            tracks.append(track)
    return Score(tracks=tracks, duration=score.duration)


def remove_pitch_bends(score: Score) -> Score:
    """Remove the pitch bends from the score."""
    tracks = []
    for track in score.tracks:
        new_track = copy(track)
        new_track.pitch_bends = np.array([], dtype=track.pitch_bends.dtype)
        tracks.append(new_track)

    return Score(tracks=tracks, duration=score.duration)


def remove_control_changes(score: Score) -> Score:
    """Remove the control changes from the score."""
    tracks = []
    for track in score.tracks:
        new_track = copy(track)
        new_track.controls = np.array([], dtype=track.controls.dtype)
        tracks.append(new_track)

    return Score(tracks=tracks, duration=score.duration)


def filter_pitch(score: Score, pitch_min: int, pitch_max: int) -> Score:
    tracks = []
    for track in score.tracks:
        new_track = copy(track)
        keep = (track.notes["pitch"] >= pitch_min) & (track.notes["pitch"] < pitch_max)
        new_track.notes = track.notes[keep]
        tracks.append(new_track)
    return Score(tracks=tracks, duration=score.duration)


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


def check_no_overlapping_notes(notes: np.ndarray) -> None:
    """Check that there are no overlapping notes at the same pitch."""
    if len(get_overlapping_notes(notes)) > 0:
        raise ValueError("Overlapping notes found")


def check_no_overlapping_notes_in_score(score: Score) -> None:
    for track in score.tracks:
        check_no_overlapping_notes(track.notes)


def crop_score(score: Score, start: float, duration: float) -> Score:
    """Crop a MIDI score to a specific time range.

    Note: the NoteOn events from before the start time are not kept
    and thus the sound may not be the same as the cropped original sound.
    """
    end = start + duration
    tracks = []
    for track in score.tracks:
        notes = track.notes
        notes_end = notes["start"] + notes["duration"]
        notes_keep = (notes["start"] < end) & (notes_end > start)
        new_notes = notes[notes_keep]
        if len(new_notes) == 0:
            continue
        new_notes["start"] = np.maximum(new_notes["start"] - start, 0)
        new_notes_end = np.minimum(notes_end[notes_keep] - start, end - start)
        new_notes["duration"] = new_notes_end - new_notes["start"]

        assert np.all(new_notes_end <= end - start), "Note end time exceeds score duration"

        check_no_overlapping_notes(new_notes)

        pedals_end = track.pedals["time"] + track.pedals["duration"]
        pedals_keep = (track.pedals["time"] < end) & (pedals_end > start)
        new_pedals = track.pedals[pedals_keep]
        new_pedals_end = np.minimum(pedals_end[pedals_keep], end) - start
        new_pedals["duration"] = new_pedals_end - new_pedals["time"]

        controls_keep = (track.controls["time"] < end) & (track.controls["time"] >= start)
        new_controls = track.controls[controls_keep]
        new_controls["time"] = np.maximum(new_controls["time"] - start, 0)

        pitch_bends_keep = (track.pitch_bends["time"] < end) & (track.pitch_bends["time"] >= start)
        new_pitch_bends = track.pitch_bends[pitch_bends_keep]
        new_pitch_bends["time"] = np.maximum(new_pitch_bends["time"] - start, 0)

        new_track = Track(
            program=track.program,
            is_drum=track.is_drum,
            name=track.name,
            notes=new_notes,
            controls=new_controls,
            pedals=new_pedals,
            pitch_bends=new_pitch_bends,
        )
        tracks.append(new_track)
    return Score(tracks=tracks, duration=duration)


def select_tracks(score: Score, track_ids: list[int]) -> Score:
    """Select only the tracks with the specified programs."""
    tracks = [score.tracks[track_id] for track_id in track_ids]
    return Score(tracks=tracks, duration=score.duration)


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


def assert_isequal(
    score1: Score,
    score2: Score,
    sort_tracks_with_programs: bool = False,
    time_tol: float = 1e-3,
    value_tol: float = 1e-2,
) -> None:
    assert len(score1.tracks) == len(score2.tracks), "The scores have different number of tracks"
    max_diff = 0
    tracks_1 = score1.tracks
    tracks_2 = score2.tracks
    if sort_tracks_with_programs:
        tracks_1 = sorted(tracks_1, key=lambda x: x.program)
        tracks_2 = sorted(tracks_2, key=lambda x: x.program)
    for track1, track2 in zip(tracks_1, tracks_2):
        # group the notes by pitch
        notes1: dict[int, list[np.ndarray]] = {}
        for note in track1.notes:
            pitch = note["pitch"]
            if pitch not in notes1:
                notes1[pitch] = []
            notes1[pitch].append(note)
        notes2: dict[int, list[np.ndarray]] = {}
        for note in track2.notes:
            pitch = note["pitch"]
            if pitch not in notes2:
                notes2[pitch] = []
            notes2[pitch].append(note)
        # make sure same pitches are used
        assert set(notes1.keys()) == set(notes2.keys()), "Different pitches are used"
        for pitch, _ in notes1.items():
            # sort notes by start time
            notes1_pitch = np.sort(np.array(notes1[pitch]), order="start")
            notes2_pitch = np.sort(np.array(notes2[pitch]), order="start")

            # max note time difference
            max_start_time_diff = max(abs(notes1_pitch["start"] - notes2_pitch["start"]))
            assert max_start_time_diff <= time_tol, f"Max note start time difference {max_start_time_diff}>{time_tol}"
            notes_stop_1 = notes1_pitch["start"] + notes1_pitch["duration"]
            notes_stop_2 = notes2_pitch["start"] + notes2_pitch["duration"]
            max_stop_diff = max(abs(notes_stop_1 - notes_stop_2))
            assert max_stop_diff <= time_tol, f"Max note end difference {max_stop_diff}>{time_tol}"

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

"""Defining groups of MIDI instruments."""

from enum import Enum

all_instruments = [
    "Acoustic Grand Piano",
    "Bright Acoustic Piano",
    "Electric Grand Piano",
    "Honky-tonk Piano",
    "Electric Piano 1",
    "Electric Piano 2",
    "Harpsichord",
    "Clavinet",
    "Celesta",
    "Glockenspiel",
    "Music Box",
    "Vibraphone",
    "Marimba",
    "Xylophone",
    "Tubular Bells",
    "Dulcimer",
    "Drawbar Organ",
    "Percussive Organ",
    "Rock Organ",
    "Church Organ",
    "Reed Organ",
    "Accordion",
    "Harmonica",
    "Tango Accordion",
    "Acoustic Guitar (nylon)",
    "Acoustic Guitar (steel)",
    "Electric Guitar (jazz)",
    "Electric Guitar (clean)",
    "Electric Guitar (muted)",
    "Overdriven Guitar",
    "Distortion Guitar",
    "Guitar Harmonics",
    "Acoustic Bass",
    "Electric Bass (finger)",
    "Electric Bass (pick)",
    "Fretless Bass",
    "Slap Bass 1",
    "Slap Bass 2",
    "Synth Bass 1",
    "Synth Bass 2",
    "Violin",
    "Viola",
    "Cello",
    "Contrabass",
    "Tremolo Strings",
    "Pizzicato Strings",
    "Orchestral Harp",
    "Timpani",
    "String Ensemble 1",
    "String Ensemble 2",
    "Synth Strings 1",
    "Synth Strings 2",
    "Choir Aahs",
    "Voice Oohs",
    "Synth Choir",
    "Orchestra Hit",
    "Trumpet",
    "Trombone",
    "Tuba",
    "Muted Trumpet",
    "French Horn",
    "Brass Section",
    "Synth Brass 1",
    "Synth Brass 2",
    "Soprano Sax",
    "Alto Sax",
    "Tenor Sax",
    "Baritone Sax",
    "Oboe",
    "English Horn",
    "Bassoon",
    "Clarinet",
    "Piccolo",
    "Flute",
    "Recorder",
    "Pan Flute",
    "Blown bottle",
    "Shakuhachi",
    "Whistle",
    "Ocarina",
    "Lead 1 (square)",
    "Lead 2 (sawtooth)",
    "Lead 3 (calliope)",
    "Lead 4 chiff",
    "Lead 5 (charang)",
    "Lead 6 (voice)",
    "Lead 7 (fifths)",
    "Lead 8 (bass + lead)",
    "Pad 1 (new age)",
    "Pad 2 (warm)",
    "Pad 3 (polysynth)",
    "Pad 4 (choir)",
    "Pad 5 (bowed)",
    "Pad 6 (metallic)",
    "Pad 7 (halo)",
    "Pad 8 (sweep)",
    "FX 1 (rain)",
    "FX 2 (soundtrack)",
    "FX 3 (crystal)",
    "FX 4 (atmosphere)",
    "FX 5 (brightness)",
    "FX 6 (goblins)",
    "FX 7 (echoes)",
    "FX 8 (sci-fi)",
    "Sitar",
    "Banjo",
    "Shamisen",
    "Koto",
    "Kalimba",
    "Bagpipe",
    "Fiddle",
    "Shanai",
    "Tinkle Bell",
    "Agogo",
    "Steel Drums",
    "Woodblock",
    "Taiko Drum",
    "Melodic Tom",
    "Synth Drum",
    "Reverse Cymbal",
    "Guitar Fret Noise",
    "Breath Noise",
    "Seashore",
    "Bird Tweet",
    "Telephone Ring",
    "Helicopter",
    "Applause",
    "Gunshot",
]


class Instrument(Enum):
    """MIDI instruments."""

    Acoustic_Grand_Piano = 0
    Bright_Acoustic_Piano = 1
    Electric_Grand_Piano = 2
    Honky_tonk_Piano = 3
    Electric_Piano_1 = 4
    Electric_Piano_2 = 5
    Harpsichord = 6
    Clavinet = 7
    Celesta = 8
    Glockenspiel = 9
    Music_Box = 10
    Vibraphone = 11
    Marimba = 12
    Xylophone = 13
    Tubular_Bells = 14
    Dulcimer = 15
    Drawbar_Organ = 16
    Percussive_Organ = 17
    Rock_Organ = 18
    Church_Organ = 19
    Reed_Organ = 20
    Accordion = 21
    Harmonica = 22
    Tango_Accordion = 23
    Acoustic_Guitar_nylon = 24
    Acoustic_Guitar_steel = 25
    Electric_Guitar_jazz = 26
    Electric_Guitar_clean = 27
    Electric_Guitar_muted = 28
    Overdriven_Guitar = 29
    Distortion_Guitar = 30
    Guitar_Harmonics = 31
    Acoustic_Bass = 32
    Electric_Bass_finger = 33
    Electric_Bass_pick = 34
    Fretless_Bass = 35
    Slap_Bass_1 = 36
    Slap_Bass_2 = 37
    Synth_Bass_1 = 38
    Synth_Bass_2 = 39
    Violin = 40
    Viola = 41
    Cello = 42
    Contrabass = 43
    Tremolo_Strings = 44
    Pizzicato_Strings = 45
    Orchestral_Harp = 46
    Timpani = 47
    String_Ensemble_1 = 48
    String_Ensemble_2 = 49
    Synth_Strings_1 = 50
    Synth_Strings_2 = 51
    Choir_Aahs = 52
    Voice_Oohs = 53
    Synth_Choir = 54
    Orchestra_Hit = 55
    Trumpet = 56
    Trombone = 57
    Tuba = 58
    Muted_Trumpet = 59
    French_Horn = 60
    Brass_Section = 61
    Synth_Brass_1 = 62
    Synth_Brass_2 = 63
    Soprano_Sax = 64
    Alto_Sax = 65
    Tenor_Sax = 66
    Baritone_Sax = 67
    Oboe = 68
    English_Horn = 69
    Bassoon = 70
    Clarinet = 71
    Piccolo = 72
    Flute = 73
    Recorder = 74
    Pan_Flute = 75
    Blown_Bottle = 76
    Shakuhachi = 77
    Whistle = 78
    Ocarina = 79
    Lead_1_square = 80
    Lead_2_sawtooth = 81
    Lead_3_calliope = 82
    Lead_4_chiff = 83
    Lead_5_charang = 84
    Lead_6_voice = 85
    Lead_7_fifths = 86
    Lead_8_bass_lead = 87
    Pad_1_new_age = 88
    Pad_2_warm = 89
    Pad_3_polysynth = 90
    Pad_4_choir = 91
    Pad_5_bowed = 92
    Pad_6_metallic = 93
    Pad_7_halo = 94
    Pad_8_sweep = 95
    FX_1_rain = 96
    FX_2_soundtrack = 97
    FX_3_crystal = 98
    FX_4_atmosphere = 99
    FX_5_brightness = 100
    FX_6_goblins = 101
    FX_7_echoes = 102
    FX_8_sci_fi = 103
    Sitar = 104
    Banjo = 105
    Shamisen = 106
    Koto = 107
    Kalimba = 108
    Bagpipe = 109
    Fiddle = 110
    Shanai = 111
    Tinkle_Bell = 112
    Agogo = 113
    Steel_Drums = 114
    Woodblock = 115
    Taiko_Drum = 116
    Melodic_Tom = 117
    Synth_Drum = 118
    Reverse_Cymbal = 119
    Guitar_Fret_Noise = 120
    Breath_Noise = 121
    Seashore = 122
    Bird_Tweet = 123
    Telephone_Ring = 124
    Helicopter = 125
    Applause = 126
    Gunshot = 127


instrument_to_program = {instrument: program for program, instrument in enumerate(all_instruments)}


midi_instruments_groups = {
    "Piano": [
        "Acoustic Grand Piano",
        "Bright Acoustic Piano",
        "Electric Grand Piano",
        "Honky-tonk Piano",
        "Electric Piano 1",
        "Electric Piano 2",
        "Harpsichord",
        "Clavinet",
    ],
    "Chromatic Percussion": [
        "Celesta",
        "Glockenspiel",
        "Music Box",
        "Vibraphone",
        "Marimba",
        "Xylophone",
        "Tubular Bells",
        "Dulcimer",
    ],
    "Organ": [
        "Drawbar Organ",
        "Percussive Organ",
        "Rock Organ",
        "Church Organ",
        "Reed Organ",
        "Accordion",
        "Harmonica",
        "Tango Accordion",
    ],
    "Guitar": [
        "Acoustic Guitar (nylon)",
        "Acoustic Guitar (steel)",
        "Electric Guitar (jazz)",
        "Electric Guitar (clean)",
        "Electric Guitar (muted)",
        "Overdriven Guitar",
        "Distortion Guitar",
        "Guitar Harmonics",
    ],
    "Bass": [
        "Acoustic Bass",
        "Electric Bass (finger)",
        "Electric Bass (pick)",
        "Fretless Bass",
        "Slap Bass 1",
        "Slap Bass 2",
        "Synth Bass 1",
        "Synth Bass 2",
    ],
    "Strings": [
        "Violin",
        "Viola",
        "Cello",
        "Contrabass",
        "Tremolo Strings",
        "Pizzicato Strings",
        "Orchestral Harp",
        "Timpani",
        "Synth Strings 1",
        "Synth Strings 2",
    ],
    "Ensemble": [
        "String Ensemble 1",
        "String Ensemble 2",
        "SynthStrings 1",
        "SynthStrings 2",
        "Choir Aahs",
        "Voice Oohs",
        "Synth Choir",
        "Orchestra Hit",
    ],
    "Brass": [
        "Trumpet",
        "Trombone",
        "Tuba",
        "Muted Trumpet",
        "French Horn",
        "Brass Section",
        "Synth Brass 1",
        "Synth Brass 2",
    ],
    "Reed": [
        "Soprano Sax",
        "Alto Sax",
        "Tenor Sax",
        "Baritone Sax",
        "Oboe",
        "English Horn",
        "Bassoon",
        "Clarinet",
    ],
    "Pipe": [
        "Piccolo",
        "Flute",
        "Recorder",
        "Pan Flute",
        "Blown Bottle",
        "Shakuhachi",
        "Whistle",
        "Ocarina",
    ],
    "Synth Lead": [
        "Lead 1 (square)",
        "Lead 2 (sawtooth)",
        "Lead 3 (calliope)",
        "Lead 4 (chiff)",
        "Lead 5 (charang)",
        "Lead 6 (voice)",
        "Lead 7 (fifths)",
        "Lead 8 (bass + lead)",
    ],
    "Synth Pad": [
        "Pad 1 (new age)",
        "Pad 2 (warm)",
        "Pad 3 (polysynth)",
        "Pad 4 (choir)",
        "Pad 5 (bowed)",
        "Pad 6 (metallic)",
        "Pad 7 (halo)",
        "Pad 8 (sweep)",
    ],
    "Synth Effects": [
        "FX 1 (rain)",
        "FX 2 (soundtrack)",
        "FX 3 (crystal)",
        "FX 4 (atmosphere)",
        "FX 5 (brightness)",
        "FX 6 (goblins)",
        "FX 7 (echoes)",
        "FX 8 (sci-fi)",
    ],
    "Ethnic": [
        "Sitar",
        "Banjo",
        "Shamisen",
        "Koto",
        "Kalimba",
        "Bagpipe",
        "Fiddle",
        "Shanai",
    ],
    "Percussive": [
        "Tinkle Bell",
        "Agogo",
        "Steel Drums",
        "Woodblock",
        "Taiko Drum",
        "Melodic Tom",
        "Synth Drum",
        "Reverse Cymbal",
    ],
    "Sound Effects": [
        "Guitar Fret Noise",
        "Breath Noise",
        "Seashore",
        "Bird Tweet",
        "Telephone Ring",
        "Helicopter",
        "Applause",
        "Gunshot",
    ],
}

midi_instruments_group_names = list(midi_instruments_groups.keys())

program_to_instrument = {program: instrument for instrument, program in instrument_to_program.items()}

instrument_to_group = {}
for group, instruments in midi_instruments_groups.items():
    for instrument in instruments:
        instrument_to_group[instrument] = group

program_to_instrument_group = {}
for program, instrument in enumerate(all_instruments):
    group = instrument_to_group[instrument] if instrument in instrument_to_group else "Unknown"
    program_to_instrument_group[program] = group

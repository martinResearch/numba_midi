"""Control names for MIDI."""

from enum import Enum


class Control(Enum):
    """Control names."""

    bank_select = 0
    modulation_wheel = 1
    breath_controller = 2
    foot_controller = 4
    portamento_time = 5
    data_entry = 6
    volume = 7
    balance = 8
    pan = 10
    expression_controller = 11
    effect_control_1 = 12
    effect_control_2 = 13
    general_purpose_controller_1 = 16
    general_purpose_controller_2 = 17
    general_purpose_controller_3 = 18
    general_purpose_controller_4 = 19
    sustain_pedal = 64
    portamento = 65
    sostenuto = 66
    soft_pedal = 67
    legato_foot_switch = 68
    hold_2 = 69
    sound_controller_1 = 70
    sound_controller_2 = 71
    sound_controller_3 = 72
    sound_controller_4 = 73
    sound_controller_5 = 74
    sound_controller_6 = 75
    sound_controller_7 = 76
    sound_controller_8 = 77
    sound_controller_9 = 78
    sound_controller_10 = 79
    general_purpose_controller_5 = 80
    general_purpose_controller_6 = 81
    general_purpose_controller_7 = 82
    general_purpose_controller_8 = 83
    portamento_control = 84
    effects_1_depth = 91
    effects_2_depth = 92
    effects_3_depth = 93
    effects_4_depth = 94
    effects_5_depth = 95
    data_increment = 96
    data_decrement = 97
    nrpn_lsb = 98
    nrpn_msb = 99
    rpn_lsb = 100
    rpn_msb = 101
    all_sound_off = 120
    reset_all_controllers = 121
    local_control = 122
    all_notes_off = 123
    omni_off = 124
    omni_on = 125
    mono_on = 126
    poly_on = 127

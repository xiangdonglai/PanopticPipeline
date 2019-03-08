CONFIG = {
    # Config for calibration

    # whether to run calibration first
    'run_calibration': False,
    # the dome sequence base name, e.g. '19XXXX_tent'
    'input_seq_name': '190301_tent',
    # indexes of dome sequences, e.g. 3 means to use '19XXXX_tent3'
    'input_seq_index': [3, 4, 8],
    # output data to save, e.g. '19XXXX'
    'output_date': '190303',
    # Nas Index e.g. '12a' for /media/posefs12a
    'input_nas_index': '12a',
    # local path for temporarily storing calib data, will mkdir 'input_nas_idx'/'output_date'
    'calib_root_path': '/home/donglaix/',

    # Config for Sequences

    # sequences names
    'sequence_names': ['190215_uthand2', '190301_utcatch3'],
    # calibration data
    'calibration_data': ['190211_calib_norm', '190301_calib_norm'],
    # start idx
    'start_index': [100, 100],
    # end idx
    'end_index': [102, 102],
    # number of camera
    'camera_number': [140, 140],
    # Captures nas
    'captures_nas': ['12a', '12a'],
    # Processed nas
    'processed_nas': ['11b', '11b'],

    # Options for 2D KP (0 for caffe_demo, 1 for Openpose_Dome)
    '2D_detector': 0
}

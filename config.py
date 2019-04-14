CONFIG = {
    # Config for calibration

    # whether to run calibration first
    'run_calibration': False,
    # the dome sequence base name, e.g. '19XXXX_tent'
    'input_seq_name': '190405_tent',
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
    'sequence_names': ['190405_pose8'],
    # calibration data
    'calibration_data': ['190405_calib_norm'],
    # start idx
    'start_index': [100],
    # end idx
    'end_index': [7000],
    # number of camera
    'camera_number': [140],
    # Captures nas
    'captures_nas': ['12a'],
    # Processed nas
    'processed_nas': ['11b'],
    # Number of GPU
    'num_gpu': [1],

    # Options for 2D KP (0 for caffe_demo, 1 for Openpose_Dome)
    '2D_detector': 0
}

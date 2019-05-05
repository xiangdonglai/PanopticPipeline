CONFIG = {
    # Config for calibration

    # whether to run calibration first
    'run_calibration': False,
    # the dome sequence base name, e.g. '19XXXX_tent'
    'input_seq_name': '190419_tent',
    # indexes of dome sequences, e.g. 3 means to use '19XXXX_tent3'
    'input_seq_index': [1, 2, 3],
    # output data to save, e.g. '19XXXX'
    'output_date': '190419',
    # Nas Index e.g. '12a' for /media/posefs12a
    'input_nas_index': '7a',
    # local path for temporarily storing calib data, will mkdir 'input_nas_idx'/'output_date'
    'calib_root_path': '/mnt/Data2/donglaix/',

    # Config for Sequences

    # sequences names
    'sequence_names': ['190425_asl1', '190425_asl2', '190425_imu1', '190425_imu2', '190419_rom2', '190419_pose1'],
    # calibration data
    'calibration_data': ['190425_calib_norm', '190425_calib_norm', '190425_calib_norm', '190425_calib_norm', '190419_calib_norm', '190419_calib_norm'],
    # start idx
    'start_index': [100, 100, 100, 100, 100, 100],
    # end idx
    'end_index': [7400, 5900, 11900, 10900, 14900, 14900],
    # number of camera
    'camera_number': [140] * 6,
    # Captures nas
    'captures_nas': ['11b', '11b', '11b', '11b', '8a', '7a'],
    # Processed nas
    'processed_nas': ['11b', '11b', '11b', '11b', '11b', '11b'],
    # category
    'category': ['specialEvents'] * 6,

    # Options for 2D KP (0 for caffe_demo, 1 for Openpose_Dome)
    '2D_detector': 1
}

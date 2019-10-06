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
    'sequence_names': ['190611_asl1', '190611_asl2', '190611_asl3', '190611_asl4', '190611_asl5', '190611_asl6', '190611_asl7', '190611_asl8', '190611_asl10', '190611_asl11', '190611_asl12', '190611_asl13', '190611_asl14', '190611_asl15', '190419_asl2', '190419_asl4', '190419_asl5', '190425_asl1', '190425_asl2', '190425_asl3', '190425_asl5', '190425_asl7', '190425_asl9', '190425_asl10'],
    # calibration data
    'calibration_data': ['190517_calib_norm'] * 14 + ['190419_calib_norm'] * 3 + ['190425_calib_norm'] * 7,
    # start idx
    'start_index': [100] * (14 + 3 + 7),
    # end idx
    'end_index': [7400, 7400, 7400, 7400, 7400, 4400, 7400, 7400, 7400, 7400, 7400, 7400, 7400, 7400] + [7000, 7000, 7000] + [7400, 5900, 7400, 5900, 5900, 7400, 5900],
    # number of camera
    'camera_number': [140] * (14 + 3 + 7),
    # Captures nas
    'captures_nas': ['2a'] * 14 + ['11b'] * 10,
    # Processed nas
    'processed_nas': ['5b'] * 14 + ['11b'] * 5 + ['5b'] * 5,
    # category
    'category': ['specialEvents'] * (14 + 3 + 7),

    # Options for 2D KP (0 for caffe_demo, 1 for Openpose_Dome)
    '2D_detector': 1
}

CONFIG = {
    # Config for calibration

    # whether to run calibration first
    'run_calibration': True,
    # the dome sequence base name, e.g. '19XXXX_tent'
    'input_seq_name': '190301_tent',
    # indexes of dome sequences, e.g. 3 means to use '19XXXX_tent3'
    'input_seq_index': [3, 4, 8],
    # output data to save, e.g. '19XXXX'
    'output_date': '190303',
    # Nas Index e.g. '12a' for /media/posefs12a
    'input_nas_index': '12a',
    # local path for temporarily storing calib data, will mkdir 'input_nas_idx'/'output_date'
    'calib_root_path': '/mnt/sda/donglaix/'
}

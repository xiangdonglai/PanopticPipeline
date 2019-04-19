from config import CONFIG
from SEQ_INFO import parse_seq
from reconstruction import run_reconstruction
import subprocess
import os

if __name__ == '__main__':
    if CONFIG['run_calibration']:
        # run calibration first
        input_seq_name = CONFIG['input_seq_name']
        assert(type(input_seq_name)) == str

        output_date = CONFIG['output_date']
        assert(type(output_date)) == str

        input_nas_idx = CONFIG['input_nas_index']
        assert(type(input_nas_idx) == str)

        input_seq_index = CONFIG['input_seq_index']
        assert(type(input_seq_index) == list)

        calib_root_path = CONFIG['calib_root_path']
        assert(type(calib_root_path) == str)

        calib_base_path = os.path.join(calib_root_path, output_date)
        if not os.path.isdir(calib_base_path):
            os.mkdir(calib_base_path)
        assert os.path.isdir(calib_base_path)

        # call script_calibImgExt.sh to extract images to calib_base_path
        for seq_index in input_seq_index:
            cmd = ['bash', './script_calibImgExt.sh', input_seq_name,
                   input_nas_idx, output_date, calib_root_path, str(seq_index)]
            subprocess.call(cmd)

        # call DomeCalib.sh
        cmd = ['bash', './DomeCalib.sh', calib_base_path]
        subprocess.call(cmd)

        assert os.path.isdir('/media/posefs1a/Calibration/{}_calib_norm'.format(output_date))

    # run reconstruction
    # process the sequence information
    seq_infos = parse_seq(CONFIG)

    assert(CONFIG['2D_detector'] in (0, 1))

    # launch reconstruction
    for seq in seq_infos:
        run_reconstruction(seq, CONFIG)

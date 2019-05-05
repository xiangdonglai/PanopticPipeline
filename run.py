from config import CONFIG
from SEQ_INFO import parse_seq
# from reconstruction import run_reconstruction
import GPUtil
import libtmux
import subprocess
import os, sys
import json


def assign_task_to_GPU(nGPU, CONFIG):
    assert nGPU > 0
    for i in range(nGPU):
        filename = 'GPU_{}.json'.format(i)
        GPU_config = {}
        for keyname in ('sequence_names', 'calibration_data', 'start_index', 'end_index', 'camera_number', 'captures_nas', 'processed_nas', 'category'):
            # assign tasks to GPUs in turn
            GPU_config[keyname] = CONFIG[keyname][i::nGPU] 
        GPU_config['2D_detector'] = CONFIG['2D_detector']
        with open(filename, 'w') as f:
            json.dump(GPU_config, f)


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

    GPUs = GPUtil.getGPUs()  # get number of GPUs
    nGPU = len(GPUs)
    nseq = len(seq_infos)
    if nseq < nGPU:
        # if there are fewer sequences than available GPU, then reduce to the number needed
        nGPU = nseq
    assign_task_to_GPU(nGPU, CONFIG)

    # launch reconstruction
    server = libtmux.Server()
    sessions = server.list_sessions()
    for iGPU in range(nGPU):
        session_name = 'GPU_{}'.format(iGPU)
        if server.find_where({'session_name': session_name}):
            print('Session: {} exists in tmux, skip'.format(session_name))
            continue
        # tmux usage: tmux new-session -d -s ${session} '${command}; exec bash' (exec bash to keep the session alive after tasks finish)
        # sys.executable is the current Python interpretor in use
        command = 'tmux new-session -d -s {}'.format(session_name)
        os.system(command)
        # send a Enter, for skynet
        command = 'tmux send-keys -t {} Enter'.format(session_name)
        os.system(command)
        command = 'tmux send-keys -t {} "export CUDA_VISIBLE_DEVICES={:}" Enter'.format(session_name, iGPU)
        os.system(command)
        command = 'tmux send-keys -t {} "{} reconstruction.py --gpu {}" Enter'.format(session_name, sys.executable, iGPU)
        os.system(command)
        print('Launch job for GPU #{} in tmux session: {}'.format(iGPU, session_name))

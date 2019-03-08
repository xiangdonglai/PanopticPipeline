import os
import SEQ_INFO
import GPUtil
import subprocess


def check_available_gpu(CONFIG):
    if CONFIG['2D_detector'] == 0:
        GPUs = GPUtil.getGPUs()
        assert len(GPUs) == 4   # designed to run on a GPU server with 4 GPUs.
        for GPU in GPUs:
            if GPU.memoryFree < 2500:   # needs around 2500MB memory per GPU
                return False
    return True


def run_reconstruction(seq_info, CONFIG):
    assert(isinstance(seq_info, SEQ_INFO.SEQ_INFO))
    assert(type(CONFIG) == dict)

    # run 2D Detector (caffe_demo or OpenPose_dome)
    if CONFIG['2D_detector'] == 0:
        assert os.path.isfile('caffe_demo/build/examples/rtpose/rtpose_han.bin')
        assert os.path.isfile('caffe_demo/build/examples/rtpose/poseResultMerger.bin')

        # try to launch a executable; should return 0
        proc = subprocess.Popen(["build/examples/rtpose/rtpose_han.bin"], cwd='./caffe_demo/')
        proc.wait()
        assert proc.returncode == 0
        # now run the program
        assert check_available_gpu(CONFIG)
        cmd = ['bash', 'run_dome.sh', seq_info.captures_nas, seq_info.processed_nas, seq_info.name, str(seq_info.start_idx), str(seq_info.end_idx), str(seq_info.cam_num)]
        proc = subprocess.Popen(cmd, cwd='./caffe_demo/')
        proc.wait()

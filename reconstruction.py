import os
import SEQ_INFO
import GPUtil
import subprocess
from script_indexMapGen_auto import IndexMap25to30_offset


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

    done_pose_file = os.path.join(seq_info.processed_path, 'done_pose_org.log')

    if not os.path.exists(done_pose_file):

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

    else:
        print('2D pose detection files exist, skip.')

    assert os.path.exists(done_pose_file)

    done_recon_vga = os.path.join(seq_info.processed_path, 'done_recon_vga.log')
    if not os.path.exists(done_recon_vga):

        # run reconstruction
        calibPath = seq_info.calib_path
        cmd = ['./Social-Capture-Ubuntu/SFMProject/build/SFMProject', 'skel_mpm_undist_19pts', seq_info.processed_path + '/body_mpm/', calibPath,
               str(seq_info.start_idx), str(seq_info.end_idx)]
        proc = subprocess.Popen(cmd)
        proc.wait()

        calibPath = seq_info.calib_wo_distortion_path
        cmd = ['./Social-Capture-Ubuntu/SFMProject/build/SFMProject', 'skel_all_vga_mpm19', seq_info.processed_path + '/body_mpm/', calibPath,
               str(seq_info.start_idx), str(seq_info.end_idx), str(seq_info.cam_num)]
        proc = subprocess.Popen(cmd)
        proc.wait()
        assert proc.returncode == 0

        cmd = ['./Social-Capture-Ubuntu/SFMProject/build/SFMProject', 'skel_export_vga_mpm19', seq_info.processed_path + '/body_mpm/', calibPath,
               str(seq_info.start_idx), str(seq_info.end_idx), str(seq_info.cam_num)]
        proc = subprocess.Popen(cmd)
        proc.wait()
        assert proc.returncode == 0

        open(done_recon_vga, 'a').close()

    else:
        print('3D vga reconstruction files exist, skip.')

    assert os.path.exists(done_recon_vga)

    IndexMap25to30_offset(seq_info.captures_nas, seq_info.processed_nas, 'specialEvents', seq_info.name)
    assert os.path.isfile(os.path.join(seq_info.processed_path, 'body_mpm', 'IndexMap25to30_offset.txt'))

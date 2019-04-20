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

        # run 2D Detector (caffe_demo)
        if CONFIG['2D_detector'] == 0:
            assert os.path.isfile('caffe_demo/build/examples/rtpose/rtpose_han.bin')
            assert os.path.isfile('caffe_demo/build/examples/rtpose/poseResultMerger.bin')

            # try to launch a executable; should return 0
            # proc = subprocess.Popen(["build/examples/rtpose/rtpose_han.bin"], cwd='./caffe_demo/')
            # proc.wait()
            # assert proc.returncode == 0

            # now run the program
            assert check_available_gpu(CONFIG)
            cmd = ['bash', 'run_dome.sh', seq_info.captures_nas, seq_info.processed_nas, seq_info.name, str(seq_info.start_idx), str(seq_info.end_idx), str(seq_info.cam_num), str(seq_info.num_gpu)]
            proc = subprocess.Popen(cmd, cwd='./caffe_demo/')
            proc.wait()

        # run 2D Detector (OpenPose_dome)
        elif CONFIG['2D_detector'] == 1:
            assert os.path.isfile('openpose/build/examples/dome/1_dome_input_and_output.bin')
            cmd = ['bash', 'run_multi_donglai.sh', seq_info.captures_nas, seq_info.processed_nas,
                   seq_info.name, str(seq_info.start_idx), str(seq_info.end_idx), str(seq_info.cam_num), str(seq_info.num_gpu)]
            proc = subprocess.Popen(cmd, cwd='./openpose/')
            proc.wait()
    else:
        print('2D pose detection files exist, skip.')

    assert os.path.exists(done_pose_file)

    if CONFIG['2D_detector'] == 0:
        pts = 19  # flags to call SFMProject
        kp_fmt = 'mpm19'
        pose_folder = 'coco19_body3DPSRecon_json_normCoord'

    elif CONFIG['2D_detector'] == 1:
        pts = 25  # flags to call SFMProject
        kp_fmt = 'op25'
        pose_folder = 'op25_body3DPSRecon_json_normCoord'

    done_recon_vga = os.path.join(seq_info.processed_path, 'done_recon_vga.log')
    if not os.path.exists(done_recon_vga):

        # run reconstruction
        calibPath = seq_info.calib_path
        cmd = ['./Social-Capture-Ubuntu/SFMProject/build/SFMProject', 'skel_mpm_undist_{}pts'.format(pts), seq_info.processed_path + '/body_mpm/', calibPath,
               str(seq_info.start_idx), str(seq_info.end_idx)]
        proc = subprocess.Popen(cmd)
        proc.wait()

        calibPath = seq_info.calib_wo_distortion_path
        cmd = ['./Social-Capture-Ubuntu/SFMProject/build/SFMProject', 'skel_all_vga_{}'.format(kp_fmt), seq_info.processed_path + '/body_mpm/', calibPath,
               str(seq_info.start_idx), str(seq_info.end_idx), str(seq_info.cam_num)]
        proc = subprocess.Popen(cmd)
        proc.wait()
        assert proc.returncode == 0

        cmd = ['./Social-Capture-Ubuntu/SFMProject/build/SFMProject', 'skel_export_vga_{}'.format(kp_fmt), seq_info.processed_path + '/body_mpm/', calibPath,
               str(seq_info.start_idx), str(seq_info.end_idx), str(seq_info.cam_num)]
        proc = subprocess.Popen(cmd)
        proc.wait()
        assert proc.returncode == 0

        open(done_recon_vga, 'a').close()

    else:
        print('3D vga reconstruction files exist, skip.')

    assert os.path.exists(done_recon_vga)

    done_recon_hd = os.path.join(seq_info.processed_path, 'done_recon_hd.log')
    if not os.path.isfile(done_recon_hd):
        # generate index mapping from VGA to HD.
        IndexMap25to30_offset(seq_info.captures_nas, seq_info.processed_nas, 'specialEvents', seq_info.name)
        assert os.path.isfile(os.path.join(seq_info.processed_path, 'body_mpm', 'IndexMap25to30_offset.txt'))

        calibPath = seq_info.calib_wo_distortion_path
        cmd = ['./Social-Capture-Ubuntu/SFMProject/build/SFMProject', 'skel_convert_vga2hd_{}'.format(kp_fmt), seq_info.processed_path + '/body_mpm/', calibPath,
               str(seq_info.start_idx), str(seq_info.end_idx), str(seq_info.cam_num)]
        proc = subprocess.Popen(cmd)
        proc.wait()
        assert proc.returncode == 0

        calibPath = seq_info.calib_wo_distortion_path
        cmd = ['./Social-Capture-Ubuntu/SFMProject/build/SFMProject', 'skel_export_hd_{}'.format(kp_fmt), seq_info.processed_path + '/body_mpm/', calibPath]
        proc = subprocess.Popen(cmd)
        proc.wait()
        assert proc.returncode == 0

        open(done_recon_hd, 'a').close()
    else:
        print('3D hd reconstruction files exist, skip.')

    done_hd_video = os.path.join(seq_info.processed_path, 'done_hd_video.log')
    if not os.path.isfile(done_hd_video):
        # extract HD videos (for face and hand)
        for machineIdx in range(31, 47):
            for diskIdx in range(1, 3):
                if machineIdx == 46 and diskIdx == 2:
                    continue
                cmd = 'bash videoGen_hd.sh {} {} {} {} {}' \
                    .format(seq_info.captures_nas, 'specialEvents', seq_info.name, machineIdx, diskIdx)
                os.system(cmd)
        open(done_hd_video, 'a').close()
    else:
        print('HD videos already generated, skip.')
    assert os.path.exists(done_hd_video)

    hd_frames_start, hd_frames_end = seq_info.read_hd_range()
    done_face_2d = os.path.join(seq_info.processed_path, 'done_face_2d.log')
    if not os.path.isfile(done_face_2d):
        cmd = 'matlab -r "seq_name = \'{}\'; processed_path = \'{}\'; calib_name = \'{}\'; frames_start = {}; frames_end = {}; pose_folder = \'{}\'; run matlab_hand_face/script_face.m; exit;"'.format(
            seq_info.name, seq_info.processed_path, seq_info.calib, hd_frames_start, hd_frames_end, pose_folder)
        os.system(cmd)
    else:
        print('2D face output exist, skip.')
    assert os.path.exists(done_face_2d)

    # TODO: Get rid of Matlab.
    # TODO: The hand and face reconstruction can be parallelized.
    # reconstruct face
    done_face_3d = os.path.join(seq_info.processed_path, 'done_face_3d.log')
    if not os.path.isfile(done_face_3d):
        cmd = ['./Social-Capture-Ubuntu/SFMProject/build/SFMProject', 'face_pm_undistort_hd', os.path.join(seq_info.processed_path, 'exp551b_fv101b_116k'),
               seq_info.calib_path, str(hd_frames_start), str(hd_frames_end)]
        proc = subprocess.Popen(cmd)
        proc.wait()
        assert proc.returncode == 0

        cmd = ['./Social-Capture-Ubuntu/SFMProject/build/SFMProject', 'face_pm_recon_hd', os.path.join(seq_info.processed_path, 'exp551b_fv101b_116k'),
               seq_info.calib_wo_distortion_path, str(hd_frames_start), str(hd_frames_end)]
        proc = subprocess.Popen(cmd)
        proc.wait()
        assert proc.returncode == 0

        cmd = ['./Social-Capture-Ubuntu/SFMProject/build/SFMProject', 'face_pm_export_hd', os.path.join(seq_info.processed_path, 'exp551b_fv101b_116k'),
               seq_info.calib_wo_distortion_path, str(hd_frames_start), str(hd_frames_end)]
        proc = subprocess.Popen(cmd)
        proc.wait()
        assert proc.returncode == 0
        open(done_face_3d, 'a').close()
    else:
        print('3D face output exist, skip.')
    assert os.path.exists(done_face_3d)

    # reconstruct hand
    done_hand_2d = os.path.join(seq_info.processed_path, 'done_hand_2d.log')
    if not os.path.isfile(done_hand_2d):
        cmd = 'matlab -r "seq_name = \'{}\'; processed_path = \'{}\'; calib_name = \'{}\'; frames_start = {}; frames_end = {}; run matlab_hand_face/script_hand_v143_han.m; exit;"'.format(
            seq_info.name, seq_info.processed_path, seq_info.calib, hd_frames_start, hd_frames_end)
        os.system(cmd)
    else:
        print('2D hand output exist, skip')
    assert os.path.isfile(done_hand_2d)

    done_hand_3d = os.path.join(seq_info.processed_path, 'done_hand_3d.log')
    if not os.path.isfile(done_hand_3d):
        cmd = ['./Social-Capture-Ubuntu/SFMProject/build/SFMProject', 'hand_undistort_hd', os.path.join(seq_info.processed_path, 'hands_v143_120k'),
               seq_info.calib_path, str(hd_frames_start), str(hd_frames_end)]
        proc = subprocess.Popen(cmd)
        proc.wait()
        assert proc.returncode == 0

        cmd = ['./Social-Capture-Ubuntu/SFMProject/build/SFMProject', 'hand_recon_hd', os.path.join(seq_info.processed_path, 'hands_v143_120k'),
               seq_info.calib_wo_distortion_path, str(hd_frames_start), str(hd_frames_end)]
        proc = subprocess.Popen(cmd)
        proc.wait()
        assert proc.returncode == 0

        cmd = ['./Social-Capture-Ubuntu/SFMProject/build/SFMProject', 'hand_export_hd', os.path.join(seq_info.processed_path, 'hands_v143_120k'),
               seq_info.calib_wo_distortion_path, str(hd_frames_start), str(hd_frames_end)]
        proc = subprocess.Popen(cmd)
        proc.wait()
        assert proc.returncode == 0

        open(done_hand_3d, 'a').close()
    else:
        print('3D hand output exist, skip.')
        assert os.path.exists(done_hand_3d)

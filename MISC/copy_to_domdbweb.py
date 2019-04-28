import sys
import os, shutil

def copy_all(name, processed_path, kp_fmt):
    assert kp_fmt in ('coco19', 'op25')
    dst_root = '/media/domedbweb/develop/webdata/dataset/{}/'.format(name)
    
    body_vga_src = os.path.join(processed_path, 'body_mpm', '{}_body3DPSRecon_json_normCoord'.format(kp_fmt), '0140')
    body_vga_dst = os.path.join(dst_root, 'vgaPose3d_stage1_{}'.format(kp_fmt)) 
    if os.path.isdir(body_vga_src) and not os.path.exists(body_vga_dst):
        shutil.copytree(body_vga_src, body_vga_dst)

    body_hd_src = os.path.join(processed_path, 'body_mpm', '{}_body3DPSRecon_json_normCoord'.format(kp_fmt), 'hd')
    body_hd_dst = os.path.join(dst_root, 'hdPose3d_stage1_{}'.format(kp_fmt)) 
    if os.path.isdir(body_hd_src) and not os.path.exists(body_hd_dst):
        shutil.copytree(body_hd_src, body_hd_dst)

    face_src = os.path.join(processed_path, 'exp551b_fv101b_116k', 'faceRecon_pm_json', 'hd_30')
    face_dst = os.path.join(dst_root, 'hdFace3d')
    if os.path.isdir(face_src) and not os.path.exists(face_dst):
        shutil.copytree(face_src, face_dst)

    hand_src = os.path.join(processed_path, 'hands_v143_120k', 'handRecon_json', 'hd_30')
    hand_dst = os.path.join(dst_root, 'hdHand3d')
    if os.path.isdir(hand_src) and not os.path.exists(hand_dst):
        shutil.copytree(hand_src, hand_dst)

if __name__ == '__main__':
    assert len(sys.argv) == 4
    args = sys.argv
    name = args[1]
    processed_path = args[2]
    kp_fmt = args[3]
    copy_all(name, processed_path, kp_fmt)


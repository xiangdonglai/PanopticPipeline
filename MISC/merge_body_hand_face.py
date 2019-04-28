# script to merge body, hand and face json to a single file
import os
import sys
import json

if __name__ == '__main__':
    args = sys.argv
    assert len(args) == 4
    print('Name: {}'.format(args[1]))
    print('start: {}'.format(args[2]))
    print('end: {}'.format(args[3]))

    name = args[1]
    start = int(args[2])
    end = int(args[3])

    root = '/media/domedbweb/develop/webdata/dataset/{}/'.format(name)
    print (root)
    assert os.path.isdir(root)

    body_root = os.path.join(root, 'hdPose3d_stage1_coco19')
    assert os.path.isdir(body_root)
    hand_root = os.path.join(root, 'hdHand3d')
    assert os.path.isdir(hand_root)
    face_root = os.path.join(root, 'hdFace3d')
    assert os.path.isdir(face_root)
    total_root = os.path.join(root, 'hdTotal3d')
    if not os.path.isdir(total_root):
        os.mkdir(total_root)

    for i in range(start, end + 1):
        print('processing file {} [{} - {}]'.format(i, start, end))

        bodyFile = os.path.join(body_root, 'body3DScene_{:08d}.json'.format(i))
        with open(bodyFile) as f:
            bodyData = json.load(f)

        handFile = os.path.join(hand_root, 'handRecon3D_hd{:08d}.json'.format(i))
        with open(handFile) as f:
            handRawData = json.load(f)

        faceFile = os.path.join(face_root, 'faceRecon3D_hd{:08d}.json'.format(i))
        with open(faceFile) as f:
            faceRawData = json.load(f)

        totalData = {'pose': bodyData, 'hand': handRawData, 'face': faceRawData}

        with open(os.path.join(total_root, 'total3D_{:08d}.json'.format(i)), 'w') as f:
            json.dump(totalData, f)

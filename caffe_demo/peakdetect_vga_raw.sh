rawMainPath="/media/posefs3a/Captures/specialEvents/160401_ian3"
mainPath="/media/posefs4b/Processed/specialEvents/160401_ian3"
#imagePath="extIdeal_160401_ian3/idealImgs/vga_25"
savePath="heatmaps/vga_25"
frameStart=$1
frameEnd=$2
camSamplingNum=140

mkdir $mainPath/$savePath
#../build/examples/rtcpm/rtcpm_rawvga.bin 0 $rawMainPath $mainPath/$savePath $camSamplingNum 1 20 $frameStart $frameEnd
parallel --gnu -j4 --xapply ../build/examples/rtcpm/rtcpm_rawvga.bin {1} $rawMainPath $mainPath/$savePath $camSamplingNum {2} {3} $frameStart $frameEnd ::: 0 1 2 3 ::: 1 6 11 16 ::: 5 10 15 20

#build/examples/rtpose/rtpose_han.bin --rawDir /media/posefs6a/Captures/hands/161021_hands2 --rawName 161021_hands2 --num_gpu 1 --logtostderr --no_frame_drops --write_txt /media/posefs1b/Users/hanbyulj/txt --resolution 640x480 --num_scales 3 --scale_gap 0.15 --frame_start 4000 --frame_end 4001 --panel_start 3 --panel_end 4 --start_device 3


build/examples/rtpose/rtpose_han.bin --rawDir /media/posefs6a/Captures/hands/161021_hands2 --rawName 161021_hands2 --num_gpu 1 --logtostderr --no_frame_drops --write_txt /media/posefs1b/Users/hanbyulj/txt --resolution 640x480 --num_scales 3 --scale_gap 0.15 --frame_start 4000 --frame_end 4001 --panel_start 3 --panel_end 4 --start_device 3

g_camNum: 480
## CameraSamplingInfoFile: /media/posefs1b/Users/hanbyulj/cameraSamplingInfo_furthest/vga_camNames_480.txt
rawFileName: /media/posefs6a/Captures/hands/161021_hands2/vga/ve03/161021_hands2
success: open raw file
g_firstTC: 6000
imgIdx: 0
Current image number in queue: 0
imgIdx: 1
Current image number in queue: 1
imgIdx: 2
Current image number in queue: 2
imgIdx: 3
Current image number in queue: 3
imgIdx: 4
Current image number in queue: 4
imgIdx: 5
Current image number in queue: 5
imgIdx: 6
Current image number in queue: 6
imgIdx: 7
Current image number in queue: 7
imgIdx: 8
Current image number in queue: 8
imgIdx: 9
Current image number in queue: 9
imgIdx: 10
Current image number in queue: 10
imgIdx: 11
Current image number in queue: 11
imgIdx: 12
Current image number in queue: 12
imgIdx: 13
Current image number in queue: 12
imgIdx: 14
Current image number in queue: 13
imgIdx: 15
Current image number in queue: 14
imgIdx: 16
Current image number in queue: 15
imgIdx: 17
Current image number in queue: 16
imgIdx: 18
Current image number in queue: 17
imgIdx: 19
Current image number in queue: 18
imgIdx: 20
Current image number in queue: 19
imgIdx: 21
Current image number in queue: 20
imgIdx: 22
Current image number in queue: 21
imgIdx: 23
Current image number in queue: 22
imgIdx: 0
Current image number in queue: 11
imgIdx: 1
Current image number in queue: 12
imgIdx: 2
Current image number in queue: 13
imgIdx: 3
Current image number in queue: 14
imgIdx: 4
Current image number in queue: 15
imgIdx: 5
Current image number in queue: 16
imgIdx: 6
Current image number in queue: 17
imgIdx: 7
Current image number in queue: 18
imgIdx: 8
Current image number in queue: 19
imgIdx: 9
Current image number in queue: 20
imgIdx: 10
Current image number in queue: 21
imgIdx: 11
Current image number in queue: 22
imgIdx: 12
Current image number in queue: 23
imgIdx: 13
Current image number in queue: 23
imgIdx: 14
Current image number in queue: 24
imgIdx: 15
Current image number in queue: 25
imgIdx: 16
Current image number in queue: 26
imgIdx: 17
Current image number in queue: 27
imgIdx: 18
Current image number in queue: 28
imgIdx: 19
Current image number in queue: 29
imgIdx: 20
Current image number in queue: 30
imgIdx: 21
Current image number in queue: 31
imgIdx: 22
Current image number in queue: 32
imgIdx: 23
Current image number in queue: 33
rawFileName: /media/posefs6a/Captures/hands/161021_hands2/vga/ve04/161021_hands2
success: open raw file
g_firstTC: 6000
imgIdx: 0
Current image number in queue: 11
imgIdx: 1
Current image number in queue: 12
imgIdx: 2
Current image number in queue: 12
imgIdx: 3
Current image number in queue: 13
imgIdx: 4
Current image number in queue: 14
imgIdx: 5
Current image number in queue: 15
imgIdx: 6
Current image number in queue: 16
imgIdx: 7
Current image number in queue: 17
imgIdx: 8
Current image number in queue: 18
imgIdx: 9
Current image number in queue: 19
imgIdx: 10
Current image number in queue: 20
imgIdx: 11
Current image number in queue: 21
imgIdx: 12
Current image number in queue: 22
imgIdx: 13
Current image number in queue: 23
imgIdx: 14
Current image number in queue: 24
imgIdx: 15
Current image number in queue: 25
imgIdx: 16
Current image number in queue: 26
imgIdx: 17
Current image number in queue: 26
imgIdx: 18
Current image number in queue: 27
imgIdx: 19
Current image number in queue: 28
imgIdx: 20
Current image number in queue: 29
imgIdx: 21
Current image number in queue: 30
imgIdx: 22
Current image number in queue: 31
imgIdx: 23
Current image number in queue: 32
imgIdx: 0
Current image number in queue: 11
imgIdx: 1
Current image number in queue: 12
imgIdx: 2
Current image number in queue: 13
imgIdx: 3
Current image number in queue: 14
imgIdx: 4
Current image number in queue: 15
imgIdx: 5
Current image number in queue: 16
imgIdx: 6
Current image number in queue: 17
imgIdx: 7
Current image number in queue: 18
imgIdx: 8
Current image number in queue: 19
imgIdx: 9
Current image number in queue: 20
imgIdx: 10
Current image number in queue: 20
imgIdx: 11
Current image number in queue: 21
imgIdx: 12
Current image number in queue: 22
imgIdx: 13
Current image number in queue: 23
imgIdx: 14
Current image number in queue: 24
imgIdx: 15
Current image number in queue: 25
imgIdx: 16
Current image number in queue: 26
imgIdx: 17
Current image number in queue: 27
imgIdx: 18
Current image number in queue: 28
imgIdx: 19
Current image number in queue: 29
imgIdx: 20
Current image number in queue: 30
imgIdx: 21
Current image number in queue: 31
imgIdx: 22
Current image number in queue: 32
imgIdx: 23
Current image number in queue: 33
no more images....
no more images....
no more images....
no more images....
no more images....
no more images....
no more images....
no more images....
no more images....
no more images....
no more images....
no more images....
no more images....

if [ "$#" -ne 6 ]; then
	echo "Usage: $0 rawFolder saveFolder frameStartIdx frameEnd camNum"
	echo "e.g., $0 /media/posefs5a/Captures/SocialGames/160422_ultimatum1 /media/posefs5b/Processed/SocialGames/160422_ultimatum1 100 500 140"
	exit 1;
fi

rawMainPath=$1
mainPath=$2
#imagePath="extIdeal_160401_ian3/idealImgs/vga_25"
savePath="heatmaps_org/vga_25"
frameStart=$3
frameEnd=$4
camSamplingNum=$5
numgpu=$6

echo "::Processing: rawDataPath: $rawMainPath"
echo "::Processing: saveMainPath: $mainPath"
mkdir -p $mainPath/$savePath
#../build/examples/rtcpm/rtcpm_rawvga.bin 0 $rawMainPath $mainPath/$savePath $camSamplingNum 5 5 $frameStart $frameEnd
#parallel -j4 --xapply ../build/examples/rtcpm/rtcpm_rawvga.bin {1} $rawMainPath $mainPath/$savePath $camSamplingNum {2} {3} $frameStart $frameEnd ::: 0 1 2 3 ::: 1 6 11 16 ::: 5 10 15 20
#build/examples/rtpose/rtpose_han.bin --rawDir $rawMainPath --num_gpu 1 --logtostderr --no_frame_drops --write_txt $mainPath/$savePath --resolution 640x480 --num_scales 3 --scale_gap 0.15 --frame_start $frameStart --frame_end $frameEnd --panel_start 3 --panel_end 4 --start_device 3 --cam_sampleNum $camSamplingNum


#build/examples/rtpose/rtpose_han.bin --rawDir $rawMainPath --num_gpu 1 --logtostderr --no_frame_drops --write_txt $mainPath/$savePath --resolution 640x480 --num_scales 3 --scale_gap 0.15 --frame_start $frameStart --frame_end $frameEnd  --start_device 3 --panel_start 1 --panel_end 20  --cam_sampleNum $camSamplingNum
#build/examples/rtpose/rtpose_han.bin --rawDir $rawMainPath --num_gpu 4 --logtostderr --no_frame_drops --write_txt $mainPath/$savePath --resolution 640x480 --num_scales 3 --scale_gap 0.15 --frame_start $frameStart --frame_end $frameEnd  --start_device 0 --panel_start 1 --panel_end 20  --cam_sampleNum $camSamplingNum
build/examples/rtpose/rtpose_han.bin --rawDir $rawMainPath --num_gpu $numgpu --logtostderr --no_frame_drops --write_txt $mainPath/$savePath --resolution 640x480 --num_scales 3 --scale_gap 0.15 --frame_start $frameStart --frame_end $frameEnd  --start_device 0 --panel_start 1 --panel_end 20  --cam_sampleNum $camSamplingNum


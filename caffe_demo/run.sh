saveFolder=/media/posefs1b/Users/shanemoon/last_week_tonight_results

for name in {1..12} 
do
mkdir $saveFolder/${name}_img
mkdir $saveFolder/${name}_json
./build/examples/rtpose/rtpose.bin --video /media/posefs1b/Users/shanemoon/last_week_tonight/${name}.mp4 --num_gpu 4 --logtostderr --no_frame_drops --write_frames $saveFolder/${name}_img/frame --write_json $saveFolder/${name}_json/frame --no_frame_drops --num_scales 3 --scale_gap 0.1
done

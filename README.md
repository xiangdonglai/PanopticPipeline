# PanopticPipeline

```
Need OpenCV
Remove libopencv-dev
git clone https://github.com/opencv/opencv.git
git checkout 2.4.9

openpose - normal compile

SOcial-Capture**

sudo apt-get install libglew-dev
sudo apt-get install libxmu-dev libxi-dev

Compile Ceres 
git clone https://github.com/ceres-solver/ceres-solver.git
git checkout 1.13.0



Need MATLAB
cd /home/cubserver3/PanopticPipeline/cocoapi/MatlabAPI
matlab
mex('CXXFLAGS=\$CXXFLAGS -std=c++11 -Wall','-largeArrayDims','private/gasonMex.cpp','../common/gason.cpp','-I../common/','-outdir','private');

sudo add-apt-repository ppa:mc3man/gstffmpeg-keep
sudo apt-get update
sudo apt-get install gstreamer0.10-ffmpeg

#############

cd /home/cubserver3/PanopticPipeline/bvlcsh/caffe
gedit Makefile.config
CUDA_DIR := /usr/local/cuda
MATLAB_DIR := /usr/local/MATLAB/R2018a/
USE_CUDNN := 0
# CUDA architecture setting: going with all of them.
# For CUDA < 6.0, comment the *_50 lines for compatibility.
CUDA_ARCH :=  -gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_50,code=compute_50


export CPATH="/usr/include/hdf5/serial/"

cd /usr/lib/x86_64-linux-gnu/hdf5/serial/
sudo ln -sf libhdf5_serial.so libhdf5.so
sudo ln -sf libhdf5_serial_hl.so libhdf5_hl.so

#############

Enable Matlab!!!

Add to CMakeLists.txt

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_calib3d opencv_video opencv_features2d opencv_ml opencv_highgui opencv_objdetect opencv_contrib opencv_legacy opencv_gpu)

#############



cd domeCalibrator/DomeCalibScript/ && wget http://ccwu.me/vsfm/download/VisualSFM_linux_64bit.zip && unzip VisualSFM_linux_64bit.zip && cd vsfm && make -j12

sudo apt-get install libdevil-dev

git clone https://github.com/pitzer/SiftGPU.git; 
cd SiftGPU
# in makefile change siftgpu_enable_cuda = 1 BOTH; 
make
copy libsiftgpu.so to vsfm/bin/
cp bin/libsiftgpu.so ../domeCalibrator/DomeCalibScript/vsfm/bin

wget http://grail.cs.washington.edu/projects/mcba/pba_v1.0.5.zip
unzip pba_v1.0.5.zip
cd pba; make;
cp bin/libpba.so ../domeCalibrator/DomeCalibScript/vsfm/bin


sudo apt-get install libsndfile-dev
wget https://ayera.dl.sourceforge.net/project/ltcsmpte/libltcsmpte/v0.4.4/libltcsmpte-0.4.4.tar.gz
tar -xvzf  libltcsmpte-0.4.4.tar.gz
cd libltcsmpte-0.4.4
./configure
make -j10
sudo make install
# Add "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/" to .bashrc and source it

cd /home/cubserver3/PanopticPipeline/panopticstudio/Application/syncTableGeneratorVHK/syncTableGeneratorVHK_code/ImageExtractorVHKProj
cd build
cmake ..; make;
cp ImageExtractorVHK ../../../

############

cd /media/posefs1a/Calibration
# Choose the calibration you want, and rename it
# 190425_calib_norm, 190425_calib_rawData -> i changed date to testing

cd /home/cubserver3/PanopticPipeline/domeCalibrator/DomeCalibScript/vsfm/bin
ensure nv.ini is copied here

sudo pip3 install GPUtil
sudo pip3 install libtmux

############

If youre in SSH-X remove the X
```

FROM ubuntu:16.04
FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

RUN apt-get update && apt-get -y install apt-utils && apt-get -y install git && apt-get install unzip && apt-get install wget
WORKDIR tools
RUN apt-get -y install g++-5 cmake
RUN apt-get -y install libopencv-dev

RUN apt-get -y install libsndfile-dev && wget https://ayera.dl.sourceforge.net/project/ltcsmpte/libltcsmpte/v0.4.4/libltcsmpte-0.4.4.tar.gz && tar xzvf libltcsmpte-0.4.4.tar.gz && cd libltcsmpte-0.4.4 && ./configure && make -j12 && make install
ENV LD_LIBRARY_PATH /usr/local/lib/:/usr/lib/

RUN apt-get -y install gtk+-2.0 libglu1-mesa-dev freeglut3-dev mesa-common-dev libdevil-dev libglew-dev

RUN useradd -ms /bin/bash donglaix
WORKDIR /home/donglaix/PanopticPipeline/
ADD ./panopticstudio /home/donglaix/PanopticPipeline/panopticstudio
RUN cd panopticstudio/Application/syncTableGeneratorVHK/syncTableGeneratorVHK_code/syncTableGeneratorVHKProj/ && mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=/usr/bin/g++-5 .. && make -j12 && cp syncTableGeneratorVHK ../../../
ADD ./domeCalibrator /home/donglaix/PanopticPipeline/domeCalibrator
RUN cd /home/donglaix/PanopticPipeline/domeCalibrator/DomeCalibScript/ && wget http://ccwu.me/vsfm/download/VisualSFM_linux_64bit.zip && unzip VisualSFM_linux_64bit.zip && cd vsfm && make -j12
RUN cd /home/donglaix/PanopticPipeline/ && git clone https://github.com/pitzer/SiftGPU.git && sed -i 's/siftgpu_enable_cuda = 0/siftgpu_enable_cuda = 1/g' SiftGPU/makefile && cd SiftGPU && make && cd ..
RUN wget http://grail.cs.washington.edu/projects/mcba/pba_v1.0.5.zip && unzip pba_v1.0.5.zip && cd pba && make && mv bin/libpba.so ../domeCalibrator/DomeCalibScript/vsfm/bin && cd .. && rm pba_v1.0.5.zip
RUN cd panopticstudio/Application/syncTableGeneratorVHK/syncTableGeneratorVHK_code/ImageExtractorVHKProj/ && mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=/usr/bin/g++-5 .. && make -j12 && cp ImageExtractorVHK ../../../
ADD nv.ini domeCalibrator/DomeCalibScript/vsfm/bin/nv.ini

ADD Social-Capture-Ubuntu/DomeBundle Social-Capture-Ubuntu/DomeBundle
RUN apt-get -y install libeigen3-dev libboost-all-dev
RUN apt-get -y install libgflags-dev libgoogle-glog-dev libsuitesparse-dev 
RUN wget https://github.com/ceres-solver/ceres-solver/archive/1.13.0.zip && unzip 1.13.0 && cd ceres-solver-1.13.0 && mkdir build && cd build && cmake .. && make -j20 && make install
RUN cd Social-Capture-Ubuntu/DomeBundle && mkdir build && cd build && cmake .. && make -j12
ADD Social-Capture-Ubuntu/DomeCorres Social-Capture-Ubuntu/DomeCorres
RUN wget https://iweb.dl.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.9/opencv-2.4.9.zip && unzip opencv-2.4.9.zip
RUN cd opencv-2.4.9 && mkdir build && cd build && cmake -DCMAKE_C_COMPILER=/usr/bin/gcc-5 -DCMAKE_CXX_COMPILER=/usr/bin/g++-5 -DCMAKE_CXX_STANDARD=11 -DWITH_CUDA=OFF .. && make -j20 && make install
RUN cd Social-Capture-Ubuntu/DomeCorres && mkdir build && cd build && cmake .. && make -j12 && cd ../../../

RUN apt-get -y install libxmu-dev
ADD Social-Capture-Ubuntu/SFMProject Social-Capture-Ubuntu/SFMProject
RUN cd Social-Capture-Ubuntu/SFMProject && mkdir build && cd build && cmake .. && make -j20
RUN chown -R donglaix /home/donglaix/PanopticPipeline

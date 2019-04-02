#include <vector>
#include "caffe/layers/imwarp_layer.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace caffe {

template <typename Dtype>
void ImWarpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	ImWarpParameter imwarp_param = this->layer_param_.imwarp_param();
	targetSpatialWidth = imwarp_param.target_spatial_width(); //temporarily
	targetSpatialHeight = imwarp_param.target_spatial_height();
}

template <typename Dtype>
void ImWarpLayer<Dtype>::setTargetDimenions(int nw, int nh){
	targetSpatialWidth = nw;
	targetSpatialHeight = nh;
}

template <typename Dtype>
void ImWarpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	vector<int> bottom_shape = bottom[0]->shape();
	vector<int> top_shape(bottom_shape);

	ImWarpParameter imwarp_param = this->layer_param_.imwarp_param();

	top_shape[3] = targetSpatialWidth;
	top_shape[2] = targetSpatialHeight;
	top_shape[0] = bottom_shape[0];
	top[0]->Reshape(top_shape);

	CHECK_EQ(bottom[1]->shape(0), 1);
	CHECK_EQ(bottom[1]->shape(1), 2);
	CHECK_EQ(bottom[1]->shape(2), 3);
	CHECK_EQ(bottom[1]->shape(3), 1);
}

template <typename Dtype>
void ImWarpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	NOT_IMPLEMENTED;

	int num = bottom[0]->shape(0);
	int channel = bottom[0]->shape(1);
	int oriSpatialHeight = bottom[0]->shape(2);
	int oriSpatialWidth = bottom[0]->shape(3);

	//stupid method
	for(int n = 0; n < num; n++){
		for(int c = 0; c < channel; c++){
			Mat src(oriSpatialWidth, oriSpatialHeight, CV_32FC1);
			Mat dst(targetSpatialWidth, targetSpatialHeight, CV_32FC1);
			//fill src
			int offset2 = oriSpatialHeight * oriSpatialWidth;
			int offset3 = offset2 * channel;
			for (int y = 0; y < oriSpatialHeight; y++){
				for (int x = 0; x < oriSpatialWidth; x++){
					Dtype* src_pointer = bottom[0]->mutable_cpu_data();
				    src.at<Dtype>(x,y) = src_pointer[n*offset3 + c*offset2 + y*oriSpatialWidth + x];
				}
			}
			//resize
			// cv::imwarp(src, dst, dst.size(), 0, 0, INTER_CUBIC);
			//fill top
			offset2 = targetSpatialHeight * targetSpatialWidth;
			offset3 = offset2 * channel;
			for (int y = 0; y < targetSpatialHeight; y++){
				for (int x = 0; x < targetSpatialWidth; x++){
					Dtype* dst_pointer = top[0]->mutable_cpu_data();
				    dst_pointer[n*offset3 + c*offset2 + y*targetSpatialWidth + x] = dst.at<Dtype>(x,y);
				}
			}
		}
	}
}

template <typename Dtype>
void ImWarpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(ImWarpLayer);
#endif

INSTANTIATE_CLASS(ImWarpLayer);
REGISTER_LAYER_CLASS(ImWarp);

}  // namespace caffe

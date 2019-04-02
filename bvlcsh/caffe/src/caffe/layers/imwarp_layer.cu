#include "caffe/layers/imwarp_layer.hpp"

#define numThreadsPerBlock_1d 16
#define numThreadsPerBlock 256

namespace caffe {

template <typename Dtype>
inline __device__ void cubic_interpolation(Dtype &out, Dtype &v0, Dtype &v1, Dtype &v2, Dtype &v3, float dx) {
    // Dtype a = (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3);
    // Dtype b = (v0 - 2.5f * v1 + 2.0 * v2 - 0.5 * v3);
    // Dtype c = (-0.5f * v0 + 0.5f * v2);
    // out = ((a * dx + b) * dx + c) * dx + v1;
    out = (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3) * dx * dx * dx
         + (v0 - 2.5f * v1 + 2.0 * v2 - 0.5 * v3) * dx * dx
         + (-0.5f * v0 + 0.5f * v2) * dx
         + v1;
}


template <typename Dtype>
__global__ void imwarp_cubic_kernel(Dtype* src_pointer, Dtype* dst_pointer,
	                                  int ow, int oh, int tw, int th, Dtype *M){
	// get pixel location (x,y)
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const Dtype M11 = M[0*3+0];
  const Dtype M12 = M[0*3+1];
  const Dtype M13 = M[0*3+2];
  const Dtype M21 = M[1*3+0];
  const Dtype M22 = M[1*3+1];
  const Dtype M23 = M[1*3+2];
	//int tid = threadIdx.y * blockDim.x + threadIdx.x;

	//get (min_x,max_x) (min_y,max_y)
	// int min_x = (blockIdx.x * blockDim.x);
	// int max_x = (blockIdx.x * blockDim.x) + blockDim.x - 1;
	// int min_y = (blockIdx.y * blockDim.y);
	// int max_y = (blockIdx.y * blockDim.y) + blockDim.y - 1;

	// int min_x_ori = min_x * (float(ow) / tw);
	// int max_x_ori = max_x * (float(ow) / tw);
	// int min_y_ori = min_y * (float(oh) / th);
	// int max_y_ori = max_y * (float(oh) / th);

	// min_x_ori = (min_x_ori - 1 < 0) ? min_x_ori : (min_x_ori - 1);
	// max_x_ori = (max_x_ori + 2 >= ow) ? (max_x_ori + 1 >= ow ? max_x_ori : max_x_ori+1) : (max_x_ori + 2);
	// min_y_ori = (min_y_ori - 1 < 0) ? min_y_ori : (min_y_ori - 1);
	// max_y_ori = (max_y_ori + 2 >= oh) ? (max_y_ori + 1 >= oh ? max_y_ori : max_y_ori+1) : (max_y_ori + 2);

	// // load into shared memory: fixed for 7x7
	// __shared__ Dtype shared[7][7];
	// if(threadIdx.x < 7 && threadIdx.y < 7 && min_x_ori + threadIdx.x < ow && min_y_ori + threadIdx.y < oh) {
	// 	int x_ref = min_x_ori + threadIdx.x;
	// 	int y_ref = min_y_ori + threadIdx.y;
	// 	shared[threadIdx.x][threadIdx.y] = src_pointer[y_ref * ow + x_ref];
	// }

	// begin compute
	if(x < tw && y < th) {
    Dtype x_on_ori = M11*x + M12*y + M13;
    Dtype y_on_ori = M21*x + M22*y + M23;

		int x_nei[4];
		x_nei[1] = int(x_on_ori + 1e-5);
		x_nei[1] = (x_nei[1]<0) ? 0 : x_nei[1];
		x_nei[1] = (x_nei[1]>=ow) ? ow-1 : x_nei[1];

		x_nei[1] = (x_nei[1] < 0) ? 0 : x_nei[1];
		x_nei[0] = (x_nei[1] - 1 < 0) ? x_nei[1] : (x_nei[1] - 1);
		x_nei[2] = (x_nei[1] + 1 >= ow) ? (ow - 1) : (x_nei[1] + 1);
		x_nei[3] = (x_nei[2] + 1 >= ow) ? (ow - 1) : (x_nei[2] + 1);
		float dx = x_on_ori - x_nei[1];
		dx = (dx<0) ? 0 : dx;
		dx = (dx>1) ? 1 : dx;

		int y_nei[4];
		y_nei[1] = int(y_on_ori + 1e-5);
		y_nei[1] = (y_nei[1]<0) ? 0 : y_nei[1];
		y_nei[1] = (y_nei[1]>=oh) ? oh-1 : y_nei[1];


		y_nei[1] = (y_nei[1] < 0) ? 0 : y_nei[1];
		y_nei[0] = (y_nei[1] - 1 < 0) ? y_nei[1] : (y_nei[1] - 1);
		y_nei[2] = (y_nei[1] + 1 >= oh) ? (oh - 1) : (y_nei[1] + 1);
		y_nei[3] = (y_nei[2] + 1 >= oh) ? (oh - 1) : (y_nei[2] + 1);
		float dy = y_on_ori - y_nei[1];
		dy = (dy<0) ? 0 : dy;
		dy = (dy>1) ? 1 : dy;

		Dtype temp[4];
		for(int i = 0; i < 4; i++){
			cubic_interpolation(temp[i], src_pointer[y_nei[i]*ow + x_nei[0]],
				                         src_pointer[y_nei[i]*ow + x_nei[1]],
				                         src_pointer[y_nei[i]*ow + x_nei[2]],
				                         src_pointer[y_nei[i]*ow + x_nei[3]], dx);
			// cubic_interpolation(temp[i], shared[x_nei[0]-min_x_ori][y_nei[i]-min_y_ori],
			// 	                         shared[x_nei[1]-min_x_ori][y_nei[i]-min_y_ori],
			// 	                         shared[x_nei[2]-min_x_ori][y_nei[i]-min_y_ori],
			// 	                         shared[x_nei[3]-min_x_ori][y_nei[i]-min_y_ori], dx);
		}
		cubic_interpolation(dst_pointer[y*tw+x], temp[0], temp[1], temp[2], temp[3], dy);
	}
}



template <typename Dtype>
__global__ void imwarp_cubic_kernel(Dtype* src_ptr, Dtype* dst_pointer, int src_offset, int num,
  int oriSpatialWidth, int oriSpatialHeight, int tw, int th, Dtype *M){
	// get pixel location (x,y)
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int ow = oriSpatialWidth;
  int oh = oriSpatialHeight;
  const Dtype M11 = M[0*3+0];
  const Dtype M12 = M[0*3+1];
  const Dtype M13 = M[0*3+2];
  const Dtype M21 = M[1*3+0];
  const Dtype M22 = M[1*3+1];
  const Dtype M23 = M[1*3+2];

	// begin compute
	if(x>=0 && y>=0 && x < tw && y < th) {
		Dtype d_temp = 0;
		Dtype sum = 0;
		for(int n = 0; n < num; n++){
      		Dtype x_on_ori = M11*x + M12*y + M13;
      		Dtype y_on_ori = M21*x + M22*y + M23;
      x_on_ori = (x_on_ori>0) ? x_on_ori : 0;
      y_on_ori = (y_on_ori>0) ? y_on_ori : 0;
			Dtype* src_pointer = src_ptr + n * src_offset;

			int x_nei[4];
			x_nei[1] = int(x_on_ori + 1e-5);
			x_nei[1] = (x_nei[1]>0) ? x_nei[1]:0;
			x_nei[1] = (x_nei[1]>=ow) ? ow-1 : x_nei[1];

			x_nei[0] = ((x_nei[1] - 1 < 0) ? x_nei[1] : (x_nei[1] - 1));
			x_nei[2] = (x_nei[1] + 1 > ow-1) ? (ow - 1) : (x_nei[1] + 1);
			x_nei[3] = ((x_nei[2] + 1 > ow-1) ? (ow - 1) : (x_nei[2] + 1));
			float dx = x_on_ori - x_nei[1];
			dx = (dx>0) ? dx : 0;
			dx = (dx>1) ? 1 : dx;

			int y_nei[4];
			y_nei[1] = int(y_on_ori + 1e-5);
			y_nei[1] = (y_nei[1]>0) ? y_nei[1] : 0;
			y_nei[1] = (y_nei[1]>=oh) ? oh-1 : y_nei[1];

			y_nei[0] = ((y_nei[1] - 1 < 0) ? y_nei[1] : (y_nei[1] - 1));
			y_nei[2] = (y_nei[1] + 1 > oh-1) ? (oh - 1) : (y_nei[1] + 1);
			y_nei[3] = ((y_nei[2] + 1 > oh-1) ? (oh - 1) : (y_nei[2] + 1));
			float dy = y_on_ori - y_nei[1];
			dy = (dy>0) ? dy : 0;
			dy = (dy>1) ? 1 : dy;

      		Dtype temp[4];
			for(int i = 0; i < 4; i++){
				cubic_interpolation(temp[i], src_pointer[y_nei[i]*(ow) + x_nei[0]],
					                         src_pointer[y_nei[i]*(ow) + x_nei[1]],
					                         src_pointer[y_nei[i]*(ow)+ x_nei[2]],
					                         src_pointer[y_nei[i]*(ow) + x_nei[3]], dx);
			}
			//cubic_interpolation(dst_pointer[y*tw+x], temp[0], temp[1], temp[2], temp[3], dy);
			cubic_interpolation(d_temp, temp[0], temp[1], temp[2], temp[3], dy);
			sum = sum + d_temp;
		}
		dst_pointer[y*tw+x] = sum / num;
	}
}

template <typename Dtype>
void ImWarpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	//Forward_cpu(bottom, top);
	//magic here
	Dtype* src_pointer = bottom[0]->mutable_gpu_data();
  Dtype* M = bottom[1]->mutable_gpu_data();
	Dtype* dst_pointer = top[0]->mutable_gpu_data();
	int oriSpatialHeight = bottom[0]->shape(2);
	int oriSpatialWidth = bottom[0]->shape(3);
  CHECK_EQ(bottom[0]->shape(0), 1);
	int num = 1; //scale number
	int channel = bottom[0]->shape(1);
	//LOG(ERROR) << "GPU num " << num << " channel " << channel;
	//LOG(ERROR) << "top[0] " << top[0]->shape(0) << " top[0]->shape(1) " << top[0]->shape(1);

	dim3 threadsPerBlock(numThreadsPerBlock_1d, numThreadsPerBlock_1d);
	dim3 numBlocks(updiv(targetSpatialWidth, threadsPerBlock.x), updiv(targetSpatialHeight, threadsPerBlock.y));
	int offset_src = oriSpatialHeight * oriSpatialWidth;
	int offset_dst = targetSpatialWidth * targetSpatialHeight;
	//int sm_width = numThreadsPerBlock_1d / (float(targetSpatialWidth) / oriSpatialWidth) + 3;
	//int sm_height = numThreadsPerBlock_1d / (float(targetSpatialHeight) / oriSpatialHeight) + 3;


		for(int c = 0; c < channel; c++){
			// imresize_cubic_kernel<<<numBlocks, threadsPerBlock>>>(src_pointer + (n * channel + c) * offset_src,
			// 	                                                dst_pointer + (n * channel + c) * offset_dst,
			// 	                                    			oriSpatialWidth, oriSpatialHeight,
			// 	                                    			targetSpatialWidth, targetSpatialHeight, M);
			imwarp_cubic_kernel<<<numBlocks, threadsPerBlock>>>(src_pointer + c * offset_src, dst_pointer + c * offset_dst,
																channel* offset_src, 1,
				                                    			oriSpatialWidth, oriSpatialHeight,
				                                    			targetSpatialWidth, targetSpatialHeight, M);
			//LOG(ERROR) << "GPU oriSpatialHeight - 2*padh " << oriSpatialHeight - 2*padh;
		}

		//fuse_kernel<<<numBlocks, threadsPerBlock>>>(src_pointer + (n * channel + c) * offset_src,
		//		                                                targetSpatialWidth, targetSpatialHeight);
}

template <typename Dtype>
void ImWarpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(ImWarpLayer);

} // namespace caffe

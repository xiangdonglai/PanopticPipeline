#include "caffe/layers/hypercolumn_layer.hpp"
#include "caffe/util/gpu_util.cuh"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <iterator>
#include <sstream>
#define numThreadsPerBlock_1d 256
#define numThreadsPerBlock 256

/*
static inline int updiv(int a, int b){
  return (a+b-1)/b;
}
*/
namespace caffe {


template <typename Dtype>
inline __device__ Dtype bilinear_interp(const Dtype &v11, const Dtype &v12,
                                        const Dtype &v21, const Dtype &v22,
                                        Dtype dx, Dtype dy) {
  typedef Dtype D;
  return (v11 * (D(1)-dy) + v21 * dy) * (D(1)-dx)
         + (v12 * (D(1)-dy) + v22 * dy) * dx;
}


template <typename Dtype>
__global__ void hypercolumn_fwd_kernel(const Dtype* bot_pointer,
                                       int offset_bot,
                                       Dtype* top_pointer,
                                       const int *rand_pointer,
                                       int N_, int n_channels_, int bottom_nums,
	                                     int sw, int sh, int sch, int tw, int th,
                                       Dtype padf, Dtype poolf) {
	// get pixel location (x,y)
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int n = i/N_;
  top_pointer += i*n_channels_;
  bot_pointer += n*sw*sh*sch;
	// begin compute
	if (i<N_*bottom_nums) {
    int x_pt = rand_pointer[i*2+0];
    int y_pt = rand_pointer[i*2+1];
    const Dtype tx = (x_pt-padf)/poolf;
    const Dtype ty = (y_pt-padf)/poolf;

    int tx1 = (int)tx;
    int ty1 = (int)ty;
    int tx2 = tx1+1;
    int ty2 = ty1+1;

    // check if they are within the size limit
    tx1 = tx1<0 ? 0 : (tx1<sw ? tx1 : sw-1);
    tx2 = tx2<0 ? 0 : (tx2<sw ? tx2 : sw-1);
    ty1 = ty1<0 ? 0 : (ty1<sh ? ty1 : sh-1);
    ty2 = ty2<0 ? 0 : (ty2<sh ? ty2 : sh-1);

    Dtype dx = tx - tx1;
    Dtype dy = ty - ty1;

    int p11 = ty1 * sw + tx1;
    int p12 = ty1 * sw + tx2;
    int p21 = ty2 * sw + tx1;
    int p22 = ty2 * sw + tx2;
    // This is maybe a bit slower than single channel, but less CPU intensive?
    for (int ch=0;ch<sch;ch++) {

      top_pointer[0] = bilinear_interp(bot_pointer[p11], bot_pointer[p12],
                                       bot_pointer[p21], bot_pointer[p22],
                                       dx, dy);
      top_pointer += 1;
      bot_pointer += offset_bot;
    }

	}
}


template <typename Dtype>
void HypercolumnLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	//Forward_cpu(bottom, top);

  SetupRandPoints(bottom, top);

  const int bottom_width = bottom[start_id_]->width();
  const int bottom_height = bottom[start_id_]->height();
  const int bottom_nums = bottom[start_id_]->num();
  int sn_channels = bottom[end_id_+2]->channels();

  // Update random point index on the gpu
  int *rand_pointer = workspace.mutable_gpu_data();
  CUDA_CHECK(cudaMemcpy(rand_pointer, &rand_points_[0],
                        rand_points_.size()*sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_LE(rand_points_.size(), workspace.shape()[0]*workspace.shape()[1]);

  std::vector<const Dtype*> bottom_layers(n_hblobs_);
  for (int b = 0; b < n_hblobs_; b++) {
    bottom_layers[b] = bottom[b]->gpu_data();
  }
  // const Dtype* sn_data = bottom[end_id_+2]->gpu_data();

  CHECK_EQ(N_*bottom_nums*2, rand_points_.size());

	dim3 threadsPerBlock(numThreadsPerBlock_1d, 1);
	dim3 numBlocks(updiv(N_*bottom_nums, threadsPerBlock.x), 1);

  // Data
  Dtype* top_pointer = top[0]->mutable_gpu_data();
  int dst_ch = 0;
  for (int b = 0; b < n_hblobs_; b++) {
    const int cur_nCh = bottom[b]->channels();
    const Dtype* bot_pointer = bottom_layers[b];
    int offset_bot = width_[b] * height_[b];
		// for(int c = 0; c < cur_nCh; c++){
			hypercolumn_fwd_kernel<<<numBlocks, threadsPerBlock>>>(
          bot_pointer, offset_bot,
          top_pointer + dst_ch,
          rand_pointer,
          N_, n_channels_, bottom_nums,
          width_[b], height_[b], cur_nCh,
          width_[0], height_[0],
          padf_[b], (Dtype)poolf_[b]);
    // }
    dst_ch+=cur_nCh;
  }

  // Labels
  top_pointer = top[1]->mutable_gpu_data();
  dst_ch = 0;
  for (int b = end_id_+2; b < end_id_+3; b++) {
    const int cur_nCh = bottom[b]->channels();
    const Dtype* bot_pointer = bottom[b]->mutable_gpu_data();
    int offset_bot = width_[0] * height_[0];
		// for(int c = 0; c < cur_nCh; c++){
			hypercolumn_fwd_kernel<<<numBlocks, threadsPerBlock>>>(bot_pointer,
                                                  offset_bot,
				                                          top_pointer + dst_ch,
                                                  rand_pointer,
                                                  N_, cur_nCh, bottom_nums,
				                                    			width_[0], height_[0], cur_nCh,
				                                    			width_[0], height_[0],
                                                  (Dtype)0, (Dtype)1);
    // }
    dst_ch+=cur_nCh;
  }

}


template <typename Dtype>
__global__ void hypercolumn_bwd_kernel_syncadd(Dtype* bot_pointer,
                                       int offset_bot, const Dtype* top_pointer,
                                       const int *rand_pointer,
                                       int N_, int n_channels_, int bottom_nums,
	                                     int sw, int sh, int sch, int tw, int th,
                                       Dtype padf, Dtype poolf) {
	// get pixel location (x,y)
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int n = i/N_;
  top_pointer += i*n_channels_;
  bot_pointer += n*sw*sh*sch;
	// begin compute
	if (i<N_*bottom_nums) {
    int x_pt = rand_pointer[i*2+0];
    int y_pt = rand_pointer[i*2+1];
    const Dtype tx = (x_pt-padf)/poolf;
    const Dtype ty = (y_pt-padf)/poolf;

    int tx1 = (int)tx;
    int ty1 = (int)ty;
    int tx2 = tx1+1;
    int ty2 = ty1+1;

    // check if they are within the size limit
    tx1 = tx1<0 ? 0 : (tx1<sw ? tx1 : sw-1);
    tx2 = tx2<0 ? 0 : (tx2<sw ? tx2 : sw-1);
    ty1 = ty1<0 ? 0 : (ty1<sh ? ty1 : sh-1);
    ty2 = ty2<0 ? 0 : (ty2<sh ? ty2 : sh-1);

    Dtype dx = tx - tx1;
    Dtype dy = ty - ty1;

    int p11 = ty1 * sw + tx1;
    int p12 = ty1 * sw + tx2;
    int p21 = ty2 * sw + tx1;
    int p22 = ty2 * sw + tx2;

    for (int ch=0;ch<sch;ch++) {

      const Dtype dv = top_pointer[0];

      caffe_gpu_atomic_add(dv * ((Dtype)1.-dy) * ((Dtype)1.-dx), bot_pointer+p11);
      caffe_gpu_atomic_add(dv * dy * ((Dtype)1.-dx), bot_pointer+p21);
      caffe_gpu_atomic_add(dv * ((Dtype)1.-dy) * dx, bot_pointer+p12);
      caffe_gpu_atomic_add(dv * dy * dx, bot_pointer+p22);

      top_pointer += 1;
      bot_pointer += offset_bot;

    }
	}
}

template <typename Dtype>
__global__ void hypercolumn_bwd_kernel_assign(Dtype* bot_pointer, const Dtype* top_pointer,
                                       const int *rand_pointer,
                                       int N_, int n_channels_, int bottom_nums,
	                                     int sw, int sh, int sch, int tw, int th,
                                       Dtype padf, Dtype poolf) {
	// get pixel location (x,y)
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int n = i/N_;
  top_pointer += i*n_channels_;
  bot_pointer += n*sw*sh*sch;
	// begin compute
	if (i<N_*bottom_nums) {
    int tx1 = rand_pointer[i*2+0];
    int ty1 = rand_pointer[i*2+1];

    int p11 = ty1 * sw + tx1;
    const Dtype dv = top_pointer[0];
    bot_pointer[p11] = dv;
	}
}

// save pixel_inds and values, sort_by_key, reduce_by_key version of above
template <typename Dtype>
__global__ void hypercolumn_bwd_kernel(Dtype* bot_pointer, const Dtype* top_pointer,
                                       const int *rand_pointer,
                                       int *pixel_inds, Dtype *pixel_vals,
                                       int N_, int n_channels_, int bottom_nums,
	                                     int sw, int sh, int sch, int tw, int th,
                                       Dtype padf, Dtype poolf) {
	// get pixel location (x,y)
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int n = i/N_;
  top_pointer += i*n_channels_;
	// begin compute
	if (i<N_*bottom_nums) {
    int x_pt = rand_pointer[i*2+0];
    int y_pt = rand_pointer[i*2+1];
    const Dtype tx = (x_pt-padf)/poolf;
    const Dtype ty = (y_pt-padf)/poolf;

    int tx1 = (int)tx;
    int ty1 = (int)ty;
    int tx2 = tx1+1;
    int ty2 = ty1+1;

    // check if they are within the size limit
    tx1 = tx1<0 ? 0 : (tx1<sw ? tx1 : sw-1);
    tx2 = tx2<0 ? 0 : (tx2<sw ? tx2 : sw-1);
    ty1 = ty1<0 ? 0 : (ty1<sh ? ty1 : sh-1);
    ty2 = ty2<0 ? 0 : (ty2<sh ? ty2 : sh-1);

    Dtype dx = tx - tx1;
    Dtype dy = ty - ty1;

    int p11 = ty1 * sw + tx1 + n*sw*sh*sch;
    int p12 = ty1 * sw + tx2 + n*sw*sh*sch;
    int p21 = ty2 * sw + tx1 + n*sw*sh*sch;
    int p22 = ty2 * sw + tx2 + n*sw*sh*sch;

    const Dtype dv = top_pointer[0];

    // Save indices and values to accumulate
    pixel_inds[i*4+0] = p11;
    pixel_inds[i*4+1] = p12;
    pixel_inds[i*4+2] = p21;
    pixel_inds[i*4+3] = p22;

    typedef Dtype D;
    pixel_vals[i*4+0] = dv * (D(1)-dy) * (D(1)-dx);
    pixel_vals[i*4+1] = dv * (D(1)-dy) * dx;
    pixel_vals[i*4+2] = dv * dy * (D(1)-dx);
    pixel_vals[i*4+3] = dv * dy * dx;
	}
}

template <typename Dtype>
__global__ void write_result_kernel(const int *pixel_inds,
                                  const Dtype *pixel_vals,
                                  Dtype* bot_pointer,
                                  int max_count){
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<max_count) {
    bot_pointer[pixel_inds[i]] = pixel_vals[i];
  }
}
template <typename Dtype>
void HypercolumnLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

  // TODO: Fix checking of propagate down to all indices

  const int bottom_width = bottom[start_id_]->width();
  const int bottom_height = bottom[start_id_]->height();
  const int bottom_nums = bottom[start_id_]->num();
  int sn_channels = bottom[end_id_+2]->channels();

  std::vector<Dtype*> bottom_layers(n_hblobs_);
  for (int b = 0; b < n_hblobs_; b++) {
    if (!propagate_down[b]) continue;
    bottom_layers[b] = bottom[b]->mutable_gpu_diff();
    caffe_gpu_set(bottom[b]->count(), (Dtype)0, bottom_layers[b]);
  }
  const Dtype* sn_data = bottom[end_id_+2]->gpu_diff();

  CHECK_EQ(N_*bottom_nums*2, rand_points_.size());

  const int *rand_pointer = workspace.gpu_data();
  // int *pixel_inds = workspace_inds.mutable_gpu_data();
  // Dtype *pixel_vals = workspace_vals.mutable_gpu_data();
  // caffe_gpu_set(N_*bottom_nums*4, (int)0, pixel_inds);
  // caffe_gpu_set(N_*bottom_nums*4, (Dtype)0, pixel_vals);

	dim3 threadsPerBlock(numThreadsPerBlock_1d, 1);
	dim3 numBlocks(updiv(N_*bottom_nums, threadsPerBlock.x), 1);
  dim3 threadsPerBlock4(numThreadsPerBlock_1d, 1);
	dim3 numBlocks4(updiv(N_*bottom_nums*4, threadsPerBlock.x), 1);

  const Dtype* top_pointer = top[0]->gpu_diff();
  int dst_ch = 0;
  for (int b = 0; b < n_hblobs_; b++) {
    if (!propagate_down[b]) continue;
    const int cur_nCh = bottom[b]->channels();
    Dtype* bot_pointer = bottom_layers[b];
    int offset_bot = width_[b] * height_[b];
		// for(int c = 0; c < cur_nCh; c++) {
      hypercolumn_bwd_kernel_syncadd<<<numBlocks, threadsPerBlock>>>(bot_pointer,
                                                  offset_bot,
                                                  top_pointer + dst_ch,
                                                  rand_pointer,
                                                  N_, n_channels_, bottom_nums,
                                                  width_[b], height_[b], cur_nCh,
                                                  width_[0], height_[0],
                                                  padf_[b], (Dtype)poolf_[b]);
      dst_ch += cur_nCh;
      // dst_ch++;

      // debug this. not working.
			// hypercolumn_bwd_kernel<<<numBlocks, threadsPerBlock>>>(bot_pointer + c*offset_bot,
			// 	                                          top_pointer + dst_ch,
      //                                             rand_pointer,
      //                                             pixel_inds, pixel_vals,
      //                                             N_, n_channels_, bottom_nums,
			// 	                                    			width_[b], height_[b], cur_nCh,
			// 	                                    			width_[0], height_[0],
      //                                             padf_[b], (Dtype)poolf_[b]);
      // dst_ch++;
      // CUDA_CHECK(cudaDeviceSynchronize());
      // // TODO: finish implementing this
      // thrust::device_ptr<int> t_pixel_inds = thrust::device_pointer_cast(pixel_inds);
      // thrust::device_ptr<Dtype> t_pixel_vals = thrust::device_pointer_cast(pixel_vals);
      //
      // std::ostringstream output;
      // output << "orig:\n";
      // thrust::copy(t_pixel_inds, t_pixel_inds+1, std::ostream_iterator<int>(output, " "));
      // output << "\n";
      // thrust::copy(t_pixel_vals, t_pixel_vals+1, std::ostream_iterator<Dtype>(output, " "));
      // output << "\n";
      // LOG(ERROR) << output;
      // output << "sorted:\n";
      // try {
      //   thrust::sort_by_key(t_pixel_inds, t_pixel_inds + N_*bottom_nums*4, t_pixel_vals);
      // } catch (thrust::system_error &e) {
      //   LOG(ERROR) << "Thrust Error: " << e.what();
      // }
      // thrust::copy(t_pixel_inds, t_pixel_inds+1, std::ostream_iterator<int>(output, " "));
      // output << "\n";
      // thrust::copy(t_pixel_vals, t_pixel_vals+1, std::ostream_iterator<Dtype>(output, " "));
      // output << "\n";
      // LOG(ERROR) << output;
      // thrust::reduce_by_key(t_pixel_inds,
      //                       t_pixel_inds + N_*bottom_nums*4,
      //                       t_pixel_vals,
      //                       t_pixel_inds + N_*bottom_nums*4,
      //                       t_pixel_vals + N_*bottom_nums*4);
      // CUDA_CHECK(cudaDeviceSynchronize());
      // write_result_kernel<<<numBlocks4, threadsPerBlock4>>>(pixel_inds + N_*bottom_nums*4,
      //                                                     pixel_vals + N_*bottom_nums*4,
      //                                                     bot_pointer + c*offset_bot,
      //                                                     N_*bottom_nums*4);
      // CUDA_CHECK(cudaDeviceSynchronize());
    // }
  }

  // output labels
  top_pointer = top[1]->gpu_diff();
  dst_ch = 0;
  if (propagate_down[end_id_+2]) {
    const int cur_nCh = bottom[end_id_+2]->channels();
    Dtype* bot_pointer = bottom[end_id_+2]->mutable_gpu_diff();
    int offset_bot = width_[0] * height_[0];
		for(int c = 0; c < cur_nCh; c++){
			hypercolumn_bwd_kernel_assign<<<numBlocks, threadsPerBlock>>>(bot_pointer + c*offset_bot,
				                                          top_pointer + dst_ch,
                                                  rand_pointer,
                                                  N_, cur_nCh, bottom_nums,
				                                    			width_[0], height_[0], cur_nCh,
				                                    			width_[0], height_[0],
                                                  (Dtype)0, (Dtype)1);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(HypercolumnLayer);

} // namespace caffe

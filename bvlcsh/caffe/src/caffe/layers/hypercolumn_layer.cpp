#include <vector>
#include <math.h>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/layers/hypercolumn_layer.hpp"
#include <iostream>
#include <random>
#include <ctime>

namespace caffe {

template <typename Dtype>
void HypercolumnLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

    params_ = this->layer_param_.hypercolumn_param();

    // hard-coded start position and end position
    start_id_ = 0;
    n_hblobs_ = bottom.size() - 2;
    end_id_ = n_hblobs_ - 1;

    // if random sampling is required else choose all the points --
    if(params_.rand_selection()){
      N_ = params_.num_output();
      randengine_ = std::default_random_engine(1773);
    } else {
      N_ = (bottom[start_id_]->height() - 2*params_.pad_factor())*
      (bottom[start_id_]->width() - 2*params_.pad_factor());
    }

    // get the pooling factor for the conv-layers
    // and their corresponding padding requirements --
    poolf_ = std::vector<int>(n_hblobs_);
    padf_ = std::vector<Dtype>(n_hblobs_);
    CHECK_EQ(params_.pooling_factor_size(), n_hblobs_);
    for (int i = 0; i < n_hblobs_; i++) {
      poolf_[i] = params_.pooling_factor(i);
      padf_[i] = static_cast<Dtype>((poolf_[i] - 1.0)/2);
    }
  }

template <typename Dtype>
void HypercolumnLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

    if (!params_.rand_selection()) {
      N_ = (bottom[start_id_]->height() - 2*params_.pad_factor())*
        (bottom[start_id_]->width() - 2*params_.pad_factor());
        // LOG(INFO) << "Full: " << bottom[start_id_]->width() << "x" << bottom[start_id_]->height();
    }

    // set the top-layer to be nxc --
    vector<int> top_shape(2);
    top_shape[0] = N_*(bottom[start_id_]->num());

    // Workspace holds rand_points_, num*N_ x 2 (x,y)
    vector<int> workspace_shape;
    workspace_shape.push_back(top_shape[0]);
    workspace_shape.push_back(2);
    workspace.Reshape(workspace_shape);

    // compute num-channels for the given bottom-data
    n_channels_ = 0;
    height_ = std::vector<int>(n_hblobs_);
    width_ = std::vector<int>(n_hblobs_);
    pixels_ = std::vector<int>(n_hblobs_);
    for (int i = 0; i < n_hblobs_; i++) {
      n_channels_ = n_channels_ + bottom[i]->channels();
      height_[i] = bottom[i]->height();
      width_[i] = bottom[i]->width();
      pixels_[i] = height_[i] * width_[i];
      CHECK_LE(bottom[start_id_]->num(), bottom[i]->num());
    }
    CHECK_EQ(bottom[start_id_]->num(), bottom[end_id_+1]->num());
    CHECK_EQ(bottom[start_id_]->width(), bottom[end_id_+1]->width());
    CHECK_EQ(bottom[start_id_]->height(), bottom[end_id_+1]->height());
    top_shape[1] = n_channels_;
    top[0]->Reshape(top_shape);

    top_shape[1] = bottom[end_id_+2]->channels();
    top[1]->Reshape(top_shape);
  }

template<typename Dtype>
void HypercolumnLayer<Dtype>::SetupRandPoints(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {

  // At training time, randomly select N_ points per image
  // At test time, take all the points in the image (1 image at test time)
  const Dtype* valid_data = bottom[end_id_+1]->cpu_data();

  rand_points_.clear();
  if (params_.rand_selection()) {
    // generate the list of points --
    const int num_data_points = (bottom[start_id_]->height())*(bottom[start_id_]->width());
    std::vector<int> shuffle_data_points;
    shuffle_data_points.reserve(num_data_points);
    for(int i=0;i<num_data_points;i++){
      shuffle_data_points.push_back(i);
    }
    int nums = bottom[start_id_]->num();
    for(int n=0;n<nums;n++) {
      // randengine_ = std::default_random_engine(n); // TODO: REMOVE!
      // std::random_shuffle(shuffle_data_points.begin(), shuffle_data_points.end(), [this](int i) {return randengine_()%i;});
      int cnt = 0;
      // shuffle the points in the image --
      //
      // find the N-valid-points from a image --
      for(int j=0;j<num_data_points;j++){
        int j_pt = shuffle_data_points[j];
        Dtype data_pt = valid_data[j_pt];
        if(data_pt > (Dtype)0.5 || 1) { // TODO: REMOVE!
          rand_points_.push_back( j_pt % (bottom[start_id_]->width()) );
          rand_points_.push_back( j_pt / (bottom[start_id_]->width()) );
          cnt++;
          if (cnt>=N_) break;
        }
      }
      if (cnt<N_) {
        LOG(INFO) << "Image needs invalid points: " << N_ << " vs " << cnt;
        for(int j=0; j<num_data_points; j++) {
          int j_pt = shuffle_data_points[j];
          rand_points_.push_back( j_pt % (bottom[start_id_]->width()) );
          rand_points_.push_back( j_pt / (bottom[start_id_]->width()) );
          cnt++;
          if (cnt>=N_) break;
        }
      }
    }
    CHECK_EQ(rand_points_.size(), N_*bottom[start_id_]->num()*2);
  } else {
    // Full image mode
    // LOG(INFO) << "Full: " << bottom[start_id_]->width() << "x" << bottom[start_id_]->height();
    // considering all the data points are considered --
    for (int n=0;n<(bottom[start_id_]->num());n++) {
      for (int j=0;j<(bottom[start_id_]->height()*bottom[start_id_]->width());j++) {
          rand_points_.push_back( j % (bottom[start_id_]->width()) );
          rand_points_.push_back( j / (bottom[start_id_]->width()) );
      }
    }
  }
}

template <typename Dtype>
void HypercolumnLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {

    SetupRandPoints(bottom, top);
    //
    const int bottom_width = bottom[start_id_]->width();
    const int bottom_height = bottom[start_id_]->height();
    const int bottom_nums = bottom[start_id_]->num();
    int sn_channels = bottom[end_id_+2]->channels();

    // get the data --
    std::vector<const Dtype*> bottom_layers(n_hblobs_);
    for (int i = 0; i < n_hblobs_; i++) {
      bottom_layers[i] = bottom[i]->cpu_data();
    }
    const Dtype* sn_data = bottom[end_id_+2]->cpu_data();

    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* top_sn = top[1]->mutable_cpu_data();

    CHECK_EQ(N_*bottom_nums*2, rand_points_.size());
    // get the hypercolumn features for the selected points --
    // #pragma omp parallel for num_threads(4)
    for(int i=0; i < N_*bottom_nums; i++) {
      int s = i*n_channels_;
      int s_sn = i*sn_channels;
      const int n = i/N_;
      DCHECK_LE(n, bottom_nums);
      const int x_pt = rand_points_[i*2+0];
      const int y_pt = rand_points_[i*2+1];
      // then find the corresponding locations
      for (int b = 0; b < n_hblobs_; b++) {
        const int ch = bottom[b]->channels();
        const Dtype tx = (x_pt-padf_[b])/poolf_[b];
        const Dtype ty = (y_pt-padf_[b])/poolf_[b];

        int tx1 = (int)floor(tx);
        int ty1 = (int)floor(ty);
        int tx2 = tx1+1;
        int ty2 = ty1+1;

        // check if they are within the size limit
        tx1 = tx1<0 ? 0 : (tx1<width_[b] ? tx1 : width_[b]-1);
        tx2 = tx2<0 ? 0 : (tx2<width_[b] ? tx2 : width_[b]-1);
        ty1 = ty1<0 ? 0 : (ty1<height_[b] ? ty1 : height_[b]-1);
        ty2 = ty2<0 ? 0 : (ty2<height_[b] ? ty2 : height_[b]-1);

        Dtype rx = tx - tx1;
        Dtype ry = ty - ty1;
        int p11 = n * ch * pixels_[b] + ty1 * width_[b] + tx1;
        int p12 = n * ch * pixels_[b] + ty1 * width_[b] + tx2;
        int p21 = n * ch * pixels_[b] + ty2 * width_[b] + tx1;
        int p22 = n * ch * pixels_[b] + ty2 * width_[b] + tx2;
        typedef Dtype D;
        for (int c = 0; c < ch; c++) {
          top_data[s++] = bottom_layers[b][p11] * (D(1)-ry) * (D(1)-rx)
                        + bottom_layers[b][p21] * ry * (D(1)-rx)
                        + bottom_layers[b][p12] * (D(1)-ry) * rx
                        + bottom_layers[b][p22] * ry * rx;
          p11 += pixels_[b];
          p12 += pixels_[b];
          p21 += pixels_[b];
          p22 += pixels_[b];
        }
      }

      // output labels
      for(int bc = 0; bc < sn_channels; bc++){
        int init_sn = n*sn_channels*bottom_width*bottom_height +
          bc*bottom_width*bottom_height + y_pt*bottom_width + x_pt;
        top_sn[s_sn++] = sn_data[init_sn];
      }
    }
  }

template <typename Dtype>
void HypercolumnLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {

  const int bottom_width = bottom[start_id_]->width();
  const int bottom_height = bottom[start_id_]->height();
  int sn_channels = bottom[end_id_+2]->channels();

  const int bottom_nums = bottom[start_id_]->num();
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_sn_diff = top[1]->cpu_diff();

  std::vector<Dtype*> bottom_layers(bottom.size());
  for (int b = 0; b < bottom.size(); b++) {
    if (propagate_down[b]) {
      bottom_layers[b] = bottom[b]->mutable_cpu_diff();
      caffe_set(bottom[b]->count(), (Dtype)0, bottom_layers[b]);
    }
  }

  // back-propagate to the layers --
  // #pragma omp parallel for num_threads(4)
  for(int i=0; i < N_*bottom_nums; i++) {
    int s = i*n_channels_;
    int s_sn = i*sn_channels;

    const int n = i/N_;
    const int x_pt = rand_points_[i*2+0];
    const int y_pt = rand_points_[i*2+1];

    // then find the corresponding locations
    for (int b = 0; b < n_hblobs_; b++) {
      if (!propagate_down[b]) continue;
      const int ch = bottom[b]->channels();
      const Dtype tx = (x_pt-padf_[b])/poolf_[b];
      const Dtype ty = (y_pt-padf_[b])/poolf_[b];

      int tx1 = (int)floor(tx);
      int ty1 = (int)floor(ty);
      int tx2 = tx1+1;
      int ty2 = ty1+1;

      // check if they are within the size limit
      tx1 = tx1<0 ? 0 : (tx1<width_[b] ? tx1 : width_[b]-1);
      tx2 = tx2<0 ? 0 : (tx2<width_[b] ? tx2 : width_[b]-1);
      ty1 = ty1<0 ? 0 : (ty1<height_[b] ? ty1 : height_[b]-1);
      ty2 = ty2<0 ? 0 : (ty2<height_[b] ? ty2 : height_[b]-1);

      Dtype rx = tx - tx1;
      Dtype ry = ty - ty1;
      int p11 = n * ch * pixels_[b] + ty1 * width_[b] + tx1;
      int p12 = n * ch * pixels_[b] + ty1 * width_[b] + tx2;
      int p21 = n * ch * pixels_[b] + ty2 * width_[b] + tx1;
      int p22 = n * ch * pixels_[b] + ty2 * width_[b] + tx2;
      typedef Dtype D;
      for (int c = 0; c < ch; c++) {
        bottom_layers[b][p11] += top_diff[s] * (D(1)-ry) * (D(1)-rx);
        bottom_layers[b][p21] += top_diff[s] * ry * (D(1)-rx);
        bottom_layers[b][p12] += top_diff[s] * (D(1)-ry) * rx;
        bottom_layers[b][p22] += top_diff[s++] * ry * rx;

        p11 += pixels_[b];
        p12 += pixels_[b];
        p21 += pixels_[b];
        p22 += pixels_[b];
      }

    }

    // output labels
    if (propagate_down[end_id_+2]) {
      for(int bc = 0; bc < sn_channels; bc++){
        int init_sn = n*sn_channels*bottom_width*bottom_height +
          bc*bottom_width*bottom_height + y_pt*bottom_width + x_pt;
          bottom_layers[end_id_+2][init_sn] = top_sn_diff[s_sn++];
      }
    }
  }
}

INSTANTIATE_CLASS(HypercolumnLayer);
REGISTER_LAYER_CLASS(Hypercolumn);

}  // namespace caffe

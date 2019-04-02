#ifndef CAFFE_HYPERCOLUMN_LAYER_HPP_
#define CAFFE_HYPERCOLUMN_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Extract hypercolumn features.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class HypercolumnLayer : public Layer<Dtype> {
 public:
  explicit HypercolumnLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Hypercolumn"; }
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void SetupRandPoints(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 Blob<int> workspace; //only used by gpu
 Blob<int> workspace_inds; //only used by gpu
 Blob<Dtype> workspace_vals; //only used by gpu

 // no. of data points per image
 int N_;
 // number of bottom blobs containing hypercol data
 int n_hblobs_;
 // bottom-blobs start and end ids
 int start_id_;
 int end_id_;
 // number of channels in the hypercol data --
 int n_channels_;
 // points which are randomly selected --
 std::vector<int> rand_points_;

 std::vector<int> poolf_; // pooling factor
 std::vector<Dtype> padf_; // padding factor

 // make a vector of height, width, num_pixels for diff conv.layers --
 std::vector<int> height_, width_, pixels_;
 std::default_random_engine randengine_;
 HypercolumnParameter params_;
};


}  // namespace caffe

#endif  // CAFFE_HYPERCOLUMN_LAYER_HPP_

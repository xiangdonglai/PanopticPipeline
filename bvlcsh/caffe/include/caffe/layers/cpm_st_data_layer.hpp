#ifndef CAFFE_CPM_ST_DATA_LAYER_HPP_
#define CAFFE_CPM_ST_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/cpm_st_data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class CPMSTDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit CPMSTDataLayer(const LayerParameter& param);
  virtual ~CPMSTDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // CPMSTDataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "CPMData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader reader_;
  Blob<Dtype> transformed_label_; // add another blob

  CPMSTTransformationParameter cpm_st_transform_param_;
  shared_ptr<CPMSTDataTransformer<Dtype> > cpm_st_data_transformer_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_

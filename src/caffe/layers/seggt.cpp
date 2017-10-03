#include <algorithm>
#include <vector>

#include "caffe/layers/seggt.hpp"

namespace caffe {

template <typename Dtype>
void SegGtLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
        seggt_param_ = this->layer_param_.seggt_param();
        background_label_id_ = seggt_param_.background_label_id();
        use_difficult_gt_ = seggt_param_.use_difficult_gt();
        num_ = bottom[1]->shape(0);
        height_ = bottom[1]->shape(2);
        width_ = bottom[1]->shape(3);
        num_gt_ = bottom[0]->height();
        top[0]->Reshape(num_, 1, height_, width_);
        min_size = (Dtype *)malloc(num_ * height_ * width_ * sizeof(Dtype));
}

template <typename Dtype>
void SegGtLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top) {
        num_ = bottom[1]->shape(0);
        height_ = bottom[1]->shape(2);
        width_ = bottom[1]->shape(3);
        num_gt_ = bottom[0]->height();
        top[0]->Reshape(num_, 1, height_, width_);
        free(min_size);
        min_size = (Dtype *)malloc(num_ * height_ * width_ * sizeof(Dtype));
}

template <typename Dtype>
void SegGtLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
        const Dtype *gt_data = bottom[0]->cpu_data();
        Dtype *top_data = top[0]->mutable_cpu_data();
        memset(top_data, 0, top[0]->count() * sizeof(Dtype));
        map<int, vector<NormalizedBBox> > all_gt_bboxes;
        map<int, vector<NormalizedBBox> >::iterator all_gt_bboxes_i;
        GetGroundTruth(gt_data, num_gt_, background_label_id_, use_difficult_gt_,
                       &all_gt_bboxes);
        for (int i = 0; i < num_ * height_ * width_; ++i) {
                min_size[i] = 10.0; // All size should <= 1.0, so 10.0 here means INF
        }
        for (int i = 0; i < num_; ++i) {
                LOG(INFO)<<"Image "<<i;
                int shift1 = i * height_ * width_;
                all_gt_bboxes_i = all_gt_bboxes.find(i);
                if (all_gt_bboxes_i == all_gt_bboxes.end()) {
                  LOG(INFO)<<"continued "<<i;
                        continue;
                }
                const vector<NormalizedBBox> gt_bboxes = all_gt_bboxes_i->second;
                for (int j = 0; j < gt_bboxes.size(); ++j) {
                        LOG(INFO)<<"BBOX "<<j;
                        const NormalizedBBox &gt_bbox = gt_bboxes[j];
                        Dtype xmin, ymin, xmax, ymax, label;
                        int xmin_idx, ymin_idx, xmax_idx, ymax_idx;
                        xmin = gt_bbox.xmin();
                        ymin = gt_bbox.ymin();
                        xmax = gt_bbox.xmax();
                        ymax = gt_bbox.ymax();
                        label = gt_bbox.label();
                        xmin_idx = width_ * xmin;
                        ymin_idx = height_ * ymin;
                        xmax_idx = width_ * xmax;
                        ymax_idx = height_ * ymax;
                        CHECK(xmin_idx >= 0);
                        CHECK(ymin_idx >= 0);
                        CHECK(xmax_idx <= width_);
                        CHECK(ymax_idx <= height_);
                        Dtype size = (ymax - ymin) * (xmax - xmin);
                        for (int y_i = ymin_idx; y_i < ymax_idx; ++y_i) {
                                int shift2 = y_i * width_;
                                for (int x_i = xmin_idx; x_i < xmax_idx; ++x_i) {
                                        int idx_ = shift1 + shift2 + x_i;
                                        if (size < min_size[idx_]) {
                                                min_size[idx_] = size;
                                                top_data[idx_] = label;
                                        }
                                }
                        }
                }
        }
}

template <typename Dtype>
void SegGtLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                     const vector<bool> &propagate_down,
                                     const vector<Blob<Dtype> *> &bottom) {
        NOT_IMPLEMENTED; // Do Nothing Here
}

#ifdef CPU_ONLY
STUB_GPU(SegGtLayer);
#endif

INSTANTIATE_CLASS(SegGtLayer);
REGISTER_LAYER_CLASS(SegGt);

} // namespace caffe

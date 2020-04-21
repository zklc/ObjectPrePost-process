#include <algorithm>
#include <vector>
#include <map>
#include <cassert>
#include <cmath>
//#include <cstdlib>
#include <iostream>

#include <cstring>
#include "pre_post_process.h"

namespace pre_post{
const char *YOLO_CLASSES[21] = {
                                  "__background__",
                                  "aeroplane", "bicycle", "bird", "boat",
                                  "bottle", "bus", "car", "cat", "chair",
                                  "cow", "diningtable", "dog", "horse",
                                  "motorbike", "person", "pottedplant",
                                  "sheep", "sofa", "train", "tvmonitor"
};

void pre_process_image(const std::vector<unsigned char>& in_img,
                       const int width, const int height,
                       const int channel, //assert channel == 4
                       const std::vector<int>& mean, //ABGR, assert size==4
                       std::vector<char>& out_img){
  assert(4 == channel);
  assert(4 == mean.size());
  assert(width * height * channel == in_img.size());

  out_img.clear();

  // int mean_a = mean.at(0);
  // int mean_b = mean.at(1);
  // int mean_g = mean.at(2);
  // int mean_r = mean.at(3);
  for(int h = 0; h < height; ++h)
    for(int w = 0; w < width; ++w)
      for(int c = 0; c < channel; ++c) {
        int addr = h * width * channel + w * channel + c;
        const float in_value = static_cast<float>(in_img.at(addr));
        //float out_value = std::floor(in_value - mean.at(c));
        float out_value = std::floor(in_value - 128.0); //fix 128 now
        assert(out_value >= -128.0 && out_value <= 127.0);
        out_img.push_back(static_cast<char>(out_value));//sequence
      }
}// pre_process_image


//conversion the number in int domain to number in float domain
//Extract information from TensorRT
  void dummy_out_char_to_float(const char *in, float *out, int size, int blob_index){
    float scale;
    if(blob_index == 0) //large
      scale = 0.18738846480846405;
    else //medium
      scale = 0.15913736820220947;
    for(int i = 0; i < size; i++){
    //TODO: use the scale from TensorRT
    //out[i] = static_cast<float>(in[i] * 127);//dummy
      out[i] = static_cast<float>(in[i] * scale);//dummy
  }

}

inline float sigmoid(float x){
  return 1. / (1. + exp(-x));
}
  void get_region_box(std::vector<float> &b, float* x, //std::vector<float> biases,
                      const float* biases,
                    int n, int index, int i, int j,
                    int lw, int lh,
                    int w, int h, int stride) {

  b.clear();
  b.push_back((i + (x[index + 0 * stride])) / lw);
  b.push_back((j + (x[index + 1 * stride])) / lh);
  b.push_back(exp(x[index + 2 * stride]) * biases[2 * n] / (w));
  b.push_back(exp(x[index + 3 * stride]) * biases[2 * n + 1] / (h));
}

struct PredictionResult{//copy from source code
  float x;
  float y;
  float w;
  float h;
  float objScore;
  float classScore;
  float confidence;
  int classType;


};

void correct_yolo_boxes(PredictionResult &det, int w, int h,
                   int netw, int neth, int relative)
  {
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
      new_w = netw;
      new_h = (h * netw)/w;
    } else {
      new_h = neth;
      new_w = (w * neth)/h;
    }
    PredictionResult &b = det;
    b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
    b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
    b.w *= (float)netw/new_w;
    b.h *= (float)neth/new_h;
    if(!relative){
      b.x *= w;
      b.w *= w;
      b.y *= h;
      b.h *= h;
    }

  }

bool BoxSortDecendScore(const PredictionResult& box1,
                        const PredictionResult& box2) {
    return box1.confidence> box2.confidence;
}


  float overlap(float x1, float w1, float x2, float w2)
  {
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
  }
  float box_intersection(std::vector<float> a, std::vector<float> b)
  {
    float w = overlap(a[0], a[2], b[0], b[2]);
    float h = overlap(a[1], a[3], b[1], b[3]);
    if (w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
  }
  float box_union(std::vector<float> a, std::vector<float> b)
  {
    float i = box_intersection(a, b);
    float u = a[2] * a[3] + b[2] * b[3] - i;
    return u;
  }

  float box_iou(std::vector<float> a, std::vector<float> b)
  {
    return box_intersection(a, b) / box_union(a, b);
  }
void ApplyNms(std::vector< PredictionResult >& boxes,
              std::vector<int>& idxes, float threshold) {
  std::map<int, int> idx_map;
  //std::cout<<__FUNCTION__<<" "<<__LINE__<<std::endl;
  //std::cout<<"boxes.size():"<<boxes.size()<<std::endl;
  for (int i = 0; i < boxes.size() - 1; ++i) {
    if (idx_map.find(i) != idx_map.end()) { //no iou > t now
      continue;
    }
    //std::cout<<__FUNCTION__<<" "<<__LINE__<<std::endl;
    //std::cout<<"i = "<<i<<std::endl;
    for (int j = i + 1; j < boxes.size(); ++j) {
      if (idx_map.find(j) != idx_map.end()) { // no iou > t now
        continue;
      }
      std::vector<float> Bbox1, Bbox2;
      Bbox1.push_back(boxes[i].x);
      Bbox1.push_back(boxes[i].y);
      Bbox1.push_back(boxes[i].w);
      Bbox1.push_back(boxes[i].h);

      Bbox2.push_back(boxes[j].x);
      Bbox2.push_back(boxes[j].y);
      Bbox2.push_back(boxes[j].w);
      Bbox2.push_back(boxes[j].h);

      float iou = box_iou(Bbox1, Bbox2);
      if (iou >= threshold) {
        idx_map[j] = 1;//tag the iou
      }

    }
  }
  //std::cout<<__FUNCTION__<<" "<<__LINE__<<std::endl;
  for (int i = 0; i < boxes.size(); ++i) {
    if (idx_map.find(i) == idx_map.end()) {
      idxes.push_back(i);//no iou, real result
    }
  }
}

  //post process the results from FPGA
//Apply nms and do some conversion
//fpga_out.size() == 2
//fpga_out.at(0).size == 1 * 75 * 13 * 13
//fpga_out.at(1).size == 1 * 75 * 26 * 26
//element of preds: if prediction.class_label == 0, then no object of this element
void post_process(const std::vector<std::vector<char>> &fpga_out,
                  std::vector<prediction> &output_preds
                  ){

  output_preds.clear();
  //the paramter from yolov3_detection_output_layer<down>
  const int len = 4 + 20 + 1;
  const int side_w_array [2] = {13, 26};
  const int side_h_array[2] = {13, 26};
  const int anchors_scale_ [2] = {32, 16};
  std::vector<PredictionResult> predicts_;
  const int num_class_ = 20;
  int side_w_, side_h_;
  float *swap_, *swap_data;
  const int gaussian_box_ = 0;
  const float confidence_threshold_ = 0.01;
  const int groups_num_ = 3;
  const float biases_[12] = {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
  const float mask_[6] = {3, 4, 5, 0, 1, 2};
  const float nms_threshold_ = 0.45;
  //the paramter from yolov3_detection_output_layer<up>
  //std::cout<<__FUNCTION__<<" "<<__LINE__<<std::endl;
  //if (Caffe::mode() == Caffe::CPU) {
  {
    int mask_offset = 0;
    predicts_.clear();
    float *class_score = new float[num_class_];

    //for (int t = 0; t < bottom.size(); t++) {
    for(int t = 0; t < 2; t++){
      // side_w_ = bottom[t]->width();
      // side_h_ = bottom[t]->height();
      side_w_ = side_w_array[t];
      side_h_ = side_h_array[t];
      int stride = side_w_*side_h_;
      //swap_.ReshapeLike(*bottom[t]);  //1 * 75 * side_w_ * side_h_
      //Dtype* swap_data = swap_.mutable_cpu_data();
      //in fact , do no need so much.
      //keep the same size only to simplify the code
      swap_data = swap_ = (float*)malloc(sizeof(float) * 1 * 75 * side_w_ * side_h_);
      assert(swap_data != NULL);
      //      const Dtype* input_data = bottom[t]->cpu_data();
      const char *input_data_raw = fpga_out.at(t).data(); //from the fpga, so char
      float *input_data = (float*)malloc(sizeof(float) * 1 * 75 * side_w_ * side_h_);
      assert(input_data != NULL);
      dummy_out_char_to_float(input_data_raw, input_data,
                              1 * 75 * side_w_ * side_h_, t);
      // int nw = side_w_*anchors_scale_[t];
      // int nh = side_w_*anchors_scale_[t]; // the source code error
      int nw = side_w_ * anchors_scale_[t];
      int nh = side_h_ * anchors_scale_[t];
      for(int b = 0; b < 1; b++){ //only for visual compatiblity
        for (int s = 0; s < side_w_*side_h_; s++) {
          for(int n = 0; n < 3; n++) {//box index
            int index = n * len * stride + s;//see index2 for a full meaning
            std::vector<float> pred;
            for (int c = 0; c < len; ++c) {
              int index2 = c * stride + index;
              //LOG(INFO)<<index2;
              if(gaussian_box_) {
                if (c == 4 || c == 6) {
                  swap_data[index2] = (input_data[index2 + 0]);
                }
                else {
                  if (c > 7) {
                    //LOG(INFO) << c - 5;
                    class_score[c - 8] = sigmoid(input_data[index2 + 0]);
                  }
                  else {
                    swap_data[index2] = sigmoid(input_data[index2 + 0]);
                  }
                }
              }
              else {//go this
                if (c == 2 || c == 3) {
                  swap_data[index2] = (input_data[index2 + 0]);
                }
                else {
                  if (c > 4) {// 5-->24
                    //LOG(INFO) << c - 5;
                    class_score[c - 5] = sigmoid(input_data[index2 + 0]);
                  }
                  else {// 0, 1
                    swap_data[index2] = sigmoid(input_data[index2 + 0]);
                  }
                }
              }
            }
            int y2 = s / side_w_;
            int x2 = s % side_w_;
            float obj_score;
            if(gaussian_box_) {
              float uc_ver = 4.0 - swap_data[index + 1 * stride] - swap_data[index + 3 * stride] - swap_data[index + 5 * stride] - swap_data[index + 7 * stride];
              obj_score = swap_data[index + 8 * stride] * uc_ver/4.0;
            }
            else {//go this
              obj_score = swap_data[index + 4 * stride];
            }
            PredictionResult predict;
            for (int c = 0; c < num_class_; ++c) {
              class_score[c] *= obj_score;//Pr(Class|Object) * Pr(Object) = Pr(Class)
              //LOG(INFO) << class_score[c];
              if (class_score[c] > confidence_threshold_)
              {
                if(gaussian_box_) {
                  // get_gaussian_yolo_box(pred, swap_data, biases_, mask_[n + mask_offset], index, x2, y2, side_w_, side_h_, nw, nh, stride);
                  ;
                }
                else {
                  get_region_box(pred, swap_data, biases_, mask_[n + mask_offset], index, x2, y2, side_w_, side_h_, nw, nh, stride);
                }

                predict.x = pred[0];
                predict.y = pred[1];
                predict.w = pred[2];
                predict.h = pred[3];
                predict.classType = c;
                predict.confidence = class_score[c];
                correct_yolo_boxes(predict,side_w_,side_h_,nw,nh,1);
                predicts_.push_back(predict);
                //preds.push_back(predict)
              }//class_score[c] > confidence_threshold_
            }// c < num_class_
          }//n < num_
        }//s < side_w_*side_h_
      }// b < 1
      mask_offset += groups_num_;
      free(swap_data);
      free(input_data);
    }

    delete[] class_score;
  }
  std::sort(predicts_.begin(), predicts_.end(),
            BoxSortDecendScore);
  //std::cout<<__FUNCTION__<<" "<<__LINE__<<std::endl;
  //std::cout<<"predicts_.size():"<<predicts_.size()<<std::endl;
  std::vector<int> idxes;
  int num_kept = 0;
  if(predicts_.size() > 0){
    //std::cout<<__FUNCTION__<<" "<<__LINE__<<std::endl;
    //LOG(INFO) << predicts.size();
    ApplyNms(predicts_, idxes, nms_threshold_);
    num_kept = idxes.size();
    //LOG(INFO) << num_kept;

  }


  //put result into the output_preds
  if(num_kept == 0){
    output_preds.clear();//no result
    //std::cout<<__FUNCTION__<<" "<<__LINE__<<std::endl;

  }else {
    //std::cout<<__FUNCTION__<<" "<<__LINE__<<std::endl;
    for(int ii = 0; ii < num_kept; ii++){
      //skip image_id
      prediction tp;
      tp.class_label = predicts_[idxes[ii]].classType + 1;
      tp.confidence = predicts_[idxes[ii]].confidence;
      float left = (predicts_[idxes[ii]].x - predicts_[idxes[ii]].w / 2.);
      float right = (predicts_[idxes[ii]].x + predicts_[idxes[ii]].w / 2.);
      float top = (predicts_[idxes[ii]].y - predicts_[idxes[ii]].h / 2.);
      float bot = (predicts_[idxes[ii]].y + predicts_[idxes[ii]].h / 2.);
      tp.left = left;
      tp.right = right;
      tp.top = top;
      tp.bot = bot;

      output_preds.push_back(tp);
    }
    //std::cout<<__FUNCTION__<<" "<<__LINE__<<std::endl;
    //std::cout<<"output_preds.size():"<<output_preds.size()<<std::endl;
    //std::cout<<"output_preds.at(154).right:"<<output_preds.at(154).right<<std::endl;
  }

}//post_process


  //for test
void post_process_float(const std::vector<std::vector<float>>& fpga_out,
                  std::vector<prediction>& output_preds
                  ){

  output_preds.clear();
  //the paramter from yolov3_detection_output_layer<down>
  const int len = 4 + 20 + 1;
  const int side_w_array [2] = {13, 26};
  const int side_h_array[2] = {13, 26};
  const int anchors_scale_ [2] = {32, 16};
  std::vector<PredictionResult> predicts_;
  const int num_class_ = 20;
  int side_w_, side_h_;
  float *swap_, *swap_data;
  const int gaussian_box_ = 0;
  const float confidence_threshold_ = 0.01;
  const int groups_num_ = 3;
  const float biases_[12] = {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
  const float mask_[6] = {3, 4, 5, 0, 1, 2};
  const float nms_threshold_ = 0.45;
  //the paramter from yolov3_detection_output_layer<up>

  //if (Caffe::mode() == Caffe::CPU) {
  {
    int mask_offset = 0;
    predicts_.clear();
    float *class_score = new float[num_class_];

    //for (int t = 0; t < bottom.size(); t++) {
    for(int t = 0; t < 2; t++){
      // side_w_ = bottom[t]->width();
      // side_h_ = bottom[t]->height();
      side_w_ = side_w_array[t];
      side_h_ = side_h_array[t];
      int stride = side_w_*side_h_;
      //swap_.ReshapeLike(*bottom[t]);  //1 * 75 * side_w_ * side_h_
      //Dtype* swap_data = swap_.mutable_cpu_data();
      //in fact , do no need so much.
      //keep the same size only to simplify the code
      swap_data = swap_ = (float*)malloc(sizeof(float) * 1 * 75 * side_w_ * side_h_);
      assert(swap_data != NULL);
      //      const Dtype* input_data = bottom[t]->cpu_data();
      // const char *input_data_raw = fpga_out.at(t).data(); //from the fpga, so char
      // float *input_data = (float*)malloc(sizeof(float) * 1 * 75 * side_w_ * side_h_);
      // dummy_out_char_to_float(input_data_raw, input_data,
      //                         1 * 75 * side_w_ * side_h_);
      const float *input_data = fpga_out.at(t).data();
      // int nw = side_w_*anchors_scale_[t];
      // int nh = side_w_*anchors_scale_[t]; // the source code error
      int nw = side_w_ * anchors_scale_[t];
      int nh = side_h_ * anchors_scale_[t];
      for(int b = 0; b < 1; b++){ //only for visual compatiblity
        for (int s = 0; s < side_w_*side_h_; s++) {
          for(int n = 0; n < 3; n++) {//box index
            int index = n * len * stride + s;//see index2 for a full meaning
            std::vector<float> pred;
            for (int c = 0; c < len; ++c) {
              int index2 = c * stride + index;
              //LOG(INFO)<<index2;
              if(gaussian_box_) {
                if (c == 4 || c == 6) {
                  swap_data[index2] = (input_data[index2 + 0]);
                }
                else {
                  if (c > 7) {
                    //LOG(INFO) << c - 5;
                    class_score[c - 8] = sigmoid(input_data[index2 + 0]);
                  }
                  else {
                    swap_data[index2] = sigmoid(input_data[index2 + 0]);
                  }
                }
              }
              else {//go this
                if (c == 2 || c == 3) {
                  swap_data[index2] = (input_data[index2 + 0]);
                }
                else {
                  if (c > 4) {// 5-->24
                    //LOG(INFO) << c - 5;
                    class_score[c - 5] = sigmoid(input_data[index2 + 0]);
                  }
                  else {// 0, 1
                    swap_data[index2] = sigmoid(input_data[index2 + 0]);
                  }
                }
              }
            }
            int y2 = s / side_w_;
            int x2 = s % side_w_;
            float obj_score;
            if(gaussian_box_) {
              float uc_ver = 4.0 - swap_data[index + 1 * stride] - swap_data[index + 3 * stride] - swap_data[index + 5 * stride] - swap_data[index + 7 * stride];
              obj_score = swap_data[index + 8 * stride] * uc_ver/4.0;
            }
            else {//go this
              obj_score = swap_data[index + 4 * stride];
            }
            PredictionResult predict;
            for (int c = 0; c < num_class_; ++c) {
              class_score[c] *= obj_score;//Pr(Class|Object) * Pr(Object) = Pr(Class)
              //LOG(INFO) << class_score[c];
              if (class_score[c] > confidence_threshold_)
              {
                if(gaussian_box_) {
                  // get_gaussian_yolo_box(pred, swap_data, biases_, mask_[n + mask_offset], index, x2, y2, side_w_, side_h_, nw, nh, stride);
                  ;
                }
                else {
                  get_region_box(pred, swap_data, biases_, mask_[n + mask_offset], index, x2, y2, side_w_, side_h_, nw, nh, stride);
                }

                predict.x = pred[0];
                predict.y = pred[1];
                predict.w = pred[2];
                predict.h = pred[3];
                predict.classType = c;
                predict.confidence = class_score[c];
                correct_yolo_boxes(predict,side_w_,side_h_,nw,nh,1);
                predicts_.push_back(predict);
                //preds.push_back(predict)
              }//class_score[c] > confidence_threshold_
            }// c < num_class_
          }//n < num_
        }//s < side_w_*side_h_
      }// b < 1
      mask_offset += groups_num_;

    }

    delete[] class_score;
  }

  std::sort(predicts_.begin(), predicts_.end(),
            BoxSortDecendScore);
  std::vector<int> idxes;
  int num_kept = 0;
  if(predicts_.size() > 0){
    //LOG(INFO) << predicts.size();
    ApplyNms(predicts_, idxes, nms_threshold_);
    num_kept = idxes.size();
    //LOG(INFO) << num_kept;

  }


  //put result into the output_preds
  if(num_kept == 0){
    output_preds.clear();//no result


  }else {
    for(int ii = 0; ii < num_kept; ii++){
      //skip image_id
      prediction tp;
      tp.class_label = predicts_[idxes[ii]].classType + 1;
      tp.confidence = predicts_[idxes[ii]].confidence;
      float left = (predicts_[idxes[ii]].x - predicts_[idxes[ii]].w / 2.);
      float right = (predicts_[idxes[ii]].x + predicts_[idxes[ii]].w / 2.);
      float top = (predicts_[idxes[ii]].y - predicts_[idxes[ii]].h / 2.);
      float bot = (predicts_[idxes[ii]].y + predicts_[idxes[ii]].h / 2.);
      tp.left = left;
      tp.right = right;
      tp.top = top;
      tp.bot = bot;

      output_preds.push_back(tp);
    }

  }

}//post_process


  //for gao laoban
  //please resize the image to the input size of FPGA before call this function
  //use for pre_process the image
  //img:H-->W-->C, ABGR planar
  //now, the mean's element always equals 128
  //---20200410 sunlaoban <down>
  // in case of memory leak, the function should first check if(*out_img != NULL),
  // then  free the previous memory.
  // so the caller should first call with *out_img == NULL,
  // next call with *out_img == previous_pointer_value
  //---20200410 sunlaoban <up>
  void pre_process_image_for_csharp(//const std::vector<unsigned char>& in_img,
                                    const unsigned char *in_img,
                                    const int *in_img_size,
                                    const int width, const int height,
                                    const int channel, //assert channel == 4
                                    //                       const std::vector<int>& mean, //ABGR, assert size==4
                                    const int *mean, //size == channel
                                    //std::vector<char>& out_img

                                    //output
                                    char **out_img,//allocate by this function
                                    int *out_img_size
                                    ){
    //sun laoban <dowm>
    free(*out_img);// if *out_img == NULL, this will not have any effect.
    //sun laoban <up>

    std::vector<unsigned char> in_img_0(in_img, in_img + (*in_img_size));
    std::vector<int> mean_0(mean, mean + channel);
    std::vector<char> out_img0;
    //std::cout<<__FUNCTION__<<" "<<__LINE__<<std::endl;

    pre_process_image(in_img_0,
                      width,  height,
                      channel, //assert channel == 4
                      //const std::vector<int>& mean, //ABGR, assert size==4
                      mean_0,
                      out_img0);
    *out_img = (char*)malloc(out_img0.size());
    assert((*out_img) != NULL);
    std::memcpy(*out_img, out_img0.data(), out_img0.size() );
    //    std::cout<<__FUNCTION__<<" "<<__LINE__<<" out_img0.size():"<<out_img0.size()<<std::endl;
    *out_img_size = out_img0.size();
  }


  void translate_cube_to_normal(int width, int height, int channel, int atom_channel,
                                const char *input, char *output){

    int src_index, dst_index;
    //transfer position and to float
    for(int c = 0; c < channel; c++){
      for(int h = 0; h < height; h++){
        for(int w = 0; w < width; w++){
          dst_index = c * height * width
            + h * width + w;
          int surface_index = c/atom_channel;
          src_index = surface_index * (atom_channel * width * height)
            + h * (atom_channel * width) //offset in ont SURF
            + w * atom_channel//offset in one line
            + c % atom_channel;//offset in one atom
          output[dst_index] = input[src_index];

        }//w
      }//h
    }//c


  }
  //for gaolaoban
  //post process the results from FPGA
  //Apply nms and do some conversion
  //fpga_out: 1 * 96 * 13 * 13 + 1 * 96 * 26 * 26
  //fpga_out.at(0).size == 1 * 96 * 13 * 13
  //fpga_out.at(1).size == 1 * 96 * 26 * 26
  //element of preds: if prediction.class_label == 0, then no object of this element
  //---20200410 sunlaoban <down>
  // in case of memory leak, the function should first check if(*preds != NULL),
  // then free the previous memory.
  // so the caller should first call with *preds == NULL,
  // next call with *preds == previous_pointer_value
  //---20200410 sunlaoban <up>
  //20200416 change fpga_out size
  //1 * 96 * 13 *13 * 5 all
  void post_process_for_csharp(//const std::vector<std::vector<char>>& fpga_out,
                               const char *fpga_out,
                               const int *fpga_out_size,


                               //std::vector<prediction>& preds
                               prediction **preds,//allocate by this function
                               int *preds_size
                               ){

    //translate nvdla cube to normal memory <down>
    char fpga_large[96 * 13 * 13];
    const char *raw_fpga_large_cube = fpga_out;
    char fpga_medium[96 * 26 * 26];
    const char *raw_fpga_medium_cube = fpga_out + 96 * 13 * 13;

    translate_cube_to_normal(13, 13, 96, 32, raw_fpga_large_cube, fpga_large);
    translate_cube_to_normal(26, 26, 96, 32, raw_fpga_medium_cube, fpga_medium);
    //translate nvdla cube to normal memory <up>

    //sun laoban <down>
    free(*preds);// if *preds == NULL, this will not have any effect.
    //sun laoban <up>
    // std::vector<char> fpga_out_e0(fpga_out, fpga_out + 75*13*13);
    // std::vector<char> fpga_out_e1(fpga_out + 75*13*13,
    //                               fpga_out + 75*13*13 + 75*26*26);
    std::vector<char> fpga_out_e0(fpga_large, fpga_large + 75*13*13);
    std::vector<char> fpga_out_e1(fpga_medium,
                                  fpga_medium + 75*26*26);

    // //std::cout<<__FUNCTION__<<" "<<__LINE__<<std::endl;

    std::vector<std::vector<char>> fpga_out0{fpga_out_e0, fpga_out_e1};

    std::vector<prediction> preds0;
    post_process(fpga_out0,
                 preds0
                 );
    //    std::cout<<"preds0.at(0).right:"<<preds0.at(154).right<<std::endl;
    //std::cout<<__FUNCTION__<<" "<<__LINE__<<std::endl;
    //std::cout<<"preds0.size():"<<preds0.size()<<std::endl;
    //printf("sizeof(prediction):%d\n", sizeof(prediction));
    //printf("preds0.at(preds0.size()-1).right=%f\n",
    //preds0.at(154).right);

    *preds = (prediction*)malloc(sizeof(prediction) * preds0.size() );
    assert((*preds) != NULL);
    //*preds = (prediction*)malloc(100);
    //*preds = new prediction[100];
    //assert(*preds != NULL);
    //char *preds_data =  (char*)preds0.data();
    //printf("preds0.at(preds0.size()-1).right=%f\n",
    //preds0.at(0).right);
    memcpy((char*)(*preds), preds0.data(), sizeof(prediction)*preds0.size());
    //std::cout<<__FUNCTION__<<" "<<__LINE__<<std::endl;

    *preds_size = preds0.size();



  }


}//namespace pre_post

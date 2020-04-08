/**
 * Functions declared in this file are useful for pre and post process the image and result.
 * Assume the input image: pixel value in unsigned char(so the intensity: 0--255), and channels in  ABGR. The scan sequences is from the top to down, left to right.
 * In memory, the image is H-->W-->C.
*/


#include <vector>

#ifndef PRE_POST_PROCESS_H
#define PRE_POST_PROCESS_H


namespace pre_post{



//please resize the image to the input size of FPGA before call this function
//use for pre_process the image
//img:H-->W-->C, ABGR planar
//now, the mean's element always equals 128
void pre_process_image(const std::vector<unsigned char>& in_img,
                       const int width, const int height,
                       const int channel, //assert channel == 4
                       const std::vector<int>& mean, //ABGR, assert size==4
                       std::vector<char>& out_img);








struct prediction{
  int class_label=0; // 0, no object. 1-->21 the 20 pascvol label
  //float class_score=0;
  float confidence;
  float left; // normalized to [0,1]
  float top;
  float right;
  float bot;
};


//char
#define ONE_IMAGE_RESULT_SIZE 75*13*13*5

//post process the results from FPGA
//Apply nms and do some conversion
//fpga_out.size() == 2
//fpga_out.at(0).size == 1 * 75 * 13 * 13
//fpga_out.at(1).size == 1 * 75 * 26 * 26
//element of preds: if prediction.class_label == 0, then no object of this element
void post_process(const std::vector<std::vector<char>>& fpga_out,
                  std::vector<prediction>& preds
                   );



//for gao laoban
//please resize the image to the input size of FPGA before call this function
//use for pre_process the image
//img:H-->W-->C, ABGR planar
//now, the mean's element always equals 128
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
                                  );

//for gaolaoban
//post process the results from FPGA
//Apply nms and do some conversion
//fpga_out: 1 * 75 * 13 * 13 + 1 * 75 * 26 * 26
//fpga_out.at(0).size == 1 * 75 * 13 * 13
//fpga_out.at(1).size == 1 * 75 * 26 * 26
//element of preds: if prediction.class_label == 0, then no object of this element
void post_process_for_csharp(//const std::vector<std::vector<char>>& fpga_out,
                  const char *fpga_out,
                  const int *fpga_out_size,


                  //std::vector<prediction>& preds
                  prediction **preds,//allocate by this function
                  int *preds_size
                  );




} //namespace pre_post
#endif

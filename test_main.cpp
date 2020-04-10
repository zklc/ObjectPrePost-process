#include <iostream>
#include <vector>
#include "pre_post_process.h"

using namespace pre_post;
using namespace std;

int main(int agrc, char **argv){
  prediction p0;
  // cout<<p0.score<<endl;

  const int width = 416, test_ci = 10;
  const int height = 416;
  const int channel = 4;

  vector<unsigned char> in_img(width * height * channel, 10);
  vector<int> mean{70, 80, 90, 100};
  vector<char> out_img;

  pre_process_image( in_img,
                     width,  height,
                     channel, //assert channel == 4
                     mean, //ABGR, assert size==4
                     out_img);

  printf("out_img.size()=%lu\n", out_img.size());
  // for(int i = 0; i < out_img.size()/ 1000; i++){
  //   //printf("0x%2.2x", (unsigned char)(out_img[i]));
  //   printf("%d\t", (out_img[i]));
  //   if(i%16 == 0)
  //     printf("\n");

  // }



  cout<<__LINE__<<endl;

  vector<char> out_img1;
  int out_img1_size;
  int in_img_size = in_img.size();
  char *out_img1_data = NULL;
  pre_process_image_for_csharp(//const std::vector<unsigned char>& in_img,
                               in_img.data(),
                               //const int *in_img_size,
                               &in_img_size,
                               width,  height,
                                channel, //assert channel == 4
                               //                       const std::vector<int>& mean, //ABGR, assert size==4
                               //const int *mean, //size == channel
                               mean.data(),
                               //std::vector<char>& out_img

                               //output
                               //char *out_img,//allocate by this function
                               //                               out_img1.data(),
                               &out_img1_data,
                               &out_img1_size
                               );
  cout<<__LINE__<<endl;


  vector<char> fpga_out0(75*13*13*5, -100);
  for(int i = 0; i < 1000; i++){
    fpga_out0[i]=0;
  }
  int fpga_out0_size = fpga_out0.size();
  prediction *preds_out = NULL;
  int out_preds_size;
  post_process_for_csharp(//const std::vector<std::vector<char>>& fpga_out,
                          //const char *fpga_out,
                          fpga_out0.data(),
                          //const int *fpga_out_size,
                          &fpga_out0_size,


                          //std::vector<prediction>& preds
                          //prediction *preds,//allocate by this function
                          &preds_out,
                          //                     int *preds_size
                          &out_preds_size
                          );
  cout<<__LINE__<<endl;


}

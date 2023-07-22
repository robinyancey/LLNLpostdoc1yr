#include <torch/script.h> 
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <string>
#include <vector>
#include <thread>
using namespace cv;

using namespace cv;
struct pred {
  std::string classification;
  double conf;

  };

void get_preds(pred* curr_pred, int x3bin, double x3abin){
 pred res;
 res.classification = (x3bin == 0) ? "Damage" : "NonDamage";
 res.conf = x3abin;
 *curr_pred = res;

}


void get_tensors(torch::Tensor* out, std::string back_path, std::string top_path, std::string ring_path) {
  std::vector <std::string> curr_im = { back_path, top_path, ring_path };
  Mat channel[3];
  Mat channel_im[3];
  Mat image1;
  const int cropSize = 448;
  for (int k = 0; k < 3; k++) {
    image1 = imread(curr_im[k], IMREAD_UNCHANGED);
    if (image1.empty()){
      std::cout << "no image data in " << curr_im[k] << std::endl;
    }
    if ( image1.channels() != 1 ){
      split(image1, channel);
      channel[0].convertTo(channel[0], CV_8UC1, (1.0/3.0));
      channel[1].convertTo(channel[1], CV_8UC1, (1.0/3.0));
      add(channel[1], channel[0], channel[0]);
      channel[2].convertTo(channel[2], CV_8UC1, (1.0/3.0));
      add(channel[2], channel[0], channel[0]);
      image1 = channel[0];
    }
    channel_im[k] = image1;
  }
  merge(channel_im, 3, image1);
  image1.convertTo(image1, CV_32FC3);// convert to 3d float, model normalizes internally
  const int offsetW = (image1.cols - cropSize) / 2;
  const int offsetH = (image1.rows - cropSize) / 2;
  const Rect roi(offsetW, offsetH, cropSize, cropSize);
  image1 = image1(roi).clone();
  auto x1 = torch::from_blob(image1.data, {image1.rows, image1.cols, 3}, torch::kFloat);// match image type
  x1 = x1.permute({ 2,0,1 });
  *out = x1.unsqueeze(0);

}

torch::Tensor sigmoid_f(torch::Tensor &x)
{    
     auto ret = 1/(1+torch::exp(-x));
     return ret;
     }


std::vector<pred> classify_set_of_images(std::vector <std::string> backlit, std::vector <std::string> toplit, std::vector <std::string> ringlit) {

  torch::jit::script::Module module;
  std::vector<pred> curr_pred_list(toplit.size());
  int batch_size = 128; // can change to 128 after testing etc.
  //if (!(backlit.size() == ringlit.size()) or !(backlit.size() == toplit.size())){
  //  std::cout << "Input vectors have different sizes!" << std::endl;
  //}
  int n_batches = (toplit.size() + (batch_size - 1))/batch_size;
  //std::cout << "n_batches" << n_batches << std::endl;
  try{//modify model path
    module = torch::jit::load("/usr/workspace/yancey5/stuff/code/project/final-model.pt", torch::kCUDA);
    std::cout << "Model loaded!\n";
    module.eval();
    torch::NoGradGuard no_grad;
  }
  catch (const c10::Error& e) {
    std::cout << "Issue Loading Model...: HAS GPU?" << torch::cuda::is_available() << std::endl;
  }

  int start = 0;
  std::thread ths[batch_size];//also modify n threads on Pascal to be used
  std::vector<torch::Tensor> results(batch_size);
  for (int l = 0; l < n_batches; l++){
    //if ((l == n_batches - 1) && (toplit.size() % batch_size != 0)){
      //batch_size =  toplit.size() % batch_size;
    //} 
    for (int id = 0; id < batch_size; id++) {
      if (( start + id ) == toplit.size()) {
        batch_size =  toplit.size() % batch_size;
        break;
      }
      ths[id] = std::thread(get_tensors, &results[id], backlit[start + id], toplit[start + id], ringlit[start + id]);
    }

    for (int id = 0; id < batch_size; id++) {
      ths[id].join();
    }
    auto batch = torch::cat(results);
    //std::cout << "dimensions of batch tensors: " << batch.sizes() << std::endl;
    auto x = batch.to(torch::kCUDA);
    auto x2 = module.forward({x}).toTensor();
    auto x3 = x2.to(torch::kCPU);
    auto x3b = torch::round(sigmoid_f(x3));
    auto x3ab = torch::max(sigmoid_f(x3), (1- sigmoid_f(x3)));

    for (int id = 0; id < batch_size; id++) {
      int xin1 = x3b[id].item<int>();
      double xin2 = x3ab[id].item<double>();
      ths[id] = std::thread(get_preds, &curr_pred_list[start + id], xin1, xin2);
    }
    for (int id = 0; id < batch_size; id++) {
      ths[id].join();
    }


    start = start + batch_size;
  }

  return curr_pred_list;
}

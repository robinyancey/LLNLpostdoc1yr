// input b/t/r paths to directories with all test images to glop filke path strings into vectors for input into main function (cassify_sets_of_images)
// input path to .csv file like Chris provided with path/calls/conf cols 
int test0(std::string B, std::string T, std::string R, std::string truths) {
  std::vector <std::string> backlit;
  std::vector <std::string> toplit;
  std::vector <std::string> ringlit;
  std::string folder;
  
  folder = B;
  std::vector<cv::String> fnB;
  glob(folder, fnB, false); 
  
  folder = T;
  std::vector<cv::String> fnT;
  glob(folder, fnT, false); 
  
  folder = R;
  std::vector<cv::String> fnR;
  glob(folder, fnR, false); 
  
  if (!(fnB.size() == fnR.size()) or !(fnB.size() == fnT.size())){
    std::cout << "Input vectors have different sizes!" << std::endl;
    std::cout << "backlit: " << fnB.size() << std::endl;
    std::cout << "toplit: " << fnT.size() << std::endl;
    std::cout << "ringlit: " << fnR.size() << std::endl;
  }
  
  std::vector <std::string> true_class;
  std::vector <double> true_conf;
  std::string line;    
  std::ifstream data;
  data.open(truths);
  if ( data.is_open() ) {
    std::cout << "Truths file open: " << std::endl;
  }
  while(std::getline(data,line))
  {
    std::stringstream lineStream(line);
    std::string cell;
    while(std::getline(lineStream,cell,','))
      {
      std::getline(lineStream,cell,',');
      true_class.push_back(cell);
      std::getline(lineStream,cell,',');
      true_conf.push_back(std::stod(cell));
      }
  }
  for (size_t i=0; i<fnB.size(); i++){
    backlit.push_back(fnB[i]);
    toplit.push_back(fnT[i]);
    ringlit.push_back(fnR[i]);
  }
  auto res = classify_set_of_images(backlit, toplit, ringlit);
  for (size_t i=0; i<fnB.size(); i++){
      //std::cout << i << " True class: " << true_class[i] << '\n';
      //std::cout << i << " True confidence: " << (double) true_conf[i] << '\n';
      if (res[i].classification != true_class[i]){
        std::cout << i << " Predicted class output: " << res[i].classification << '\n';
        std::cout << i << " Predicted confidence output: " << res[i].conf << '\n';
        std::cout << i << " True class: " << true_class[i] << '\n';
        std::cout << i << " True confidence: " << (double) true_conf[i] << '\n';
        std::cout << "Test didnt pass!!!" << std::endl;
      } 
    }
return 0;
}

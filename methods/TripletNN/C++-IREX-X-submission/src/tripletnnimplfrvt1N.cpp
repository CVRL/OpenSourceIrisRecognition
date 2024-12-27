/*
 * This software was developed at the National Institute of Standards and
 * Technology (NIST) by employees of the Federal Government in the course
 * of their official duties. Pursuant to title 17 Section 105 of the
 * United States Code, this software is not subject to copyright protection
 * and is in the public domain. NIST assumes no responsibility whatsoever for
 * its use by other parties, and makes no guarantees, expressed or implied,
 * about its quality, reliability, or any other characteristic.
 */

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cassert>
#include <functional>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include <chrono>
#include <filesystem>

#include "tripletnnimplfrvt1N.h"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <unistd.h>

using namespace std;
using namespace std::chrono;
using namespace std::chrono_literals;
using namespace FRVT;
using namespace FRVT_1N;

/*
int global_image_count = 0;

cv::Mat get_image_from_tensor(at::Tensor tensor) {
    tensor = tensor.detach();
    tensor = tensor.clamp(0.0, 255.0).to(torch::kU8);
    int64_t height = tensor.size(0);
    int64_t width = tensor.size(1);
    auto img = cv::Mat(height, width, CV_8UC1, tensor.data_ptr());
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR, 3);
    return img;
}

void save_polar_mask(at::Tensor mask_polar) {
  cv::Mat mask_rgb_cv2 = get_image_from_tensor(mask_polar);
  std::stringstream ss;
  ss << std::this_thread::get_id();
  cv::imwrite("/hd2/masks_polar/" + to_string(getpid()) + "_" + ss.str() + "_" + to_string(global_image_count) + ".ppm", mask_rgb_cv2);
}

void save_polar_image(at::Tensor image_polar) {
  cv::Mat image_rgb_cv2 = get_image_from_tensor(image_polar);
  std::stringstream ss;
  ss << std::this_thread::get_id();
  cv::imwrite("/hd2/images_polar/" + to_string(getpid()) + "_" + ss.str() + "_" + to_string(global_image_count) + ".ppm", image_rgb_cv2);

}

void save_mask(at::Tensor mask) {
  cv::Mat mask_rgb_cv2 = get_image_from_tensor(mask);
  std::stringstream ss;
  ss << std::this_thread::get_id();
  cv::imwrite("/hd2/masks/" + to_string(getpid()) + "_" + ss.str() + "_" + to_string(global_image_count) + ".ppm", mask_rgb_cv2);
}

void save_image(cv::Mat irisim, at::Tensor pupil_xyr, at::Tensor iris_xyr) {
  cv::Mat irisim_rgb;
  cv::cvtColor(irisim, irisim_rgb, cv::COLOR_GRAY2BGR, 3);
    
  int thickness = 2;
  cv::Scalar pupil_color(255, 0, 0);
  int px = pupil_xyr[0].item<int>();
  int py = pupil_xyr[1].item<int>();
  cv::Point pc(px, py);
  int pr = pupil_xyr[2].item<int>();
  cv::circle(irisim_rgb, pc, pr, pupil_color, thickness);
  cv::Scalar iris_color(0, 0, 255);
  int ix = iris_xyr[0].item<int>();
  int iy = iris_xyr[1].item<int>();
  cv::Point ic(px, py);
  int ir = iris_xyr[2].item<int>();
  cv::circle(irisim_rgb, ic, ir, iris_color, thickness);
  std::stringstream ss;
  ss << std::this_thread::get_id();
  cv::imwrite("/hd2/images/" + to_string(getpid()) + "_" + ss.str() + "_" + to_string(global_image_count) + ".ppm", irisim_rgb);
}
*/

TripletNNImplFRVT1N::TripletNNImplFRVT1N() {
  resolution.push_back(0);
  resolution.push_back(0);
  norm_params_circle.push_back(0);
  norm_params_circle.push_back(0);
  norm_params_mask.push_back(0);
  norm_params_mask.push_back(0);
  init = false;
  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  cv::setNumThreads(1);
  torch::globalContext().setDeterministicCuDNN(true);
  torch::globalContext().setDeterministicAlgorithms(true, true);
}

TripletNNImplFRVT1N::~TripletNNImplFRVT1N() {
  //delete enable_grad;
}
std::shared_ptr<FRVT_1N::Interface> FRVT_1N::Interface::getImplementation() {
  return std::make_shared<TripletNNImplFRVT1N>();
}

/* Public Interface */

/**
  * @brief Before images are sent to the template
  * creation function, the test harness will call this initialization
  * function.
  * @details This function will be called N=1 times by the NIST application,
  * prior to parallelizing M >= 1 calls to createTemplate() via fork().
  *
  * This function will be called from a single process/thread.
  *
  * @param[in] configDir
  * A read-only directory containing any developer-supplied configuration
  * parameters or run-time data files.
  * @param[in] role
  * A value from the TemplateRole enumeration that indicates the intended
  * usage of the template to be generated.  In this case, either a 1:N
  * enrollment template used for gallery enrollment or 1:N identification
  * template used for search.
  */
ReturnStatus TripletNNImplFRVT1N::initializeTemplateCreation(const std::string &configDir, FRVT::TemplateRole role) {
  if (init == true) {
    //////////cerr << "Already initialized, skipping..." << endl;
    return ReturnCode::Success;
  }
  
  
  //cerr << "Changes are made 2." << endl;
  
  init = true;

  std::string yamlpath = configDir + "/cfg.yaml";
  load_cfg(yamlpath.c_str());

  string vector_model_path = (configDir + "/" + cfg["vector_model_path"]).c_str();
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    vector_model = torch::jit::load(vector_model_path, torch::kCPU);
  } catch (const c10::Error &e) {
    ////////////cerr << "error loading the mask model\n";
    return ReturnCode::ConfigError;
  }
  vector_model.eval();

  string circle_param_model_path = (configDir + "/" + cfg["circle_param_model_path"]).c_str();
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    circle_model = torch::jit::load(circle_param_model_path, torch::kCPU);
  } catch (const c10::Error &e) {
    ////////////cerr << "error loading the circle model\n";
    return ReturnCode::ConfigError;
  }
  circle_model.eval();

  string mask_model_path = (configDir + "/" + cfg["coarse_mask_model_path"]).c_str();
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    mask_model = torch::jit::load(mask_model_path, torch::kCPU);
  } catch (const c10::Error &e) {
    ////////////cerr << "error loading the circle model\n";
    return ReturnCode::ConfigError;
  }
  mask_model.eval();

  polar_height = stoi(cfg["polar_height"]);
  polar_width = stoi(cfg["polar_width"]);

  resolution[0] = stoi(cfg["seg_width"]);
  resolution[1] = stoi(cfg["seg_height"]);

  norm_params_mask[0] = stod(cfg["norm_mean_mask"]);
  norm_params_mask[1] = stod(cfg["norm_std_mask"]);

  norm_params_circle[0] = stod(cfg["norm_mean_circle"]);
  norm_params_circle[1] = stod(cfg["norm_std_circle"]);

  codeSize = stoi(cfg["vector_dim"]);

  min_pupil_radius = stoi(cfg["MINIMUM_PUPIL_RADIUS"]);
  min_iris_radius = stoi(cfg["MINIMUM_IRIS_RADIUS"]);

  //cerr << "initialization worked!" << endl;

  return ReturnCode::Success;
}

ReturnStatus TripletNNImplFRVT1N::createFaceTemplate(
    const std::vector<FRVT::Image> &faces,
    FRVT::TemplateRole role,
    std::vector<uint8_t> &templ,
    std::vector<FRVT::EyePair> &eyeCoordinates
) {
  return ReturnStatus(ReturnCode::NotImplemented);
}

ReturnStatus TripletNNImplFRVT1N::createFaceTemplate(
    const FRVT::Image &image,
    FRVT::TemplateRole role,
    std::vector<std::vector<uint8_t>> &templs,
    std::vector<FRVT::EyePair> &eyeCoordinates
) {
  return ReturnStatus(ReturnCode::NotImplemented);
}

/**
  * @brief This function supports template generation from one or more iris images of
  * exactly one person.  It takes as input a vector of images and outputs a template
  * and optionally, the associated location of the iris in each image.
  *
  * @details For enrollment templates: If the function
  * executes correctly (i.e. returns a successful exit status),
  * the template will be enrolled into a gallery.  The NIST
  * calling application may store the resulting template,
  * concatenate many templates, and pass the result to the enrollment
  * finalization function. 
  *
  * When the implementation fails to produce a
  * template, it shall still return a blank template (which can be zero
  * bytes in length). The template will be included in the
  * enrollment database/manifest like all other enrollment templates, but
  * is not expected to contain any feature information.
  * <br>For identification templates: If the function returns a
  * non-successful return status, the output template will be not be used
  * in subsequent search operations.
  *
  * @param[in] irises
  * A vector of input iris images
  * @param[in] role
  * A value from the TemplateRole enumeration that indicates the intended
  * usage of the template to be generated.  In this case, either a 1:N
  * enrollment template used for gallery enrollment or 1:N identification
  * template used for search.
  * @param[out] templ
  * The output template.  The format is entirely unregulated.  This will be
  * an empty vector when passed into the function, and the implementation can
  * resize and populate it with the appropriate data.
  * @param[out] irisLocations
  * (Optional) The function may choose to return the estimated iris locations 
  * for the input iris images.
  */

int global_index = 0;

FRVT::ReturnStatus TripletNNImplFRVT1N::createIrisTemplate(
    const std::vector<FRVT::Image> &irises,
    FRVT::TemplateRole role,
    std::vector<uint8_t> &templ,
    std::vector<FRVT::IrisAnnulus> &irisLocations
) {
  
  //auto start = high_resolution_clock::now();

  // //cout << "Creating Iris Template: "<< endl;
  torch::AutoGradMode enable_grad(false);
  //cerr << "create iris template got called!" << endl;
  
  //////cerr << irises.size() << endl;

  for (int i = 0; i < irises.size(); i++) {
    const FRVT::Image &iris = irises[i];

    bool isRGB = (iris.depth == 24);
    cv::Mat irisim = get_cv2_image(iris.data, iris.width, iris.height, isRGB);
    cv::Scalar mean,stddev;
    cv::meanStdDev(irisim,mean,stddev);

    //cerr << "std: " << stddev.val[0] << endl;
    if (stddev.val[0] <= 10.0) {
        continue;
    }

    //if(irisim.empty())
    //{
      //////cerr << "irisim.empty() 1" << endl;
    //}

    this->fix_image(irisim);
    //cerr << "Fix image worked!" << endl;

    //if(irisim.empty())
    //{
      //////cerr << "irisim.empty() 2" << endl;
    //}
    
    //cerr << "Before segmentation" << endl;
    map<string, at::Tensor>* seg_im = new map<string, at::Tensor>;
    this->segment_and_circApprox(irisim.clone(), seg_im);
    double ix = (*seg_im)["iris_xyr"].index({0}).item<double>();
    double iy = (*seg_im)["iris_xyr"].index({1}).item<double>();
    double ir = (*seg_im)["iris_xyr"].index({2}).item<double>();
    double px = (*seg_im)["pupil_xyr"].index({0}).item<double>();
    double py = (*seg_im)["pupil_xyr"].index({1}).item<double>();
    double pr = (*seg_im)["pupil_xyr"].index({2}).item<double>();

    if (pr <= min_pupil_radius) {
      ////////////cerr << "The pupil radius is too small." << endl;
      continue;
    }

    if (ir <= min_iris_radius) {
      ////////////cerr << "The iris radius is too small." << endl;
      continue;
    }

    double alpha = pr / ir;
    if (alpha <= 0.1 || alpha >= 0.75) {
      continue;
    }

    double center_dist = sqrt( pow((px - ix), 2) + pow((py - iy), 2) );
    //cerr << "center dist: " << center_dist << endl;
    if (double(center_dist / ir) > 0.2) {
      continue;
    }

    //cerr << ix << " " << iy << " " << ir << " " << px << " " << py << " " << pr << " " << M_PI * ir * ir << endl;

    at::Tensor mask = (*seg_im)["mask"].clone().detach();
    at::Tensor mask_ones = at::where(mask > 127.5, 1.0, 0.0);

    at::Tensor ys = torch::linspace(0, mask_ones.sizes()[0]-1, mask_ones.sizes()[0]);
    at::Tensor xs = torch::linspace(0, mask_ones.sizes()[1]-1, mask_ones.sizes()[1]);

    vector<at::Tensor> xsys;
    xsys.push_back(xs);
    xsys.push_back(ys);

    vector<at::Tensor> XY = at::meshgrid(xsys, "xy");
    at::Tensor X = XY[0];
    at::Tensor Y = XY[1];

    at::Tensor dist_from_center_iris = torch::sqrt(torch::pow((X - ix), 2) + torch::pow((Y - iy), 2));
    at::Tensor iris_mask = at::where(dist_from_center_iris <= ir, 1.0, 0.0);    

    at::Tensor mask_ones_inside_iris = mask_ones * iris_mask;  

    //cerr << "mask ones size: " << mask_ones.sizes() << "mask ones sum: " << mask_ones.sum().item<double>() << "mask ones inside iris sum: " << mask_ones_inside_iris.sum().item<double>() << endl;
    double mask_ratio = (mask_ones_inside_iris.sum().item<double>() / (double)(M_PI * ir * ir));
    //cerr << "mask ratio: " << mask_ratio << endl;
    if (mask_ratio < 0.20) {
      continue;
    }

    //cerr << "circApprox worked!" << endl;

    /*
    if (irisLocations.size() > i){
      if (irisLocations[i].limbusCenterX != 0) {
          (*seg_im)["iris_xyr"].index({0}) = (double)irisLocations[i].limbusCenterX;
      }else{
        irisLocations[i].limbusCenterX = (uint16_t) static_cast<unsigned int>((*seg_im)["iris_xyr"][0].item<double>() + 0.5);
      }

      if (irisLocations[i].limbusCenterY != 0) {
          (*seg_im)["iris_xyr"].index({1}) = (double)irisLocations[i].limbusCenterY;
      }else{
        irisLocations[i].limbusCenterY = (uint16_t) static_cast<unsigned int>((*seg_im)["iris_xyr"][1].item<double>() + 0.5);
      }

      if (irisLocations[i].limbusRadius != 0) {
          (*seg_im)["iris_xyr"].index({2}) = (double)irisLocations[i].limbusRadius;
      }else{
        irisLocations[i].limbusRadius = (uint16_t) static_cast<unsigned int>((*seg_im)["iris_xyr"][2].item<double>() + 0.5);
      }

      if (irisLocations[i].pupilRadius != 0) {
          (*seg_im)["pupil_xyr"].index({2}) = (double)irisLocations[i].pupilRadius; 
      }else{
        irisLocations[i].pupilRadius = (uint16_t) static_cast<unsigned int>((*seg_im)["pupil_xyr"][2].item<double>() + 0.5);
      }
    }   
    */ 

    //cerr << "Before cartToPol" << endl;

    map<string, at::Tensor>* c2p_im = new map<string, at::Tensor>;
    this->cartToPol(irisim.clone(), (*seg_im)["pupil_xyr"].clone().detach(), (*seg_im)["iris_xyr"].clone().detach(), c2p_im);
    //at::Tensor image_polar = at::clamp((*c2p_im)["image_polar"].clone().detach(), 0, 255).to(torch::kU8);
    //cv::Mat image_polar_cv2(cv::Size(image_polar.sizes()[3], image_polar.sizes()[2]), CV_8UC1, image_polar.data_ptr());

    //cv::imwrite("./samples/" + to_string(global_index++) + "_" + to_string(i) + ".ppm", image_polar_cv2);

    //cerr << "everything before code extraction worked!" << endl;

    vector<double> code = this->extractCode((*c2p_im)["image_polar"].clone().detach());
    
    //////cerr << code.size() << endl; 

    templ.push_back((uint8_t) iris.irisLR);
    uint8_t* codeBegin = reinterpret_cast<uint8_t*>(&code[0]);
    int codeSize_uint8t = code.size() * (int)(sizeof(double)/sizeof(uint8_t));
    templ.insert(templ.end(), codeBegin, codeBegin + codeSize_uint8t);
    
    delete seg_im;
    delete c2p_im;
  }
  
  //auto stop = high_resolution_clock::now();
  //auto duration = duration_cast<microseconds>(stop - start);
  
  //ofstream timeLog("creation_times.txt", ios::app);
  //timeLog << duration.count() << endl;
  
  if (templ.size() == 0) {
    // //cout << "No templates created." << endl;
    return ReturnStatus(ReturnCode::TemplateCreationError);
  }
  
  return ReturnCode::Success;
}

FRVT::ReturnStatus TripletNNImplFRVT1N::createFaceAndIrisTemplate(
    const std::vector<FRVT::Image> &facesIrises, FRVT::TemplateRole role, std::vector<uint8_t> &templ
) {
  return ReturnStatus(ReturnCode::NotImplemented);
}

  /**
  * @brief This function will be called after all enrollment templates have
  * been created and freezes the enrollment data.
  * After this call, the enrollment dataset will be forever read-only.
  *
  * @details This function allows the implementation to conduct,
  * for example, statistical processing of the feature data, indexing and
  * data re-organization.  The function may create its own data structure.
  * It may increase or decrease the size of the stored data.  No output is
  * expected from this function, except a return code.  The function will
  * generally be called in a separate process after all the enrollment processes
  * are complete.
  * NOTE: Implementations shall not move the input data.  Implementations
  * shall not point to the input data.
  * Implementations should not assume the input data would be readable
  * after the call.  Implementations must,
  * <b>at a minimum, copy the input data</b> or otherwise extract what is
  * needed for search.
  *
  * This function will be called from a single process/thread.
  *
  * @param[in] configDir
  * A read-only directory containing any developer-supplied configuration
  * parameters or run-time data files.
  * @param[in] enrollmentDir
  * The top-level directory in which enrollment data was placed. This
  * variable allows an implementation
  * to locate any private initialization data it elected to place in the
  * directory.
  * @param[in] edbName
  * The name of a single file containing concatenated templates, i.e. the
  * EDB described in <em>File structure for enrolled template collection</em>.
  * While the file will have read-write-delete permission, the implementation
  * should only alter the file if it preserves the necessary content, in
  * other files for example.
  * The file may be opened directly.  It is not necessary to prepend a
  * directory name.  This is a NIST-provided
  * input - implementers shall not internally hard-code or assume any values.
  * @param[in] edbManifestName
  * The name of a single file containing the EDB manifest described in
  * <em>File structure for enrolled template collection</em>.
  * The file may be opened directly.  It is not necessary to prepend a
  * directory name.  This is a NIST-provided
  * input - implementers shall not internally hard-code or assume any values.
  * @param[in] galleryType
  * The composition of the gallery as enumerated by GalleryType.
  */
ReturnStatus TripletNNImplFRVT1N::finalizeEnrollment(
    const std::string &configDir,
    const std::string &enrollmentDir,
    const std::string &edbName,
    const std::string &edbManifestName,
    FRVT_1N::GalleryType galleryType
) {
  ifstream edbsrc(edbName, ios::binary);
  ofstream edbdest(enrollmentDir + "/" + this->edb, ios::binary);
  ifstream manifestsrc(edbManifestName, ios::binary);
  ofstream manifestdest(enrollmentDir + "/" + this->manifest, ios::binary);
  //////////cerr << "Copying edb and manifest to enrollment directory..." << endl;

  edbdest << edbsrc.rdbuf();
  manifestdest << manifestsrc.rdbuf();
  return ReturnCode::Success;
}

/**
  * @brief This function will be called once prior to one or more calls to
  * identifyTemplate().  The function might set static internal variables
  * and read the enrollment gallery into memory
  * so that the enrollment database is available to the subsequent
  * identification searches.
  *
  * This function will be called from a single process/thread.
  *
  * @param[in] configDir
  * A read-only directory containing any developer-supplied configuration
  * parameters or run-time data files.
  * @param[in] enrollmentDir
  * The read-only top-level directory in which enrollment data was placed.
  */
ReturnStatus TripletNNImplFRVT1N::initializeIdentification(const std::string &configDir, const std::string &enrollmentDir) {
  ////////////cerr << "Initializing Identification: "<< endl;
  
  torch::AutoGradMode enable_grad(false);
  
  
  auto edbManifestName = enrollmentDir + "/" + this->manifest;
  auto edbName = enrollmentDir + "/" + this->edb;

  ifstream manifestStream(edbManifestName.c_str());
  if (!manifestStream.is_open()) {
      ////////////cerr << "Failed to open stream for " << edbManifestName << "." << endl;
      return ReturnStatus(ReturnCode::ConfigError);
  }

  ifstream edbStream(edbName, ios::in | ios::binary);
  if (!edbStream.is_open()) {
      ////////////cerr << "Failed to open stream for " << edbName << "." << endl;
      return ReturnStatus(ReturnCode::ConfigError);
  }

  string templId, size, offset;
  while (manifestStream >> templId >> size >> offset) {
      edbStream.seekg(atol(offset.c_str()), ios::beg);
      std::vector<uint8_t> templData(atol(size.c_str()));
      edbStream.read((char*) &templData[0], atol(size.c_str()));

      PrepDatabaseEntry templateIris;
      templateIris.id = templId;

      vector<FRVT::Image::IrisLR> labels;
      vector<vector<double>> codes;
      convert_uint8_to_doublevector(labels, codes, templData);

      for (int idx = 0; idx < labels.size(); idx++) {
        templateIris.searchCodes[labels[idx]].push_back(codes[idx]);
      }
      templates.push_back(templateIris);
  }
  
  ////////////cerr << "done." << endl;

  return ReturnStatus(ReturnCode::Success);
}

FRVT::ReturnStatus
TripletNNImplFRVT1N::insertEnrollment(const uint32_t id, const std::vector<uint8_t> &tmplData){
  PrepDatabaseEntry templateIris;
  templateIris.id = id;

  vector<FRVT::Image::IrisLR> labels;
  vector<vector<double>> codes;
  convert_uint8_to_doublevector(labels, codes, tmplData);

  for (int idx = 0; idx < labels.size(); idx++) {
    templateIris.searchCodes[labels[idx]].push_back(codes[idx]);
  }
  templates.push_back(templateIris);

  return ReturnStatus(ReturnCode::Success);
}

std::string
TripletNNImplFRVT1N::getConfigValue(const std::string& key){
  if (this->cfg.find(key) == this->cfg.end()){
    return "";
  }
  return this->cfg[key];
}

/**
  * @brief This function searches an identification template against the
  * enrollment set, and outputs a vector containing candidateListLength
  * Candidates.
  *
  * @details Each candidate shall be populated by the implementation
  * and added to candidateList.  Note that candidateList will be an empty
  * vector when passed into this function.  
  * 
  * For face recognition: the candidates shall appear in descending order
  * of similarity score - i.e. most similar entries appear first.
  * For iris recognition: the candidates shall appear in ascending order of
  * dissimilarity - i.e. the least dissimilar entries appear first.
  * For multimodal face and iris, the candidates shall appear in descending order
  * of similarity score - i.e. most similar entries appear first. 
  *
  * @param[in] idTemplate
  * A template from the implemented template creation function.  If the value 
  * returned by that function was non-successful, the contents of idTemplate will not be
  * used, and this function will not be called.
  *
  * @param[in] candidateListLength
  * The number of candidates the search should return.
  * @param[out] candidateList
  * Each candidate shall be populated by the implementation.  The candidates
  * shall appear in descending order of similarity score - i.e. most similar
  * entries appear first.
  */
ReturnStatus TripletNNImplFRVT1N::identifyTemplate(
    const std::vector<uint8_t> &idTemplate,
    const uint32_t candidateListLength,
    std::vector<FRVT_1N::Candidate> &candidateList
) {

  torch::AutoGradMode enable_grad(false);
  
  
  //auto start = high_resolution_clock::now();
  
  if (idTemplate.size() == 0) {
    //////////////////cerr << "Template doesn't contain matchable data" << endl;
    return ReturnCode::VerifTemplateError;
  }
  //cout << "Identify Template: " << endl;

  map<FRVT::Image::IrisLR, vector<vector<double>>> searchCodes;
  vector<vector<double>> emptyvec1;
  searchCodes[FRVT::Image::IrisLR::Unspecified] = emptyvec1;
  vector<vector<double>> emptyvec2;
  searchCodes[FRVT::Image::IrisLR::RightIris] = emptyvec2;
  vector<vector<double>> emptyvec3;
  searchCodes[FRVT::Image::IrisLR::LeftIris] = emptyvec3;

  vector<FRVT::Image::IrisLR> labels;
  vector<vector<double>> codes;

  convert_uint8_to_doublevector(labels, codes, idTemplate);

  for (int idx = 0; idx < labels.size(); idx++) {
    searchCodes[labels[idx]].push_back(codes[idx]);
  }

  vector<FRVT_1N::Candidate> all_candidates;

  for (int i = 0; i < templates.size(); i++) {
    double scoreU = -1.0;
    double scoreL = -1.0;
    double scoreR = -1.0;
    double scoreU1 = -1.0;
    double scoreU2 = -1.0;
    double minScore = -1.0;

    if (searchCodes[FRVT::Image::IrisLR::Unspecified].size() > 0 &&
        templates[i].searchCodes[FRVT::Image::IrisLR::Unspecified].size() > 0) {
      scoreU = match(
          searchCodes[FRVT::Image::IrisLR::Unspecified],
          templates[i].searchCodes[FRVT::Image::IrisLR::Unspecified]
      );
      if (scoreU >= 0) {
        minScore = scoreU;
      } else {
        minScore = -1.0;
      }

    } else if (searchCodes[FRVT::Image::IrisLR::Unspecified].size() > 0) {
      if (templates[i].searchCodes[FRVT::Image::IrisLR::LeftIris].size() > 0) {
        scoreU1 = match(
            searchCodes[FRVT::Image::IrisLR::Unspecified],
            templates[i].searchCodes[FRVT::Image::IrisLR::LeftIris]
        );
      }
      if (templates[i].searchCodes[FRVT::Image::IrisLR::RightIris].size() > 0) {
        scoreU2 = match(
            searchCodes[FRVT::Image::IrisLR::Unspecified],
            templates[i].searchCodes[FRVT::Image::IrisLR::RightIris]
        );
      }
      if (scoreU1 >= 0 && scoreU2 >= 0) {
        minScore = min({scoreU1, scoreU2});
      } else if ((scoreU1 < 0) && (scoreU2 >= 0)) {
        minScore = scoreU2;
      } else if ((scoreU2 < 0) && (scoreU1 >= 0)) {
        minScore = scoreU1;
      } else {
        minScore = -1.0;
      }

    } else if (templates[i].searchCodes[FRVT::Image::IrisLR::Unspecified].size() > 0) {
      if (searchCodes[FRVT::Image::IrisLR::LeftIris].size() > 0) {
        scoreU1 = match(
            templates[i].searchCodes[FRVT::Image::IrisLR::Unspecified],
            searchCodes[FRVT::Image::IrisLR::LeftIris]
        );
      }
      if (searchCodes[FRVT::Image::IrisLR::RightIris].size() > 0) {
        scoreU2 = match(
            templates[i].searchCodes[FRVT::Image::IrisLR::Unspecified],
            searchCodes[FRVT::Image::IrisLR::RightIris]
        );
      }
      if (scoreU1 >= 0 && scoreU2 >= 0) {
        minScore = min({scoreU1, scoreU2});
      } else if ((scoreU1 < 0) && (scoreU2 >= 0)) {
        minScore = scoreU2;
      } else if ((scoreU2 < 0) && (scoreU1 >= 0)) {
        minScore = scoreU1;
      } else {
        minScore = -1.0;
      }

    } else {
      if (searchCodes[FRVT::Image::IrisLR::LeftIris].size() > 0 &&
          templates[i].searchCodes[FRVT::Image::IrisLR::LeftIris].size() > 0) {
            scoreL = match(
              searchCodes[FRVT::Image::IrisLR::LeftIris],
              templates[i].searchCodes[FRVT::Image::IrisLR::LeftIris]
            );
      }
      if (searchCodes[FRVT::Image::IrisLR::RightIris].size() > 0 &&
          templates[i].searchCodes[FRVT::Image::IrisLR::RightIris].size() > 0) {
            scoreR = match(
              searchCodes[FRVT::Image::IrisLR::RightIris],
              templates[i].searchCodes[FRVT::Image::IrisLR::RightIris]
            );
      }
      if (scoreL >= 0 && scoreR >= 0) {
        minScore = min({scoreL, scoreR});
      } else if ((scoreL < 0) && (scoreR >= 0)) {
        minScore = scoreR;
      } else if ((scoreR < 0) && (scoreL >= 0)) {
        minScore = scoreL;
      } else {
        minScore = -1.0;
      }
    }
    //////cerr << "minScore: " << minScore << endl;
    if (minScore >= 0) {
      FRVT_1N::Candidate candidate;
      candidate.isAssigned = true;
      candidate.templateId = templates[i].id;
      candidate.score = minScore;
      all_candidates.push_back(candidate);
    }
  }
  if (all_candidates.size() == 0) {
    //////////////////cerr << "No candidates found." << endl;
    FRVT_1N::Candidate candidate;
    candidate.isAssigned = false;
    candidateList.push_back(candidate);
    return ReturnCode::UnknownError;
  }

  int n_candidates = min({(int) all_candidates.size(), (int) candidateListLength});
  for (int i = 0; i < n_candidates; i++) {
    double min_distance = all_candidates[0].score;
    int min_index = 0;
    for (int j = 1; j < all_candidates.size(); j++) {
      if (all_candidates[j].score < min_distance) {
        min_distance = all_candidates[j].score;
        min_index = j;
      }
    }
    candidateList.push_back(all_candidates[min_index]);
    if (all_candidates.size() > 0) {
      all_candidates.erase(all_candidates.begin() + min_index);
    } else {
      break;
    }
  }
  
  //auto stop = high_resolution_clock::now();
  //auto duration = duration_cast<microseconds>(stop - start);
  
  //ofstream timeLog("identification_times.txt", ios::app);
  //timeLog << duration.count() << endl;

  return ReturnCode::Success;
}


/* Private Code */

void TripletNNImplFRVT1N::load_cfg(string cfg_path) {
  string line;
  ifstream fin;
  fin.open(cfg_path);
  int i = 0;
  string del = ">";
  while (getline(fin, line)) {
    if (i < 3) {
      i++;
      continue;
    }
    vector<string> options(2);
    int start, end = -1 * del.size();
    int j = 0;
    do {
      start = end + del.size();
      end = line.find(del, start);
      string string_part = line.substr(start, end - start);
      string_part.erase(0, string_part.find_first_not_of(" \n\r\t"));
      string_part.erase(string_part.find_last_not_of(" \n\r\t") + 1);
      options[j] = string_part;
      j++;
    } while (end != -1);

    if (options[1][0] == '\"') {
      options[1] = options[1].substr(1, options[1].size() - 2);
    }

    cfg[options[0]] = options[1];
  }

  // Close the file
  fin.close();
}

/*
double TripletNNImplFRVT1N::matchCodes(at::Tensor code1, at::Tensor code2, at::Tensor mask1_inp, at::Tensor mask2_inp) {
  
  torch::AutoGradMode enable_grad(false);
  
  
  int margin = (int) floor(filter_size / 2);
  code1 = code1.to(torch::kCPU);
  code2 = code2.to(torch::kCPU);
  at::Tensor mask1 = mask1_inp.index({Slice(margin, -margin), Slice(None, None)}).to(torch::kCPU);
  at::Tensor mask2 = mask2_inp.index({Slice(margin, -margin), Slice(None, None)}).to(torch::kCPU);

  at::Tensor scoreC = at::zeros({num_filters, 2 * max_shift + 1});
  for (int shift = -max_shift; shift <= max_shift; shift++) {
    at::Tensor andMasks = at::logical_and(mask1, at::roll(mask2, shift, 1));
    at::Tensor xorCodes = at::logical_xor(code1, at::roll(code2, shift, 2));
    // debug code
    std::ostringstream oss;
    oss << "./Codes/" << getpid() << "_" <<  shift + max_shift << ".ppm";
    
    ifstream f(oss.str());
    if (!f.good()) {
      auto img_uint = xorCodes.index({0, Slice(None, None), Slice(None, None)});
      img_uint = (img_uint * 255).toType(torch::kU8);
  		auto img = cv::Mat(48, 512, CV_8UC1, img_uint.data_ptr());
      cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
      cv::imwrite(oss.str(), img);
    }
    // debug code end
    vector<at::Tensor> mask_list_v;
    for (int i = 0; i < num_filters; i++) {
      mask_list_v.push_back(andMasks.clone().detach());
    }
    at::TensorList mask_list_t = at::TensorList(mask_list_v);
    at::Tensor andMasksRep = at::stack(mask_list_t, 0);
    at::Tensor xorCodesMasked = at::logical_and(xorCodes, andMasksRep);
    at::Tensor score_results = (at::sum(xorCodesMasked, vector<int64_t> {1, 2}) / at::sum(andMasks));
    scoreC.index({Slice(None, None), shift}) = score_results;
  }

  at::Tensor scoreMean = at::mean(scoreC, 0);
  at::Tensor score = at::min(scoreMean);

  return score.item<double>();
}
*/

double TripletNNImplFRVT1N::matchCodes(vector<double> code1, vector<double> code2) {  
  //cerr << "match codes called!" << endl;
  /*
  if (filesystem::exists("/mnt/c/TripletNNDebug/vector_after_uint8.txt")) {
    cerr << "file exists!" << endl;
  } else {
    cerr << "saving file after uint8" << endl;
    ofstream txt_file;      // pay attention here! ofstream
    txt_file.open("/mnt/c/TripletNNDebug/vector_after_uint8.txt");
    for (double num : code1) {
      txt_file << std::setprecision(std::numeric_limits<double>::digits10 + 1) << num << ", ";
    }
    txt_file.close();
  }
  */
  double dist = 0.0;
  for (int i = 0; i < codeSize; i++) {
    dist += (double) pow(code1[i] - code2[i], 2);
  }
  return sqrt(dist);
}

cv::Mat TripletNNImplFRVT1N::get_cv2_image(const std::shared_ptr<uint8_t> &data, uint16_t width, uint16_t height, bool isRGB) {
  int h = height;
  int w = width;

  cv::Mat cv2im(h, w, isRGB ? CV_8UC3 : CV_8UC1, data.get());
  if (isRGB == true) {
    ////////////////////cerr << "Getting R channel of the BGR image." << endl;
    vector<cv::Mat> channels(3);
    cv::split(cv2im, channels);
    return channels[2];
  } else {
    //////////////////cerr << "Getting the grayscale image." << endl;
    return cv2im;
  }
}

at::Tensor TripletNNImplFRVT1N::grid_sample(at::Tensor input, at::Tensor grid) {
  
  torch::AutoGradMode enable_grad(false);
  
  // grid: [-1, 1]
  int N = input.sizes()[0];
  int C = input.sizes()[1];
  int H = input.sizes()[2];
  int W = input.sizes()[3];
  at::Tensor gridx = grid.index({Slice(None, None), Slice(None, None), Slice(None, None), 0});
  at::Tensor gridy = grid.index({Slice(None, None), Slice(None, None), Slice(None, None), 1});
  gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1;
  gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1;
  vector<at::Tensor> grid_list_v;
  grid_list_v.push_back(gridx);
  grid_list_v.push_back(gridy);
  at::TensorList grid_list_t = at::TensorList(grid_list_v);
  at::Tensor newgrid = at::stack(grid_list_t, -1);
  auto gridsampleoptions = torch::nn::functional::GridSampleFuncOptions()
                                .mode(torch::kBilinear)
                                .padding_mode(torch::kBorder)
                                .align_corners(true);
  at::Tensor ret =
    torch::nn::functional::grid_sample(input.to(torch::kCPU), newgrid.to(torch::kCPU), gridsampleoptions);
  return ret;
}

void TripletNNImplFRVT1N::fix_image(cv::Mat &ret) {
  // ret.convertTo(ret, CV_32FC3, 1.0f / 255.0f);
  int w = ret.cols;
  int h = ret.rows;
  double aspect_ratio = (double) w / (double) h;
  cv::Scalar value(127, 127, 127);
  if (aspect_ratio >= 1.333 && aspect_ratio <= 1.334) {
    cv::resize(ret, ret, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR_EXACT);
  } else if (aspect_ratio < 1.333) {
    double w_new = h * (4.0 / 3.0);
    double w_pad = (w_new - w) / 2;
    int left = (int) w_pad;
    int right = (int) w_pad;
    cv::copyMakeBorder(ret, ret, 0, 0, left, right, cv::BORDER_CONSTANT, value);
    cv::resize(ret, ret, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR_EXACT);
  } else {
    double h_new = w * (3.0 / 4.0);
    double h_pad = (h_new - h) / 2;
    int top = (int) h_pad;
    int bottom = (int) h_pad;
    cv::copyMakeBorder(ret, ret, top, bottom, 0, 0, cv::BORDER_CONSTANT, value);
    cv::resize(ret, ret, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR_EXACT);
  }
}

void TripletNNImplFRVT1N::segment_and_circApprox(cv::Mat image, map<string, at::Tensor>* seg_im) {
  
  torch::AutoGradMode enable_grad(false);
  c10::InferenceMode guard(false);

  //cerr << "circApprox called!" << endl;  

  int w = image.cols;
  int h = image.rows;
  
  cv::resize(image, image, cv::Size(resolution[0], resolution[1]), 0, 0, 1);

  double diag = (double) sqrt(w * w + h * h);

  at::Tensor input_tensor = torch::from_blob(image.data, { image.rows, image.cols, 1 }, at::kByte).clone().detach();
  input_tensor = input_tensor.unsqueeze_(0);
  input_tensor = input_tensor.permute({0, 3, 1, 2});
  input_tensor = input_tensor.to(at::kFloat);
  input_tensor = input_tensor.div(255.0);

  // Mask
  at::Tensor mask_input_tensor = ((input_tensor.clone().detach() - norm_params_mask[0]) / norm_params_mask[1]);
  //vector<torch::jit::IValue> *inputs_mask = new vector<torch::jit::IValue>;
  //inputs_mask->push_back(mask_input_tensor);
  //at::Tensor out_tensor = mask_model.forward(*inputs_mask).toTensor().clone().detach();
  //at::Tensor out_tensor = mask_model.forward(*inputs_mask).toTensor().clone().detach();
  //delete inputs_mask;
  at::Tensor out_tensor = mask_model.forward({mask_input_tensor}).toTensor().clone().detach();
  out_tensor = torch::sigmoid(out_tensor);
  
  at::Tensor mask = at::where(out_tensor > 0.5, 255.0, 0.0);
  mask = mask.to(torch::kU8);
  mask = torch::nn::functional::interpolate(
      mask, torch::nn::functional::InterpolateFuncOptions().size(vector<int64_t> {h, w}).mode(torch::kNearest)
  );
  mask = mask.squeeze();

  // Circle params
  at::Tensor circ_input_tensor = ((input_tensor.clone().detach() - norm_params_circle[0]) / norm_params_circle[1]);
  circ_input_tensor = circ_input_tensor.repeat({1, 3, 1, 1});
  circ_input_tensor = circ_input_tensor.to(torch::kCPU).to(at::kFloat);
  //vector<torch::jit::IValue> *inputs_circle = new vector<torch::jit::IValue>;
  //inputs_circle->push_back(circ_input_tensor);
  //at::Tensor inp_xyr_t = circle_model.forward(*inputs_circle).toTensor();
  at::Tensor inp_xyr_t = circle_model.forward({circ_input_tensor}).toTensor().clone().detach().to(at::kDouble);
  //delete inputs_circle;
  inp_xyr_t = inp_xyr_t.to(torch::kCPU);
  at::Tensor inp_xyr = inp_xyr_t.squeeze();
  inp_xyr[0] *= (double)w;
  inp_xyr[1] *= (double)h;
  inp_xyr[2] *= 0.5 * 0.8 * diag;
  inp_xyr[3] *= (double)w;
  inp_xyr[4] *= (double)h;
  inp_xyr[5] *= 0.5 * diag;
  
  (*seg_im)["pupil_xyr"] = inp_xyr.index({Slice(None, 3)}).clone().detach();
  (*seg_im)["iris_xyr"] = inp_xyr.index({Slice(3, None)}).clone().detach();
  (*seg_im)["mask"] = mask;
  //cerr << "circApprox worked!!" << endl;
}

/*
void TripletNNImplFRVT1N::circApprox(cv::Mat image, map<string, at::Tensor>* seg_im) {
  
  torch::AutoGradMode enable_grad(false);
  

  int w = image.cols;
  int h = image.rows;

  cv::resize(image, image, cv::Size(resolution[0], resolution[1]), 0, 0, 1);

  double diagonal = sqrt(w * w + h * h) / 2;

  at::Tensor input_tensor = torch::from_blob(image.data, { image.rows, image.cols, 1 }, at::kByte).clone().detach();
  input_tensor = input_tensor.unsqueeze_(0);
  input_tensor = input_tensor.permute({0, 3, 1, 2});
  input_tensor = input_tensor.to(at::kFloat);
  input_tensor = input_tensor.mul(1.0/255.0);


  // Circle params
  at::Tensor circ_input_tensor = ((input_tensor.clone().detach() - norm_params_circle[0]) / norm_params_circle[1]);
  circ_input_tensor = circ_input_tensor.repeat({1, 3, 1, 1});
  circ_input_tensor = circ_input_tensor.to(torch::kCPU);
  vector<torch::jit::IValue> *inputs_circle = new vector<torch::jit::IValue>;
  inputs_circle->push_back(circ_input_tensor);
  at::Tensor inp_xyr_t = circle_model.forward(*inputs_circle).toTensor();
  delete inputs_circle;
  inp_xyr_t = inp_xyr_t.to(torch::kCPU);
  at::Tensor inp_xyr = inp_xyr_t.squeeze();
  inp_xyr[0] *= w;
  inp_xyr[1] *= h;
  inp_xyr[2] *= 0.8 * diagonal;
  inp_xyr[3] *= w;
  inp_xyr[4] *= h;
  inp_xyr[5] *= diagonal;
  
  (*seg_im)["pupil_xyr"] = inp_xyr.clone().detach().index({Slice(None, 3)});
  (*seg_im)["iris_xyr"] = inp_xyr.clone().detach().index({Slice(3, None)});
}
*/

/*
int i = 0;
void saveCvImage(at::Tensor tensor)
{
    int width = tensor.sizes()[0];
    int height = tensor.sizes()[1];
    try
    {
        cv::Mat output_mat(cv::Size{ height, width }, CV_8UC3, tensor.data_ptr<uchar>());
        cv::Mat image = output_mat.clone();
        cv::imwrite("./samples/" + to_string(i++) + ".png", image);
    }
    catch (const c10::Error& e)
    {
        std::cout << "an error has occured : " << e.msg() << std::endl;
    }
}
*/

void TripletNNImplFRVT1N::cartToPol(cv::Mat image, at::Tensor pupil_xyr, at::Tensor iris_xyr, map<string, at::Tensor>* c2p_im) {

  torch::AutoGradMode enable_grad(false);
  c10::InferenceMode guard(false);
  //cerr << "cartToPol called!" << endl;
  
  int w = image.cols;
  int h = image.rows;
  image.convertTo(image, CV_32FC3);
  auto image_tensor = torch::from_blob(image.data, {1, h, w, 1}).clone().detach();
  image_tensor = image_tensor.permute({0, 3, 1, 2});
  double width = (double) image_tensor.sizes()[3];
  double height = (double) image_tensor.sizes()[2];

  pupil_xyr = pupil_xyr.unsqueeze({0}).to(at::kFloat);
  iris_xyr = iris_xyr.unsqueeze({0}).to(at::kFloat);

  at::Tensor theta = 2 * M_PI * (at::linspace(0, polar_width - 1, polar_width) / polar_width);
  theta = theta.to(at::kFloat).to(torch::kCPU);

  at::Tensor pxCirclePoints =
      pupil_xyr.index({Slice(None, None), 0}).reshape({-1, 1}) +
      at::matmul(pupil_xyr.index({Slice(None, None), 2}).reshape({-1, 1}), torch::cos(theta).reshape({1, polar_width}));
  at::Tensor pyCirclePoints =
      pupil_xyr.index({Slice(None, None), 1}).reshape({-1, 1}) +
      at::matmul(pupil_xyr.index({Slice(None, None), 2}).reshape({-1, 1}), torch::sin(theta).reshape({1, polar_width}));

  at::Tensor ixCirclePoints =
      iris_xyr.index({Slice(None, None), 0}).reshape({-1, 1}) +
      at::matmul(iris_xyr.index({Slice(None, None), 2}).reshape({-1, 1}), torch::cos(theta).reshape({1, polar_width}));
  at::Tensor iyCirclePoints =
      iris_xyr.index({Slice(None, None), 1}).reshape({-1, 1}) +
      at::matmul(iris_xyr.index({Slice(None, None), 2}).reshape({-1, 1}), torch::sin(theta).reshape({1, polar_width}));

  at::Tensor radius = (at::linspace(1, polar_height, polar_height) / polar_height).reshape({-1, 1}).to(torch::kCPU);
  at::Tensor pxCoords = at::matmul((1 - radius), pxCirclePoints.reshape({-1, 1, polar_width}));
  at::Tensor pyCoords = at::matmul((1 - radius), pyCirclePoints.reshape({-1, 1, polar_width}));
  at::Tensor ixCoords = at::matmul(radius, ixCirclePoints.reshape({-1, 1, polar_width}));
  at::Tensor iyCoords = at::matmul(radius, iyCirclePoints.reshape({-1, 1, polar_width}));
  at::Tensor x = (pxCoords + ixCoords).to(at::kFloat).to(torch::kCPU);
  at::Tensor x_norm = ((x - 1) / (width - 1)) * 2 - 1;
  at::Tensor y = (pyCoords + iyCoords).to(at::kFloat).to(torch::kCPU);
  at::Tensor y_norm = ((y - 1) / (height - 1)) * 2 - 1;
  at::Tensor grid_sample_mat = at::cat({x_norm.unsqueeze({-1}), y_norm.unsqueeze({-1})}, -1).to(torch::kCPU);
  at::Tensor image_polar = grid_sample(image_tensor, grid_sample_mat);
  image_polar = at::clamp(at::round(image_polar.index({0, 0, Slice(None, None), Slice(None, None)})), 0, 255);

  (*c2p_im)["image_polar"] = image_polar.clone().detach().to(torch::kU8);
  //cerr << "cartToPol worked!" << endl;
  //saveCvImage((*c2p_im)["image_polar"]);
}

vector<double> TripletNNImplFRVT1N::extractCode(at::Tensor image_polar) {
  
  torch::AutoGradMode enable_grad(false);
  c10::InferenceMode guard(false);
  //cerr << "Extracting code..." << endl;
  
  at::Tensor image_polar_f = image_polar.to(at::kFloat).clone().detach();
  at::Tensor image_polar_mean = image_polar_f.mean();
  at::Tensor image_polar_std = image_polar_f.std();
  at::Tensor image_polar_norm = torch::clamp(torch::nan_to_num(((image_polar_f - image_polar_mean) / image_polar_std), 4.0, 4.0, -4.0), -4.0, 4.0);
  at::Tensor vector_input_tensor = image_polar_norm.repeat({1, 3, 1, 1});
  vector_input_tensor = vector_input_tensor.to(torch::kCPU);
  at::Tensor code_t = vector_model.forward({vector_input_tensor}).toTensor().flatten().clone().detach().to(at::kDouble);
  vector<double> code;
  if (code_t.sizes()[0] != codeSize) {
    cout << "Not matching, should match: " << code_t.sizes()[0] << ", " << codeSize << endl;
  }
  for (int i = 0; i < code_t.sizes()[0]; i++){
    code.push_back(code_t[i].item<double>());
  }
  //double* codePtr = (double*) code_t.data_ptr();
  //vector<double> code(codePtr, codePtr + codeSize);
  /*
  if (filesystem::exists("/mnt/c/TripletNNDebug/vector_before_uint8.txt")) {
    cerr << "file exists! updated" << endl;
  } else {
    cerr << "saving file before uint8" << endl;
    ofstream txt_file;      // pay attention here! ofstream
    txt_file.open("/mnt/c/TripletNNDebug/vector_before_uint8.txt");
    for (double num : code) {
      txt_file << std::setprecision(std::numeric_limits<double>::digits10 + 1) << num << ", ";
    }
    txt_file.close();
  }
  */
  return code;
}

bool TripletNNImplFRVT1N::hasEnding(const string &fullString, const string &ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

void TripletNNImplFRVT1N::convert_uint8_to_doublevector(
    vector<FRVT::Image::IrisLR> &labels,
    vector<vector<double>> &codes,
    const vector<uint8_t> &vec
) {
  
  
  int idx = 0;
  if (vec.size() == 0) {
    //////////////////////cerr << "No templates found, using zero codes and mask" << endl;
    labels.push_back(FRVT::Image::IrisLR::Unspecified);
    vector<double> code(codeSize, 0.0);
    codes.push_back(code);
  } else {
    //////////////////cerr << "Templates found" << endl;
    while (idx < vec.size()) {
      labels.push_back((FRVT::Image::IrisLR) vec[idx]);
      idx++;
      uint8_t *codeData = new uint8_t[codeSize * (sizeof(double)/sizeof(uint8_t))];
      std::memcpy(codeData, &vec[idx], codeSize * (sizeof(double)/sizeof(uint8_t)));
      vector<double> code(reinterpret_cast<double*>(codeData), reinterpret_cast<double*>(codeData) + codeSize);
      codes.push_back(code);
      idx += (codeSize * (sizeof(double)/sizeof(uint8_t)));
    }
  }
}

double TripletNNImplFRVT1N::match(
    vector<vector<double>> codes1, vector<vector<double>> codes2
) {
  
  vector<double> scores;
  for (int i = 0; i < codes1.size(); i++) {
    vector<double> code1 = codes1[i];
    for (int j = 0; j < codes2.size(); j++) {
      vector<double> code2 = codes2[j];
      scores.push_back(this->matchCodes(code1, code2));
    }
  }

  if (scores.size() > 1) {
    sort(scores.begin(), scores.end());
    if (scores.size() % 2 == 0) {
      int ind = int(scores.size() / 2);
      return (scores[ind] + scores[ind-1]) / 2.0;
    } else {
      int ind = int((scores.size() - 1)/2);
      return scores[ind];
    }
  } else {
    return scores[0];
  }
  return -1.0;
}

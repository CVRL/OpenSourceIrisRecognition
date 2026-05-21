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

#include "arcirisimplfrvt1N.h"
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

ArcIrisImplFRVT1N::ArcIrisImplFRVT1N() {
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

ArcIrisImplFRVT1N::~ArcIrisImplFRVT1N() {
  //delete enable_grad;
}
std::shared_ptr<FRVT_1N::Interface> FRVT_1N::Interface::getImplementation() {
  return std::make_shared<ArcIrisImplFRVT1N>();
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
ReturnStatus ArcIrisImplFRVT1N::initializeTemplateCreation(const std::string &configDir, FRVT::TemplateRole role) {
  if (init == true) {
    //////////cerr << "Already initialized, skipping..." << endl;
    return ReturnStatus(ReturnCode::Success, "Already Initialized");
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
    return ReturnStatus(ReturnCode::ConfigError, "Error loading arciris model");
  }
  vector_model.eval();

  string circle_param_model_path = (configDir + "/" + cfg["circle_param_model_path"]).c_str();
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    circle_model = torch::jit::load(circle_param_model_path, torch::kCPU);
  } catch (const c10::Error &e) {
    ////////////cerr << "error loading the circle model\n";
    return ReturnStatus(ReturnCode::ConfigError, "Error loading circle detection model");
  }
  circle_model.eval();

  string mask_model_path = (configDir + "/" + cfg["coarse_mask_model_path"]).c_str();
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    mask_model = torch::jit::load(mask_model_path, torch::kCPU);
  } catch (const c10::Error &e) {
    ////////////cerr << "error loading the circle model\n";
    return ReturnStatus(ReturnCode::ConfigError, "Error loading mask detection model");
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
  min_std_dev = stod(cfg["MINIMUM_IMAGE_STD_DEV"]);
  min_iris_mask_to_circle_ratio = stod(cfg["MINIMUM_IRIS_MASK_TO_CIRCLE_RATIO"]);
  min_alpha = stod(cfg["MINIMUM_ALPHA"]);
  max_alpha = stod(cfg["MAXIMUM_ALPHA"]);
  maximum_pupil_limbus_center_shift = stod(cfg["MAXIMUM_PUPIL_LIMBUS_CENTER_SHIFT"]);

  //cerr << "initialization worked!" << endl;

  return ReturnStatus(ReturnCode::Success);
}

ReturnStatus ArcIrisImplFRVT1N::createFaceTemplate(
    const std::vector<FRVT::Image> &faces,
    FRVT::TemplateRole role,
    std::vector<uint8_t> &templ,
    std::vector<FRVT::EyePair> &eyeCoordinates
) {
  return ReturnStatus(ReturnCode::NotImplemented);
}

ReturnStatus ArcIrisImplFRVT1N::createFaceTemplate(
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

FRVT::ReturnStatus ArcIrisImplFRVT1N::createIrisTemplate(
    const std::vector<FRVT::Image> &irises,
    FRVT::TemplateRole role,
    std::vector<uint8_t> &templ,
    std::vector<FRVT::IrisAnnulus> &irisLocations
) {
  
  torch::AutoGradMode enable_grad(false);

  for (int i = 0; i < irises.size(); i++) {
    const FRVT::Image &iris = irises[i];

    bool isRGB = (iris.depth == 24);
    cv::Mat irisim = get_cv2_image(iris.data, iris.width, iris.height, isRGB);
    cv::Scalar mean,stddev;
    cv::meanStdDev(irisim,mean,stddev);

    if (stddev.val[0] <= min_std_dev) {
        return ReturnStatus(ReturnCode::TemplateCreationError, "Iris image contrast too low");
    }

    this->fix_image(irisim);
    
    map<string, at::Tensor>* seg_im = new map<string, at::Tensor>;
    this->segment_and_circApprox(irisim.clone(), seg_im);
    double ix = (*seg_im)["iris_xyr"].index({0}).item<double>();
    double iy = (*seg_im)["iris_xyr"].index({1}).item<double>();
    double ir = (*seg_im)["iris_xyr"].index({2}).item<double>();
    double px = (*seg_im)["pupil_xyr"].index({0}).item<double>();
    double py = (*seg_im)["pupil_xyr"].index({1}).item<double>();
    double pr = (*seg_im)["pupil_xyr"].index({2}).item<double>();
    
    if (ir <= pr) {
      delete seg_im;
      return ReturnStatus(ReturnCode::TemplateCreationError, "Iris radius smaller than pupil radius");        
    }

    if (pr <= min_pupil_radius) {
      delete seg_im;
      return ReturnStatus(ReturnCode::TemplateCreationError, "Pupil radius is too small");
    }

    if (ir <= min_iris_radius) {
      delete seg_im;
      return ReturnStatus(ReturnCode::TemplateCreationError, "Iris radius is too small");
    }

    double alpha = pr / ir;
    if (alpha < min_alpha || alpha > max_alpha) {
      delete seg_im;
      return ReturnStatus(ReturnCode::TemplateCreationError, "Pupil-to-iris ratio doesn't fall in the valid range i.e., alpha < 0.1 or alpha > 0.75");
    }

    double center_dist = sqrt( pow((px - ix), 2) + pow((py - iy), 2) );
    
    if (double(center_dist / ir) >= maximum_pupil_limbus_center_shift) {
      delete seg_im;
      return ReturnStatus(ReturnCode::TemplateCreationError, "Pupil and iris centers are too far apart, more than half of the iris radius.");
    }

    // Get original mask and threshold it
    at::Tensor mask = (*seg_im)["mask"].clone().detach();
    at::Tensor mask_ones = at::where(mask > 127.5, 1.0, 0.0);

    // Get dimensions to create a spatial coordinate grid
    int64_t H = mask.size(-2);
    int64_t W = mask.size(-1);
    auto options = at::TensorOptions().dtype(at::kFloat).device(mask.device());

    auto y = at::arange(H, options);
    auto x = at::arange(W, options);

    // Create grid (using "ij" indexing so y maps to rows, x maps to cols)
    std::vector<at::Tensor> grid = at::meshgrid({y, x}, "ij");
    at::Tensor grid_y = grid[0];
    at::Tensor grid_x = grid[1];

    // Calculate squared distances to pupil and iris centers for every pixel
    at::Tensor dist_sq_pupil = (grid_x - px).pow(2) + (grid_y - py).pow(2);
    at::Tensor dist_sq_iris  = (grid_x - ix).pow(2) + (grid_y - iy).pow(2);

    // Create the geometric ring mask: strictly outside pupil AND inside iris
    at::Tensor ring_mask = (dist_sq_pupil > (pr * pr)).logical_and(dist_sq_iris <= (ir * ir));

    // Filter the original mask
    // Multiply thresholded mask by the ring mask (casted to Float to match mask_ones)
    at::Tensor filtered_mask_ones = mask_ones * ring_mask.to(mask_ones.options());

    // Calculate the final ratio using the filtered mask
    double ring_area = ring_mask.sum().item<double>();
    double mask_ratio = (ring_area > 0.0)
        ? (filtered_mask_ones.sum().item<double>() / ring_area)
        : 0.0;
   
    if (mask_ratio <= min_iris_mask_to_circle_ratio) {
      delete seg_im;
      return ReturnStatus(ReturnCode::TemplateCreationError, "Detected mask is too small.");
    }
    
    if (irisLocations.size() > i){
      if (irisLocations[i].limbusCenterX > 0) {
          (*seg_im)["iris_xyr"].index({0}) = (double)irisLocations[i].limbusCenterX;
      }

      if (irisLocations[i].limbusCenterY > 0) {
          (*seg_im)["iris_xyr"].index({1}) = (double)irisLocations[i].limbusCenterY;
      }

      if (irisLocations[i].limbusRadius > 0) {
          (*seg_im)["iris_xyr"].index({2}) = (double)irisLocations[i].limbusRadius;
      }
    }   

    map<string, at::Tensor>* c2p_im = new map<string, at::Tensor>;
    this->cartToPol(irisim.clone(), (*seg_im)["pupil_xyr"].clone().detach(), (*seg_im)["iris_xyr"].clone().detach(), c2p_im);

    vector<double> code = this->extractCode((*c2p_im)["image_polar"].clone().detach());

    templ.push_back((uint8_t) iris.irisLR);
    uint8_t* codeBegin = reinterpret_cast<uint8_t*>(&code[0]);
    int codeSize_uint8t = code.size() * (int)(sizeof(double)/sizeof(uint8_t));
    templ.insert(templ.end(), codeBegin, codeBegin + codeSize_uint8t);
    
    delete seg_im;
    delete c2p_im;
  }
  
  
  if (templ.size() == 0) {
    // //cout << "No templates created." << endl;
    return ReturnStatus(ReturnCode::ExtractError, "Template size zero, creation failed");
  }
  
  return ReturnStatus(ReturnCode::Success);
}

FRVT::ReturnStatus ArcIrisImplFRVT1N::createFaceAndIrisTemplate(
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
ReturnStatus ArcIrisImplFRVT1N::finalizeEnrollment(
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
  return ReturnStatus(ReturnCode::Success);
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
ReturnStatus ArcIrisImplFRVT1N::initializeIdentification(const std::string &configDir, const std::string &enrollmentDir) {
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
ArcIrisImplFRVT1N::insertEnrollment(const uint32_t id, const std::vector<uint8_t> &tmplData){
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
ArcIrisImplFRVT1N::getConfigValue(const std::string& key){
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
ReturnStatus ArcIrisImplFRVT1N::identifyTemplate(
    const std::vector<uint8_t> &idTemplate,
    const uint32_t candidateListLength,
    std::vector<FRVT_1N::Candidate> &candidateList
) {

  torch::AutoGradMode enable_grad(false);
  
  if (idTemplate.size() == 0) {
    //////////////////cerr << "Template doesn't contain matchable data" << endl;
    return ReturnStatus(ReturnCode::VerifTemplateError, "Template size zero");
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
    double scoreU = M_PI + 0.1;
    double scoreL = M_PI + 0.1;
    double scoreR = M_PI + 0.1;
    double scoreUL = M_PI + 0.1;
    double scoreUR = M_PI + 0.1;
    double scoreULR = M_PI + 0.1;
    double scoreLU = M_PI + 0.1;
    double scoreRU = M_PI + 0.1;
    double scoreLRU = M_PI + 0.1;
    double minScore = M_PI + 0.1;

    if (searchCodes[FRVT::Image::IrisLR::Unspecified].size() > 0 &&
        templates[i].searchCodes[FRVT::Image::IrisLR::Unspecified].size() > 0) {
      scoreU = match(
          searchCodes[FRVT::Image::IrisLR::Unspecified],
          templates[i].searchCodes[FRVT::Image::IrisLR::Unspecified]
      );
      
    } 
    if (searchCodes[FRVT::Image::IrisLR::Unspecified].size() > 0) {
      if (templates[i].searchCodes[FRVT::Image::IrisLR::LeftIris].size() > 0) {
        scoreUL = match(
            searchCodes[FRVT::Image::IrisLR::Unspecified],
            templates[i].searchCodes[FRVT::Image::IrisLR::LeftIris]
        );
      }
      if (templates[i].searchCodes[FRVT::Image::IrisLR::RightIris].size() > 0) {
        scoreUR = match(
            searchCodes[FRVT::Image::IrisLR::Unspecified],
            templates[i].searchCodes[FRVT::Image::IrisLR::RightIris]
        );
      }
      scoreULR = min(scoreUL, scoreUR);

    }
    if (templates[i].searchCodes[FRVT::Image::IrisLR::Unspecified].size() > 0) {
      if (searchCodes[FRVT::Image::IrisLR::LeftIris].size() > 0) {
        scoreLU = match(
            templates[i].searchCodes[FRVT::Image::IrisLR::Unspecified],
            searchCodes[FRVT::Image::IrisLR::LeftIris]
        );
      }
      if (searchCodes[FRVT::Image::IrisLR::RightIris].size() > 0) {
        scoreRU = match(
            templates[i].searchCodes[FRVT::Image::IrisLR::Unspecified],
            searchCodes[FRVT::Image::IrisLR::RightIris]
        );
      }
      scoreLRU = min(scoreLU, scoreRU);

    } 
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
    
    minScore = min({scoreU, scoreULR, scoreLRU, scoreL, scoreR});
    
    if (minScore >= 0.0 && minScore <= M_PI) {
      FRVT_1N::Candidate candidate;
      candidate.templateId = templates[i].id;
      candidate.score = minScore;
      candidate.isAssigned = true;
      all_candidates.push_back(candidate);
    }
  }
  if (all_candidates.size() == 0) {
    return ReturnStatus(ReturnCode::VerifTemplateError, "No candidates found");
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
  return ReturnStatus(ReturnCode::Success);
}


/* Private Code */

void ArcIrisImplFRVT1N::load_cfg(string cfg_path) {
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
double ArcIrisImplFRVT1N::matchCodes(vector<double> code1, vector<double> code2) {  
  double dist = 0.0;
  for (int i = 0; i < codeSize; i++) {
    dist += (double) pow(code1[i] - code2[i], 2);
  }
  return sqrt(dist);
}
*/

double ArcIrisImplFRVT1N::matchCodes(std::vector<double> code1, std::vector<double> code2) {  
  double dot = 0.0;
  double mag1_sq = 0.0;
  double mag2_sq = 0.0;
  
  for (int i = 0; i < codeSize; i++) {
    dot += code1[i] * code2[i];
    mag1_sq += code1[i] * code1[i];
    mag2_sq += code2[i] * code2[i];
  }
  
  double denominator = sqrt(mag1_sq) * sqrt(mag2_sq);

  // 1. The Epsilon Check: Check if the denominator is effectively zero
  if (denominator < std::numeric_limits<double>::epsilon()) {
    return M_PI + 0.1; 
  }

  double cosine_sim = dot / denominator;

  // 2. The Domain Check: Protect acos from NaN errors
  if (cosine_sim > 1.0) cosine_sim = 1.0;
  if (cosine_sim < -1.0) cosine_sim = -1.0;

  return acos(cosine_sim);
}

cv::Mat ArcIrisImplFRVT1N::get_cv2_image(const std::shared_ptr<uint8_t> &data, uint16_t width, uint16_t height, bool isRGB) {
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

at::Tensor ArcIrisImplFRVT1N::grid_sample(at::Tensor input, at::Tensor grid) {
  
  torch::AutoGradMode enable_grad(false);
  c10::InferenceMode guard(true);
  
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

void ArcIrisImplFRVT1N::fix_image(cv::Mat &ret) {
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

void ArcIrisImplFRVT1N::segment_and_circApprox(cv::Mat image, map<string, at::Tensor>* seg_im) {
  
  torch::AutoGradMode enable_grad(false);
  c10::InferenceMode guard(true);

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

void ArcIrisImplFRVT1N::cartToPol(cv::Mat image, at::Tensor pupil_xyr, at::Tensor iris_xyr, map<string, at::Tensor>* c2p_im) {

  torch::AutoGradMode enable_grad(false);
  c10::InferenceMode guard(true);
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
}

vector<double> ArcIrisImplFRVT1N::extractCode(at::Tensor image_polar) {
  
  torch::AutoGradMode enable_grad(false);
  c10::InferenceMode guard(true);
  
  at::Tensor image_polar_f = image_polar.to(at::kFloat).clone().detach();
  at::Tensor image_polar_norm = ((image_polar_f / 255.0) - 0.5) / 0.5;
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
  return code;
}

bool ArcIrisImplFRVT1N::hasEnding(const string &fullString, const string &ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

void ArcIrisImplFRVT1N::convert_uint8_to_doublevector(
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

double ArcIrisImplFRVT1N::match(
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
  if (scores.size() == 0) {
    return M_PI + 0.1;
  } else if (scores.size() == 1) {
    return scores[0];
  } else {
    sort(scores.begin(), scores.end());
    if (scores.size() % 2 == 0) {
      int ind = int(scores.size() / 2);
      return (scores[ind] + scores[ind-1]) / 2.0;
    } else {
      int ind = int((scores.size() - 1)/2);
      return scores[ind];
    }
  }
}

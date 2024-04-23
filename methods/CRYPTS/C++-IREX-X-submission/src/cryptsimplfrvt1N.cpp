/*
 * This software was developed at the National Institute of Standards and
 * Technology (NIST) by employees of the Federal Government in the course
 * of their official duties. Pursuant to title 17 Section 105 of the
 * United States Code, this software is not subject to copyright protection
 * and is in the public domain. NIST assumes no responsibility whatsoever for
 * its use by other parties, and makes no guarantees, expressed or implied,
 * about its quality, reliability, or any other characteristic.
 */

#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

#include "cryptsimplfrvt1N.h"
#include "ortools_linprog.h"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <tuple>
#include <utility>
#include <vector>

using namespace std;
using namespace FRVT;
using namespace FRVT_1N;

CryptsImplFRVT1N::CryptsImplFRVT1N() {
    //cerr << "CRYPTS object created" << endl;
    resolution.push_back(0);
    resolution.push_back(0);
    norm_params_mask.push_back(0);
    norm_params_mask.push_back(0);
    norm_params_circle.push_back(0);
    norm_params_circle.push_back(0);
    init = false;
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    cv::setNumThreads(0);
}

CryptsImplFRVT1N::~CryptsImplFRVT1N() {}

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
ReturnStatus CryptsImplFRVT1N::initializeTemplateCreation(const std::string &configDir, FRVT::TemplateRole role) {
  if (init == true) {
    return ReturnStatus(ReturnCode::Success, "Already initialized.");
  }
  //cerr << "Not initialized before, Initializing..." << endl;

  init = true;

  string yamlpath = configDir + "/cfg.yaml";
  load_cfg(yamlpath.c_str());
  
  string fine_mask_model_path = (configDir + "/" + cfg["fine_mask_model_path"]).c_str();
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    mask_model = torch::jit::load(fine_mask_model_path, torch::kCPU);
  } catch (const c10::Error &e) {
    //cerr << "error loading the mask model" << endl;
    return ReturnCode::ConfigError;
  }

  string circle_param_model_path = (configDir + "/" + cfg["circle_param_model_path"]).c_str();
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    circle_model = torch::jit::load(circle_param_model_path, torch::kCPU);
  } catch (const c10::Error &e) {
    //cerr << "error loading the circle model" << endl;
    return ReturnCode::ConfigError;
  }

  polar_height = stoi(cfg["polar_height"]);
  polar_width = stoi(cfg["polar_width"]);

  //cerr << "Setting resolution..." << endl;
  resolution[0] = stoi(cfg["seg_width"]);
  resolution[1] = stoi(cfg["seg_height"]);

  norm_params_mask[0] = stod(cfg["norm_mean_mask"]);
  norm_params_mask[1] = stod(cfg["norm_std_mask"]);
  norm_params_circle[0] = stod(cfg["norm_mean_circle"]);
  norm_params_circle[1] = stod(cfg["norm_std_circle"]);

  CRYPTS_INITIAL_ITERATIVE_STR_EL_RADIUS = stoi(cfg["CRYPTS_INITIAL_ITERATIVE_STR_EL_RADIUS"]);
  CRYPTS_MIN_ITERATIVE_STR_EL_RADIUS = stoi(cfg["CRYPTS_MIN_ITERATIVE_STR_EL_RADIUS"]);
  CRYPTS_SMOOTH_GAUSS_SIZE = stoi(cfg["CRYPTS_SMOOTH_GAUSS_SIZE"]);
  CRYPTS_SMOOTH_GAUSS_SIGMA = stoi(cfg["CRYPTS_SMOOTH_GAUSS_SIGMA"]);
  CRYPTS_BACKGROUND_GAUSS_SIZE = stoi(cfg["CRYPTS_BACKGROUND_GAUSS_SIZE"]);
  CRYPTS_BACKGROUND_GAUSS_SIGMA = stoi(cfg["CRYPTS_BACKGROUND_GAUSS_SIGMA"]);
  CRYPTS_MIN_CRYPT_AREA = stod(cfg["CRYPTS_MIN_CRYPT_AREA"]);
  CRYPTS_MIN_CRYPT_AREA_HS = stod(cfg["CRYPTS_MIN_CRYPT_AREA_HS"]);
  CRYPTS_MAX_CRYP_AREA_LB = stod(cfg["CRYPTS_MAX_CRYP_AREA_LB"]);
  CRYPTS_MIN_CRYP_AREA_UB = stod(cfg["CRYPTS_MIN_CRYP_AREA_UB"]);
  CRYPTS_CUT_STD = stod(cfg["CRYPTS_CUT_STD"]);
  CRYPTS_MATCH_COINCIDENCE_TOLERANCE = stod(cfg["CRYPTS_MATCH_COINCIDENCE_TOLERANCE"]);
  CRYPTS_MATCH_DIST_ALPHA = stod(cfg["CRYPTS_MATCH_DIST_ALPHA"]);
  CRYPTS_MATCH_PDIST = stoi(cfg["CRYPTS_MATCH_PDIST"]);
  CRYPTS_MATCH_MAX_SHIFT = stoi(cfg["CRYPTS_MATCH_MAX_SHIFT"]);
  CRYPTS_GD_MAX_SHIFT = stoi(cfg["CRYPTS_GD_MAX_SHIFT"]);
  CRYPTS_FRACTION_TOTAL_PAIRS = stof(cfg["CRYPTS_FRACTION_TOTAL_PAIRS"]);
  CRYPTS_MASK_THRESHOLD = stoi(cfg["CRYPTS_MASK_THRESHOLD"]);
  return ReturnCode::Success;
}

ReturnStatus CryptsImplFRVT1N::createFaceTemplate(
    const std::vector<FRVT::Image> &faces,
    FRVT::TemplateRole role,
    std::vector<uint8_t> &templ,
    std::vector<FRVT::EyePair> &eyeCoordinates
) {
  return ReturnStatus(ReturnCode::NotImplemented);
}

ReturnStatus CryptsImplFRVT1N::createFaceTemplate(
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
FRVT::ReturnStatus CryptsImplFRVT1N::createIrisTemplate(
    const std::vector<FRVT::Image> &irises,
    FRVT::TemplateRole role,
    std::vector<uint8_t> &templ,
    std::vector<FRVT::IrisAnnulus> &irisLocations
) {
  //cerr << "Creating templates..." << endl;
  for (int i = 0; i < irises.size(); i++) {
    const FRVT::Image &iris = irises[i];

    bool isRGB = (iris.depth == 24);
    cv::Mat irisim = get_cv2_image(iris.data, iris.width, iris.height, isRGB);
    this->fix_image(irisim);
    map<string, at::Tensor> seg_im = this->segment_and_circApprox(irisim.clone());
    //cerr << "Segmentation and circle approximation done." << endl;

    if (irisLocations.size() > i){
      if (irisLocations[i].limbusCenterX != 0) {
          seg_im["iris_xyr"].index({0}) = (float)irisLocations[i].limbusCenterX;
      }else{
        irisLocations[i].limbusCenterX = (uint16_t) seg_im["iris_xyr"][0].item<float>();
      }

      if (irisLocations[i].limbusCenterY != 0) {
          seg_im["iris_xyr"].index({1}) = (float)irisLocations[i].limbusCenterY;
      }else{
        irisLocations[i].limbusCenterY = (uint16_t) seg_im["iris_xyr"][1].item<float>();
      }

      if (irisLocations[i].limbusRadius != 0) {
          seg_im["iris_xyr"].index({2}) = (float)irisLocations[i].limbusRadius;
      }else{
        irisLocations[i].limbusRadius = (uint16_t) seg_im["iris_xyr"][2].item<float>();
      }

      if (irisLocations[i].pupilRadius != 0) {
          seg_im["pupil_xyr"].index({2}) = (float)irisLocations[i].pupilRadius; 
      }else{
        irisLocations[i].pupilRadius = (uint16_t) seg_im["pupil_xyr"][2].item<float>();
      }
    }
    
    if (seg_im["pupil_xyr"].index({2}).item<float>() < 4) {
        //cerr << "The pupil radius is too small." << endl;
        continue;
    }
    
    if (seg_im["iris_xyr"].index({2}).item<float>() < 12) {
        //cerr << "The iris radius is too small." << endl; 
        continue;
    }

    map<string, at::Tensor> c2p_im = this->cartToPol(irisim, seg_im["mask"], seg_im["pupil_xyr"], seg_im["iris_xyr"]);
    at::Tensor img_polar_tensor = c2p_im["image_polar"].contiguous().to(torch::kCPU);
    at::Tensor mask_polar_tensor = c2p_im["mask_polar"].contiguous().to(torch::kCPU);

    if (mask_polar_tensor.sum().item<float>() < CRYPTS_MASK_THRESHOLD) {
      //cerr << "Mask is too small." << endl;
      continue;
    }
    //cerr << "Saving image and mask polar as cv::Mat." << endl;
    cv::Mat image_polar(this->polar_height, this->polar_width, 0);
    memcpy((void *) image_polar.data, img_polar_tensor.data_ptr(), sizeof(torch::kU8) * img_polar_tensor.numel());
    //cv::imwrite("image_polar_check.png", image_polar);
    //cerr << "Tensor image converted to cv" << endl;

    cv::Mat mask_polar(this->polar_height, this->polar_width, 0);
    memcpy((void *) mask_polar.data, mask_polar_tensor.data_ptr(), sizeof(torch::kU8) * mask_polar_tensor.numel());
    //cv::imwrite("mask_polar_check.png", mask_polar);
    //cerr << "Tensor mask converted to cv" << endl;

    // //cerr << "Detecting crypts ..." << endl;
    cv::Mat *im_polar = new cv::Mat(image_polar.clone());
    cv::Mat *m_polar = new cv::Mat(mask_polar.clone());
    cv::Mat crypts;
    this->detectCrypts_emd(
        &crypts, im_polar, m_polar
    ); // detectCrypts_emd(cv::Mat* outputCryptsMask, cv::Mat* inputIris, cv::Mat* inputIrisMask)
    //cerr << "Crypts detected. Adding them to template array" << endl;

    templ.push_back((uint8_t) iris.irisLR);
    crypts.convertTo(crypts, CV_8UC1);

    for (int j = 0; j < this->polar_height; j++) {
      for (int k = 0; k < this->polar_width; k++) {
        templ.push_back(crypts.at<uchar>(j, k));
      }
    }
    //cerr << "Template populated." << endl;

    delete im_polar;
    delete m_polar;
  }

  if (templ.size() == 0) {
    // // cout << "No templates created." << endl;
    return ReturnStatus(ReturnCode::TemplateCreationError);}

  return ReturnStatus(ReturnCode::Success);
}

FRVT::ReturnStatus CryptsImplFRVT1N::createFaceAndIrisTemplate(
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
ReturnStatus CryptsImplFRVT1N::finalizeEnrollment(
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
  //cerr << "Copying edb and manifest to enrollment directory..." << endl;

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
ReturnStatus
CryptsImplFRVT1N::initializeIdentification(const std::string &configDir, const std::string &enrollmentDir) {
  auto edbManifestName = enrollmentDir + "/" + this->manifest;
  auto edbName = enrollmentDir + "/" + this->edb;

  ifstream manifestStream(edbManifestName.c_str());
  if (!manifestStream.is_open()) {
      //cerr << "Failed to open stream for " << edbManifestName << "." << endl;
      return ReturnStatus(ReturnCode::ConfigError);
  }

  ifstream edbStream(edbName, ios::in | ios::binary);
  if (!edbStream.is_open()) {
      //cerr << "Failed to open stream for " << edbName << "." << endl;
      return ReturnStatus(ReturnCode::ConfigError);
  }

  string templId, size, offset;
  while (manifestStream >> templId >> size >> offset) {
      edbStream.seekg(atol(offset.c_str()), ios::beg);
      std::vector<uint8_t> templData(atol(size.c_str()));
      edbStream.read((char*) &templData[0], atol(size.c_str()));

      vector<FRVT::Image::IrisLR> labels;

      PrepDatabaseEntry_emd templateIris;
      templateIris.id = templId;

      vector<cv::Mat> cryptsList;
      convert_uint8_to_emd_crypts(labels, cryptsList, templData);
      //cerr << "uint8 converted to crypts" << endl;
      //cerr << templData.size() << " " << labels.size() << " " << cryptsList.size() << endl;
      for (int idx = 0; idx < labels.size(); idx++) {
        templateIris.searchCrypts[labels[idx]].push_back(cryptsList[idx]);
      }
      //cerr << "Template created ." << endl;
      this->templates_emd.push_back(templateIris);
      //cerr << "Template loaded ." << endl;
  }

  return ReturnStatus(ReturnCode::Success);
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
ReturnStatus CryptsImplFRVT1N::identifyTemplate(
    const std::vector<uint8_t> &idTemplate,
    const uint32_t candidateListLength,
    std::vector<FRVT_1N::Candidate> &candidateList
) {
  if (idTemplate.size() == 0) {
    //cerr << "Template doesn't contain matchable data" << endl;
    return ReturnCode::VerifTemplateError;
  }

  vector<FRVT_1N::Candidate> all_candidates;

  map<FRVT::Image::IrisLR, vector<cv::Mat>> searchCrypts;
  vector<cv::Mat> emptyvec1;
  searchCrypts[FRVT::Image::IrisLR::Unspecified] = emptyvec1;
  vector<cv::Mat> emptyvec2;
  searchCrypts[FRVT::Image::IrisLR::RightIris] = emptyvec2;
  vector<cv::Mat> emptyvec3;
  searchCrypts[FRVT::Image::IrisLR::LeftIris] = emptyvec3;

  vector<FRVT::Image::IrisLR> labels;
  vector<cv::Mat> cryptsList;

  convert_uint8_to_emd_crypts(labels, cryptsList, idTemplate);

  for (int idx = 0; idx < labels.size(); idx++) {
    searchCrypts[labels[idx]].push_back(cryptsList[idx]);
  }

  //cerr << "Template created in identify. " << templates_emd.size() << " " << candidateListLength << endl;

  for (int i = 0; i < templates_emd.size(); i++) {
    float scoreU = -1.0;
    float scoreL = -1.0;
    float scoreR = -1.0;
    float scoreU1 = -1.0;
    float scoreU2 = -1.0;
    float minScore = -1.0;
    ////// cout << "Matching template " << i << endl;
    if (searchCrypts[FRVT::Image::IrisLR::Unspecified].size() > 0 &&
        templates_emd[i].searchCrypts[FRVT::Image::IrisLR::Unspecified].size() > 0) {
      ////// cout << "Both irises are unspecified" << endl;
      scoreU = this->match_emd(
          searchCrypts[FRVT::Image::IrisLR::Unspecified],
          templates_emd[i].searchCrypts[FRVT::Image::IrisLR::Unspecified]
      );
      if (scoreU >= 0.0 && scoreU <= 1.0) {
        minScore = scoreU;
      } else {
        minScore = -1.0;
      }

    } else if (searchCrypts[FRVT::Image::IrisLR::Unspecified].size() > 0) {
      ////// cout << "Search iris unspecified" << endl;
      if (templates_emd[i].searchCrypts[FRVT::Image::IrisLR::LeftIris].size() > 0) {
        scoreU1 = this->match_emd(
            searchCrypts[FRVT::Image::IrisLR::Unspecified], templates_emd[i].searchCrypts[FRVT::Image::IrisLR::LeftIris]
        );
      }
      if (templates_emd[i].searchCrypts[FRVT::Image::IrisLR::RightIris].size() > 0) {
        scoreU2 = this->match_emd(
            searchCrypts[FRVT::Image::IrisLR::Unspecified],
            templates_emd[i].searchCrypts[FRVT::Image::IrisLR::RightIris]
        );
      }
      if (scoreU1 >= 0.0 && scoreU1 <= 1.0 && scoreU2 >= 0.0 && scoreU2 <= 1.0) {
        minScore = min(scoreU1, scoreU2);
      } else if (scoreU2 >= 0.0 && scoreU2 <= 1.0) {
        minScore = scoreU2;
      } else if (scoreU1 >= 0.0 && scoreU1 <= 1.0) {
        minScore = scoreU1;
      } else {
        minScore = -1.0;
      }

    } else if (templates_emd[i].searchCrypts[FRVT::Image::IrisLR::Unspecified].size() > 0) {
      ////// cout << "Template iris unspecified" << endl;
      if (searchCrypts[FRVT::Image::IrisLR::LeftIris].size() > 0) {
        scoreU1 = this->match_emd(
            templates_emd[i].searchCrypts[FRVT::Image::IrisLR::Unspecified], searchCrypts[FRVT::Image::IrisLR::LeftIris]
        );
      }
      if (searchCrypts[FRVT::Image::IrisLR::RightIris].size() > 0) {
        scoreU2 = this->match_emd(
            templates_emd[i].searchCrypts[FRVT::Image::IrisLR::Unspecified],
            searchCrypts[FRVT::Image::IrisLR::RightIris]
        );
      }
      if (scoreU1 >= 0.0 && scoreU1 <= 1.0 && scoreU2 >= 0.0 && scoreU2 <= 1.0) {
        minScore = min(scoreU1, scoreU2);
      } else if (scoreU2 >= 0.0 && scoreU2 <= 1.0) {
        minScore = scoreU2;
      } else if (scoreU1 >= 0.0 && scoreU1 <= 1.0) {
        minScore = scoreU1;
      } else {
        minScore = -1.0;
      }

    } else {
      ////// cout << "Both iris specified" << endl;
      if (searchCrypts[FRVT::Image::IrisLR::LeftIris].size() > 0 &&
          templates_emd[i].searchCrypts[FRVT::Image::IrisLR::LeftIris].size() > 0) {
        scoreL = this->match_emd(
            searchCrypts[FRVT::Image::IrisLR::LeftIris], templates_emd[i].searchCrypts[FRVT::Image::IrisLR::LeftIris]
        );
      }
      if (searchCrypts[FRVT::Image::IrisLR::RightIris].size() > 0 &&
          templates_emd[i].searchCrypts[FRVT::Image::IrisLR::RightIris].size() > 0) {
        scoreR = this->match_emd(
            searchCrypts[FRVT::Image::IrisLR::RightIris], templates_emd[i].searchCrypts[FRVT::Image::IrisLR::RightIris]
        );
      }
      if (scoreL >= 0.0 && scoreL <= 1.0 && scoreR >= 0.0 && scoreR <= 1.0) {
        minScore = min(scoreL, scoreR);
      } else if (scoreR >= 0.0 && scoreR <= 1.0) {
        minScore = scoreR;
      } else if (scoreL >= 0.0 && scoreL <= 1.0) {
        minScore = scoreL;
      } else {
        minScore = -1.0;
      }
    }
    ////// cout << "Score found" << endl;
    // //cerr << "Min score found: " << minScore << endl;
    if (minScore >= 0.0 && minScore <= 1.0) {
      FRVT_1N::Candidate candidate;
      candidate.templateId = templates_emd[i].id;
      candidate.score = minScore;
      all_candidates.push_back(candidate);
    }
    ////// cout << "Candidate added" << endl;
  }

  //cerr << "Scores found." << endl;
  if (all_candidates.size() == 0) {
    //cerr << "No candidates found." << endl;
    return ReturnCode::UnknownError;
  }
  // // cout << "All candidates found" << endl;
  sort(all_candidates.begin(), all_candidates.end(), [](const FRVT_1N::Candidate &lhs, const FRVT_1N::Candidate &rhs) {
    return lhs.score < rhs.score;
  });
  if (all_candidates.size() <= (int) candidateListLength) {
    candidateList = all_candidates;
  } else {
    for (int i = 0; i < (int) candidateListLength; i++)
      candidateList.push_back(all_candidates[i]);
  }
  return ReturnCode::Success;
}

/* Private Code */
int CryptsImplFRVT1N::getPolarHeight() { return (int) this->polar_height; }

int CryptsImplFRVT1N::getPolarWidth() { return (int) this->polar_width; }

cv::Mat normalizeMinMax(const cv::Mat &matrix) {
  double minVal, maxVal;
  cv::minMaxLoc(matrix, &minVal, &maxVal);
  cv::Mat output = matrix.clone();
  output.convertTo(output, CV_32FC1);
  output = (output - minVal) / (maxVal - minVal);
  output = cv::min(output, 1.0);
  output = cv::max(output, 0.0);
  return output;
}

cv::Mat getSE(int strElRad) { // replicating strel('disk', radius, 0) from MATLAB
  if (strElRad <= 0) {
    strElRad = 1;
  }
  int size = strElRad * 2 + 1;
  at::Tensor X = torch::arange(0, size).view({1, size});
  at::Tensor Y = torch::arange(0, size).view({size, 1});
  at::Tensor dist_from_center = torch::sqrt(torch::pow(X - strElRad, 2) + torch::pow(Y - strElRad, 2));
  at::Tensor se_t = torch::where(dist_from_center <= strElRad, 1, 0).to(torch::kU8);
  cv::Mat se(size, size, CV_8UC1);
  memcpy((void *) se.data, se_t.data_ptr(), sizeof(torch::kU8) * se_t.numel());
  return se;
}

/* MATLAB inspired functions. */
/** Implements the MATLAB "imreconstruct" function. */
void imreconstruct(
    cv::Mat *marker, cv::Mat *mask, cv::Mat *output, int connectivity = 8
) { // only supports 2D images, i.e. connectivity is either 4 or 8
  // cv::Mat kernel = getSE(radius);//cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2*radius+1, 2*radius+1));
  cv::Mat kernel;
  if (connectivity == 4) {
    kernel = getSE(1);
  } else {
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  }
  cv::Mat prev(*marker);
  cv::Mat expanded;
  while (true) {
    cv::dilate(prev, expanded, kernel);
    cv::min(expanded, *mask, expanded);
    cv::Mat diff;
    cv::absdiff(expanded, prev, diff);
    double s = cv::sum(diff)[0];
    if (s == 0) {
      // expanded.copyTo(*output);
      *output = expanded.clone();
      break;
    }
    // expanded.copyTo(prev);
    prev = expanded.clone();
  }
}

/** Implements the MATLAB "imfill" function, for the particular case of masks. */
void maskfill(cv::Mat *mask, cv::Mat *output) {
  cv::Mat floodedImg = mask->clone();
  cv::floodFill(floodedImg, cv::Point(0, 0), cv::Scalar(255));
  cv::Mat floodedImgInv = cv::Scalar::all(255) - floodedImg;
  cv::max(*mask, floodedImgInv, *output);
}

/** Implements the MATLAB "bwconncomp" function using cv::connectedComponentsWithStats. */
void bwconncomp(const cv::Mat &bw, vector<float> &areas, vector<cv::Mat> &componentMasks, int conn = 8) {
  cv::Mat labels;
  cv::Mat stats;
  cv::Mat centroids;

  cv::connectedComponentsWithStats(bw, labels, stats, centroids, conn);

  for (int i = 1; i < stats.rows; i++) {
    areas.push_back((float) stats.at<int>(i, cv::CC_STAT_AREA));
    cv::Mat componentMask = (labels == i);
    componentMask.convertTo(componentMask, CV_32FC1);
    componentMask *= 255;
    componentMask.convertTo(componentMask, CV_8UC1);
    componentMasks.push_back(componentMask.clone());
  }
}

/** Implements the MATLAB "bwareaopen" function using cv::connectedComponentsWithStats */
void bwareaopen(const cv::Mat &bw, cv::Mat &output, float minArea, int conn = 8) {
  cv::Mat labels;
  cv::Mat stats;
  cv::Mat centroids;

  cv::connectedComponentsWithStats(bw, labels, stats, centroids, conn);

  cv::Mat outputMask = cv::Mat::zeros(labels.size(), CV_8UC1);

  for (int i = 1; i < stats.rows; i++) {
    float area = (float) stats.at<int>(i, cv::CC_STAT_AREA);
    if (area >= minArea) {
      cv::Mat componentMask = (labels == i);
      componentMask.convertTo(componentMask, CV_32FC1);
      componentMask *= 255;
      componentMask.convertTo(componentMask, CV_8UC1);
      cv::bitwise_or(outputMask, componentMask, outputMask);
    }
  }
  // outputMask.copyTo(output);
  output = outputMask.clone();
}
void shiftImg(cv::Mat &img, int shiftx, int shifty) {
  if (shiftx != 0 || shifty != 0) {

    cv::Mat floatImg = img.clone();
    floatImg.convertTo(floatImg, CV_32FC1, 1.0f / 255.0f);

    auto input_tensor = torch::from_blob(floatImg.data, {1, floatImg.rows, floatImg.cols, 1}, torch::kCPU);
    input_tensor = input_tensor.permute({0, 3, 1, 2});
    at::Tensor zero_tensor = torch::zeros({1, 1, floatImg.rows, floatImg.cols});

    if (shiftx != 0)
      input_tensor = at::roll(input_tensor, shiftx, 3);

    input_tensor = input_tensor.squeeze(0).detach().permute({1, 2, 0}).clamp(0.0, 1.0).to(at::kFloat).to(torch::kCPU);
    cv::Mat output(input_tensor.sizes()[0], input_tensor.sizes()[1], CV_32FC1);
    memcpy((void *) output.data, input_tensor.data_ptr<float>(), sizeof(float) * input_tensor.numel());

    cv::Mat output_im = (output * 255);

    if (shifty != 0) {
      float warp_values[] = {1.0, 0.0, 0.0, 0.0, 1.0, (float) shifty};
      cv::Mat translation_mat = cv::Mat(2, 3, CV_32F, warp_values);
      cv::warpAffine(output_im, output_im, translation_mat, output_im.size());
    }

    output_im.convertTo(output_im, CV_8UC1);
    output_im.copyTo(img);
  }
}
/*Support for Gaussian Blurring with even-sized filters*/
cv::Mat GaussianBlur(
    const cv::Mat &input,
    int filter_size,
    float sigma,
    string tb_padding_type = "replicate",
    string lr_padding_type = "wrap"
) {
  
  auto input_tensor = torch::from_blob(input.data, {1, input.rows, input.cols, 1}, torch::kCPU);
  input_tensor = input_tensor.permute({0, 3, 1, 2});
  cv::Mat gaussKernel1d = cv::getGaussianKernel(filter_size, sigma);
  gaussKernel1d.convertTo(gaussKernel1d, CV_32FC1);
  
  auto gaussKernel1d_tensor = torch::from_blob(gaussKernel1d.data, {1, filter_size, 1, 1}, torch::kCPU);
  gaussKernel1d_tensor = gaussKernel1d_tensor.view({filter_size, 1});
  at::Tensor gaussKernel2d_tensor = torch::matmul(gaussKernel1d_tensor, gaussKernel1d_tensor.transpose(0, 1));
  gaussKernel2d_tensor = gaussKernel2d_tensor.view({1, 1, filter_size, filter_size});

  int64_t pad_size_1;
  int64_t pad_size_2;

  if (filter_size % 2 == 0) {
    pad_size_1 = floor(filter_size / 2) - 1;
    pad_size_2 = floor(filter_size / 2);
  } else {
    pad_size_1 = floor(filter_size / 2);
    pad_size_2 = pad_size_1;
  }

  at::Tensor padded_tensor;

  if (lr_padding_type == "replicate") {
    padded_tensor = torch::nn::functional::pad(
        input_tensor, torch::nn::functional::PadFuncOptions({pad_size_1, pad_size_2, 0, 0}).mode(torch::kReplicate)
    );
  } else if (lr_padding_type == "constant") {
    padded_tensor = torch::nn::functional::pad(
        input_tensor, torch::nn::functional::PadFuncOptions({pad_size_1, pad_size_2, 0, 0}).mode(torch::kConstant)
    );
  } else if (lr_padding_type == "reflect") {
    padded_tensor = torch::nn::functional::pad(
        input_tensor, torch::nn::functional::PadFuncOptions({pad_size_1, pad_size_2, 0, 0}).mode(torch::kReflect)
    );
  } else if (lr_padding_type == "wrap") {
    padded_tensor = torch::nn::functional::pad(
        input_tensor, torch::nn::functional::PadFuncOptions({pad_size_1, pad_size_2, 0, 0}).mode(torch::kCircular)
    );
  }

  if (tb_padding_type == "replicate") {
    padded_tensor = torch::nn::functional::pad(
        padded_tensor, torch::nn::functional::PadFuncOptions({0, 0, pad_size_1, pad_size_2}).mode(torch::kReplicate)
    );
  } else if (tb_padding_type == "constant") {
    padded_tensor = torch::nn::functional::pad(
        padded_tensor, torch::nn::functional::PadFuncOptions({0, 0, pad_size_1, pad_size_2}).mode(torch::kConstant)
    );
  } else if (tb_padding_type == "reflect") {
    padded_tensor = torch::nn::functional::pad(
        padded_tensor, torch::nn::functional::PadFuncOptions({0, 0, pad_size_1, pad_size_2}).mode(torch::kReflect)
    );
  } else if (tb_padding_type == "wrap") {
    padded_tensor = torch::nn::functional::pad(
        padded_tensor, torch::nn::functional::PadFuncOptions({0, 0, pad_size_1, pad_size_2}).mode(torch::kCircular)
    );
  }

  at::Tensor gaussianOutput = torch::nn::functional::conv2d(padded_tensor, gaussKernel2d_tensor);

  gaussianOutput = gaussianOutput.squeeze(0).detach().permute({1, 2, 0}).clamp(0.0, 1.0).to(at::kFloat).to(torch::kCPU);

  cv::Mat output(gaussianOutput.sizes()[0], gaussianOutput.sizes()[1], CV_32FC1);
  memcpy((void *) output.data, gaussianOutput.data_ptr<float>(), sizeof(float) * gaussianOutput.numel());

  return output;
}

/** Implements the proper morpohological operation.
 *  Based on Mr. Jianxu Chen's MATLAB implementation. */
void morphOperator(cv::Mat *inputImage, int strElRad, cv::Mat *occlusionMask, cv::Mat *output) {
  // image dilate and complement
  cv::Mat *aux1 = new cv::Mat();
  cv::Mat se = getSE(strElRad);
  cv::dilate(*inputImage, *aux1, se);
  // cv::imwrite("Morph1_" + to_string(iter) + ".png", aux1->clone()*255);

  cv::Mat *aux2 = new cv::Mat();
  cv::Mat om_inv = 1.0 - occlusionMask->clone();
  cv::multiply(*aux1, om_inv, *aux2);
  delete aux1;
  //////cv::imwrite("Morph2_" + to_string(iter) + ".png", aux2->clone()*255);

  // reconstruct
  cv::Mat *markerImage = new cv::Mat(1.0 - *aux2);
  //////cv::imwrite("Morph_markerImage_" + to_string(iter) + ".png", markerImage->clone()*255);
  cv::Mat *maskImage = new cv::Mat(1.0 - *inputImage);
  //////cv::imwrite("Morph_maskImage_" + to_string(iter) + ".png", maskImage->clone()*255);
  delete aux2;

  cv::Mat *rctR = new cv::Mat();

  imreconstruct(markerImage, maskImage, rctR);
  //////cv::imwrite("Morph_rctr_" + to_string(iter) + ".png", rctR->clone()*255);
  delete markerImage;
  delete maskImage;

  cv::Mat *rct = new cv::Mat(1.0 - *rctR);
  //////cv::imwrite("Morph_rct_" + to_string(iter) + ".png", rct->clone()*255);
  delete rctR;

  cv::Mat *mm = new cv::Mat(*rct - *inputImage);
  delete rct;

  cv::Mat r;
  cv::threshold(*mm, r, 0.0, 1.0, cv::THRESH_BINARY);
  delete mm;

  // remove masked areas
  cv::multiply(r, om_inv, *output);
  //////cv::imwrite("Morph_R-om_" + to_string(iter) + ".png", output->clone()*255);
}



void CryptsImplFRVT1N::convert_uint8_to_emd_crypts(
    vector<FRVT::Image::IrisLR> &labels, vector<cv::Mat> &cryptsList, const vector<uint8_t> &vec
) {
  //cerr << "Converting uint8 to crypts for emd" << endl;
  if (vec.size() == 0) {
    //cerr << "No templates found." << endl;
  } else {
    int ind = 0;
    while (ind < vec.size()) {
      labels.push_back((FRVT::Image::IrisLR) vec[ind]);
      ind++;
      cv::Mat crypts = cv::Mat::zeros(this->polar_height, this->polar_width, CV_8UC1);
      for (int i = 0; i < this->polar_height; i++) {
        for (int j = 0; j < this->polar_width; j++) {
          crypts.at<uchar>(i, j) = vec[ind];
          ind++;
        }
      }
      cryptsList.push_back(crypts);
    }
    //cerr << labels.size() << " " << cryptsList.size() << endl;
    //cerr << "Conversion ended" << endl;
  }
}

float CryptsImplFRVT1N::GroundDistance(const cv::Mat &obj1, const cv::Mat &obj2, bool checkShift) {

  cv::Mat overlap;
  cv::bitwise_and(obj1, obj2, overlap);

  // distance is -1 for infeasible pairs
  float numOverlap = (float) cv::countNonZero(overlap);

  if (numOverlap < 1) {
    return -1.0;
  }

  float siz1 = (float) cv::countNonZero(obj1);
  float siz2 = (float) cv::countNonZero(obj2);

  // symmetric distance
  float sd = ((siz1 - numOverlap) + (siz2 - numOverlap)) / (siz1 + siz2 - numOverlap);
  sd = 1 / (1 + exp(-10 * sd + 5)); // rescale

  // Hausdorff distance
  cv::Mat A(this->polar_height, this->polar_width, CV_8UC1);
  cv::Mat not_overlap;
  cv::bitwise_not(overlap, not_overlap);
  cv::bitwise_and(obj1, not_overlap, A);
  A.convertTo(A, CV_32FC1, 1.0f / 255.0f);

  cv::Mat distB(this->polar_height, this->polar_width, CV_32FC1);
  cv::Mat obj2_inv(this->polar_height, this->polar_width, CV_8UC1);
  cv::bitwise_not(obj2, obj2_inv);
  cv::distanceTransform(obj2_inv, distB, cv::DIST_L2, cv::DIST_MASK_PRECISE, CV_32F);

  cv::Mat B(this->polar_height, this->polar_width, CV_8UC1);
  cv::bitwise_and(obj2, not_overlap, B);
  B.convertTo(B, CV_32FC1, 1.0f / 255.0f);

  cv::Mat distA(this->polar_height, this->polar_width, CV_32FC1);
  cv::Mat obj1_inv(this->polar_height, this->polar_width, CV_8UC1);
  cv::bitwise_not(obj1, obj1_inv);
  cv::distanceTransform(obj1_inv, distA, cv::DIST_L2, cv::DIST_MASK_PRECISE, CV_32F);

  double minVal;
  double d1, d2;
  cv::Point minLoc;
  cv::Point maxLoc;

  cv::Mat A_mul_distB(this->polar_height, this->polar_width, CV_32FC1);
  cv::multiply(A, distB, A_mul_distB);

  cv::minMaxLoc(A_mul_distB, &minVal, &d1, &minLoc, &maxLoc);

  cv::Mat B_mul_distA(this->polar_height, this->polar_width, CV_32FC1);
  cv::multiply(B, distA, B_mul_distA);

  cv::minMaxLoc(B_mul_distA, &minVal, &d2, &minLoc, &maxLoc);

  float dd = max(d1, d2);

  float hd = 1 / (1 + exp(-dd + 3)); // rescale

  /*
  %%%% linear combination of two distances %%%%%
  */
  float alpha = (float) this->CRYPTS_MATCH_DIST_ALPHA;
  float dist = alpha * sd + (1 - alpha) * hd;

  if (checkShift) {
    float minDist = dist;

    // boundary situation

    cv::Mat obj1_copy = obj1.clone();
    cv::Mat obj2_copy = obj2.clone();

    cv::Rect obj1ColsFirstRoI(0, 0, this->CRYPTS_GD_MAX_SHIFT, obj1.rows);
    cv::Rect obj2ColsFirstRoI(0, 0, this->CRYPTS_GD_MAX_SHIFT, obj2.rows);
    cv::Rect obj1ColsLastRoI(obj1.cols - this->CRYPTS_GD_MAX_SHIFT - 1, 0, this->CRYPTS_GD_MAX_SHIFT, obj1.rows);
    cv::Rect obj2ColsLastRoI(obj2.cols - this->CRYPTS_GD_MAX_SHIFT - 1, 0, this->CRYPTS_GD_MAX_SHIFT, obj2.rows);

    if (!(cv::countNonZero(obj1(obj1ColsFirstRoI)) == 0 && cv::countNonZero(obj2(obj2ColsFirstRoI)) == 0)) {
      shiftImg(obj1_copy, this->CRYPTS_GD_MAX_SHIFT, 0);
      shiftImg(obj2_copy, this->CRYPTS_GD_MAX_SHIFT, 0);
    } else if (!(cv::countNonZero(obj1(obj1ColsLastRoI)) == 0 && cv::countNonZero(obj2(obj2ColsLastRoI)) == 0)) {
      shiftImg(obj1_copy, -this->CRYPTS_GD_MAX_SHIFT, 0);
      shiftImg(obj2_copy, -this->CRYPTS_GD_MAX_SHIFT, 0);
    }

    cv::Rect obj1RowsFirstRoI(0, 0, obj1_copy.cols, this->CRYPTS_GD_MAX_SHIFT);
    cv::Rect obj2RowsFirstRoI(0, 0, obj2_copy.cols, this->CRYPTS_GD_MAX_SHIFT);
    cv::Rect obj1RowsLastRoI(0, obj1_copy.rows - this->CRYPTS_GD_MAX_SHIFT, obj1_copy.cols, this->CRYPTS_GD_MAX_SHIFT);
    cv::Rect obj2RowsLastRoI(0, obj2_copy.rows - this->CRYPTS_GD_MAX_SHIFT, obj2_copy.cols, this->CRYPTS_GD_MAX_SHIFT);

    if (!(cv::countNonZero(obj1_copy(obj1RowsFirstRoI)) == 0 && cv::countNonZero(obj2_copy(obj2RowsFirstRoI)) == 0)) {
      shiftImg(obj1_copy, 0, this->CRYPTS_GD_MAX_SHIFT);
      shiftImg(obj2_copy, 0, this->CRYPTS_GD_MAX_SHIFT);
    } else if (!(cv::countNonZero(obj1_copy(obj1RowsLastRoI)) == 0 && cv::countNonZero(obj2_copy(obj2RowsLastRoI)) == 0
               )) {
      shiftImg(obj1_copy, 0, -this->CRYPTS_GD_MAX_SHIFT);
      shiftImg(obj2_copy, 0, -this->CRYPTS_GD_MAX_SHIFT);
    }

    // shift check

    for (int xshift = -this->CRYPTS_GD_MAX_SHIFT; xshift <= this->CRYPTS_GD_MAX_SHIFT; xshift++) {
      for (int yshift = -this->CRYPTS_GD_MAX_SHIFT; yshift <= this->CRYPTS_GD_MAX_SHIFT; yshift++) {
        if (xshift != 0 || yshift != 0) {
          cv::Mat obj1_sx_ty = obj1_copy.clone();
          shiftImg(obj1_sx_ty, xshift, yshift);
          // cv::imwrite("./shift_images_2/" + to_string(xshift) + "_" + to_string(yshift) + ".png", obj1_sx_ty);
          float tdist = this->GroundDistance(obj1_sx_ty, obj2_copy, false);
          if (tdist > 0 && tdist < minDist) {
            // // cout << "GroundDistance: " << xshift << ", " << yshift << endl;
            minDist = tdist;
          }
        }
      }
    }
    dist = minDist;
  }

  return dist;
}

std::shared_ptr<FRVT_1N::Interface> FRVT_1N::Interface::getImplementation() {
  return std::make_shared<CryptsImplFRVT1N>();
}

vector<int> vec_onlyunique(const vector<int> &vec) {
  vector<int> uniques;
  for (int i = 0; i < vec.size(); i++) {
    if (count(uniques.begin(), uniques.end(), vec.at(i)) == 0) {
      uniques.push_back(vec.at(i));
    }
  }
  return uniques;
}
vector<int> vec_threshold(const vector<int> &vec, const int thres) {
  vector<int> thresholded;
  for (int i = 0; i < vec.size(); i++) {
    if (vec.at(i) > thres) {
      thresholded.push_back(vec.at(i));
    }
  }
  return thresholded;
}

float CryptsImplFRVT1N::matchCrypts_emd(const cv::Mat &cryptMask1, const cv::Mat &cryptMask2) {
  /*
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%% step 1: shift compensation %%%%%%%%%%%%%%%%%%%%%
  % Shift the the image in x- and y- direction to find the maximum
  % overlap of I1 and I2. This is meant to compensate for the minor
  % decrepency caused during unwrapping.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  */

  cv::Mat init_overlap;
  cv::bitwise_and(cryptMask1, cryptMask2, init_overlap);
  int MaxOverlap = cv::countNonZero(init_overlap);
  int shiftPosX = 0;
  int shiftPosY = 0;

  cv::Mat tmp = cryptMask1.clone();

  for (int x_shift = -this->CRYPTS_MATCH_MAX_SHIFT; x_shift <= this->CRYPTS_MATCH_MAX_SHIFT; x_shift++) {
    for (int y_shift = -int(round(this->CRYPTS_MATCH_MAX_SHIFT / 3));
         y_shift <= int(round(this->CRYPTS_MATCH_MAX_SHIFT / 3));
         y_shift++) {
      if (x_shift != 0 || y_shift != 0) {
        tmp = cryptMask1.clone();
        shiftImg(tmp, x_shift, y_shift);
        cv::Mat overlap_tmp;
        cv::bitwise_and(tmp, cryptMask2, overlap_tmp);
        int sizeOverlap = cv::countNonZero(overlap_tmp);
        if (sizeOverlap > MaxOverlap) {
          MaxOverlap = sizeOverlap;
          shiftPosX = x_shift;
          shiftPosY = y_shift;
        }
      }
    }
  }

  cv::Mat bestCryptMask1 = cryptMask1.clone();
  if (shiftPosX != 0 || shiftPosY != 0) {
    shiftImg(bestCryptMask1, shiftPosX, shiftPosY);
  }

  /*
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%% step 2: remove unrelated regions %%%%%%%%%%%%%%%%%%%%
  % This step is to remove non-overlapping regions to reduce computation
  % It is OK to comment out this section.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  */

  vector<float> cm1_areas;
  vector<cv::Mat> cm1_componentMasks;
  vector<float> cm1_areas_filt;
  vector<cv::Mat> cm1_componentMasks_filt;
  bwconncomp(bestCryptMask1, cm1_areas, cm1_componentMasks, 4);
  cv::Mat procCryptMask1 = cv::Mat::zeros(this->polar_height, this->polar_width, CV_8UC1);
  int intersect;
  for (int i = 0; i < cm1_componentMasks.size(); i++) {
    cv::Mat overlap_tmp;
    cv::bitwise_and(cm1_componentMasks[i], cryptMask2, overlap_tmp);
    intersect = cv::countNonZero(overlap_tmp);
    if (intersect > 0) {
      // //cerr << "Found overlap for mask1, adding component " << i << endl;
      cv::bitwise_or(procCryptMask1, cm1_componentMasks[i], procCryptMask1);
      cm1_areas_filt.push_back(cm1_areas[i]);
      cm1_componentMasks_filt.push_back(cm1_componentMasks[i]);
    }
  }

  vector<float> cm2_areas;
  vector<cv::Mat> cm2_componentMasks;
  vector<float> cm2_areas_filt;
  vector<cv::Mat> cm2_componentMasks_filt;
  bwconncomp(cryptMask2, cm2_areas, cm2_componentMasks, 4);
  cv::Mat procCryptMask2 = cv::Mat::zeros(this->polar_height, this->polar_width, CV_8UC1);
  for (int i = 0; i < cm2_componentMasks.size(); i++) {
    cv::Mat overlap_tmp;
    cv::bitwise_and(cm2_componentMasks[i], procCryptMask1, overlap_tmp);
    intersect = cv::countNonZero(overlap_tmp);
    if (intersect > 0) {
      cv::bitwise_or(procCryptMask2, cm2_componentMasks[i], procCryptMask2);
      cm2_areas_filt.push_back(cm2_areas[i]);
      cm2_componentMasks_filt.push_back(cm2_componentMasks[i]);
    }
  }

  /*
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%%% step 3: pre-check %%%%%%%%%%%%%%%%%%%%%%%%%%%
  % This is the early rejection step. Namely, obvious non-match will
  % be discarded immediately.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  */

  float a0 = (float) cv::countNonZero(procCryptMask1);
  float b0 = (float) cv::countNonZero(procCryptMask2);
  cv::Mat procOverlap;
  cv::bitwise_and(procCryptMask1, procCryptMask2, procOverlap);
  float c0 = (float) cv::countNonZero(procOverlap);

  float minVal = min(a0, b0);
  float maxVal = max(a0, b0);

  float r1 = minVal / maxVal;
  float r2 = c0 / maxVal;
  float r3 = c0 / minVal;

  if (r1 < 0.5 || r2 < 0.25 || (r1 < 0.7 && r3 < 0.35)) {
    return 1.0;
  }

  /*
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%% step 4: main function %%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  */

  int srcNum = cm1_componentMasks_filt.size();
  int tarNum = cm2_componentMasks_filt.size();
  int varNum = 0;

  cv::Mat distMat = cv::Mat::zeros(srcNum, tarNum, CV_32FC1);


  for (int i = 0; i < srcNum; i++) {
    for (int j = 0; j < tarNum; j++) {
      float dist = this->GroundDistance(cm1_componentMasks_filt[i], cm2_componentMasks_filt[j], false);
      if (dist >= 0.0) {
        distMat.at<float>(i, j) = dist;
        varNum++;
      }
    }
  }

  /* build the LP problem */

  cv::Mat weight = cv::Mat::zeros(1, varNum + srcNum + tarNum, CV_32FC1);
  cv::Mat A = cv::Mat::zeros(srcNum + tarNum, varNum + srcNum + tarNum, CV_32FC1);
  cv::Mat b = cv::Mat::zeros(srcNum + tarNum, 1, CV_32FC1);

  float S1 = 0.0;
  int sig = 0;

  for (int i = 0; i < srcNum; i++) {
    vector<cv::Point> locations;
    cv::Mat row = distMat.row(i);
    cv::findNonZero(row, locations);
    int len = locations.size();

    b.at<float>(i, 0) = cm1_areas_filt[i];
    S1 = S1 + cm1_areas_filt[i];

    for (int j = 0; j < len; j++) {
      int indInTar = locations[j].x;
      weight.at<float>(0, sig) = distMat.at<float>(i, indInTar);

      A.at<float>(i, sig) = 1.0;
      A.at<float>(indInTar + srcNum, sig) = 1.0;
      sig++;
    }
    A.at<float>(i, varNum + i) = 1.0;
  }

  for (int i = sig; i < sig + srcNum + tarNum; i++) {
    weight.at<float>(0, i) = this->CRYPTS_MATCH_PDIST;
  }

  float S2 = 0.0;
  for (int i = 0; i < tarNum; i++) {
    b.at<float>(i + srcNum, 0) = cm2_areas_filt[i];
    S2 = S2 + cm2_areas_filt[i];
    A.at<float>(i + srcNum, varNum + srcNum + i) = 1;
  }

  float beq = min(S1, S2);

  if (cv::countNonZero(A) == 0 || cv::countNonZero(b) == 0) {
    return 1.0;
  }

  vector<float> solution_values;
  
  ortools_linprog::linprog(solution_values, varNum, srcNum, tarNum, weight, A, b, beq);

  sig = 0;
  vector<int> srcList[srcNum];
  vector<int> tarList[tarNum];

  for (int i = 0; i < srcNum; i++) {
    vector<cv::Point> locations;
    cv::Mat row = distMat.row(i);
    cv::findNonZero(row, locations);
    int len = locations.size();
    // // cout << "len: " << len << endl;
    for (int j = 0; j < len; j++) {
      int indInTar = locations[j].x;
      if (solution_values[sig] > this->CRYPTS_MATCH_COINCIDENCE_TOLERANCE * cm1_areas_filt[i] ||
          solution_values[sig] > this->CRYPTS_MATCH_COINCIDENCE_TOLERANCE * cm2_areas_filt[indInTar]) {
        srcList[i].push_back(indInTar);
        tarList[indInTar].push_back(i);
      }
      sig++;
    }
  }

  for (int i = 0; i < srcNum; i++) {
    srcList[i] = vec_onlyunique(srcList[i]);
  }

  for (int i = 0; i < tarNum; i++) {
    tarList[i] = vec_onlyunique(tarList[i]);
  }

  /*
  %%%% this part is to deal with multiple to multiple matching         %%%%
  %%%% namely, when one set of regions match to another set of regions %%%%
  %%%% we should known that.                                           %%%%
  */

  std::vector<int> srcID(srcNum);
  std::vector<int> tarID(tarNum);

  int cid = 0;
  for (int i = 0; i < srcNum; i++) {
    if (srcID[i] == 0) {
      if (srcList[i].size() > 0) {
        cid++;
        vector<int> srcQue;
        srcID[i] = cid;
        for (int j = 0; j < srcList[i].size(); j++) {
          int tid = srcList[i][j];
          tarID[tid] = cid;
          for (int k = 0; k < tarList[tid].size(); k++) {
            if (tarList[tid][k] != i) {
              srcQue.push_back(tarList[tid][k]);
            }
          }
        }
        while (!srcQue.empty()) {
          srcQue = vec_onlyunique(srcQue);
          sort(srcQue.begin(), srcQue.end());
          for (int ii = 0; ii < srcQue.size(); ii++) {
            int sid = srcQue[ii];
            srcQue[ii] = 0;
            srcID[sid] = cid;
            for (int jj = 0; jj < srcList[sid].size(); jj++) {
              int tid = srcList[sid][jj];
              tarID[tid] = cid;
              for (int kk = 0; kk < tarList[tid].size(); kk++) {
                if (srcID[tarList[tid][kk]] == 0) {
                  if (count(srcQue.begin(), srcQue.end(), tarList[tid][kk]) == 0) {
                    srcQue.push_back(tarList[tid][kk]);
                  }
                }
              }
            }
          }
          srcQue = vec_threshold(srcQue, 0);
        }
      }
    }
  }

  /*
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%% step 5: compute a score for the matching result %%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  */

  int numPair = cid;

  vector<tuple<float, float>> score_tuples; // hold score and size ratio of each matched pair

  for (int p = 1; p <= numPair; p++) {
    cv::Mat obj1 = cv::Mat::zeros(this->polar_height, this->polar_width, CV_8UC1);
    for (int i = 0; i < srcNum; i++) {
      if (srcID[i] == p) {
        cv::bitwise_or(obj1, cm1_componentMasks_filt[i], obj1);
      }
    }
    cv::Mat obj2 = cv::Mat::zeros(this->polar_height, this->polar_width, CV_8UC1);
    for (int i = 0; i < tarNum; i++) {
      if (tarID[i] == p) {
        cv::bitwise_or(obj2, cm2_componentMasks_filt[i], obj2);
      }
    }

    float score_p = this->GroundDistance(obj1, obj2, true);

    float sz1 = (float) cv::countNonZero(obj1);
    float sz2 = (float) cv::countNonZero(obj2);
    float tt = min(sz1, sz2) / max(sz1, sz2);
    float sizeRatio = 0.5 * (sz1 + sz2) / (1 + exp(-10 * tt + 2));

    score_tuples.push_back(make_tuple(score_p, sizeRatio));
  }

  int finalNum = ceil(this->CRYPTS_FRACTION_TOTAL_PAIRS * (float) numPair);
  sort(score_tuples.begin(), score_tuples.end());

  float score = 0;
  float sumSize = 0;
  for (int i = 0; i < finalNum; i++) {
    score += get<0>(score_tuples[i]) * get<1>(score_tuples[i]);
    sumSize += get<1>(score_tuples[i]);
  }
  score /= sumSize;
  return score;
}

float CryptsImplFRVT1N::match_emd(const vector<cv::Mat> &cryptsList1, const vector<cv::Mat> &cryptsList2) {
  vector<float> scores;
  for (int i = 0; i < cryptsList1.size(); i++) {
    for (int j = 0; j < cryptsList2.size(); j++) {
      float score = this->matchCrypts_emd(cryptsList1[i], cryptsList2[j]);
      if (score >= 0) {
        scores.push_back(score);
      }
    }
  }

  if (scores.size() > 1) {
    float min_score = *min_element(scores.begin(), scores.end());
    return min_score;
  } else if (scores.size() == 1) {
    return scores[0];
  } else {
    return -1.0;
  }
}
void CryptsImplFRVT1N::load_cfg(string cfg_path) {
  // // cout << "Path to be loaded: " << cfg_path << endl;
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
      //////// cout << string_part << endl;
    } while (end != -1);

    if (options[1][0] == '\"') {
      options[1] = options[1].substr(1, options[1].size() - 2);
    }

    cfg[options[0]] = options[1];
  }

  // Close the file
  fin.close();
}

bool CryptsImplFRVT1N::hasEnding(const string &fullString, const string &ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

cv::Mat
CryptsImplFRVT1N::get_cv2_image(const std::shared_ptr<uint8_t> &data, uint16_t width, uint16_t height, bool isRGB) {

  cv::Mat cv2im((int) height, (int) width, isRGB ? CV_8UC3 : CV_8UC1, data.get());
  if (isRGB == true) {
    // //cerr << "Getting R channel of the RGB image." << endl;
    vector<cv::Mat> channels(3);
    cv::split(cv2im, channels);
    return channels[0];
  } else {
    // //cerr << "Getting the grayscale image." << endl;
    return cv2im;
  }
}

void CryptsImplFRVT1N::fix_image(cv::Mat &ret) {
  // ret.convertTo(ret, CV_32FC3, 1.0f / 255.0f);
  int w = ret.cols;
  int h = ret.rows;
  float aspect_ratio = (float) w / (float) h;
  cv::Scalar value(127, 127, 127);
  if (aspect_ratio >= 1.333 && aspect_ratio <= 1.334) {
    cv::resize(ret, ret, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);
  } else if (aspect_ratio < 1.333) {
    float w_new = h * (4.0 / 3.0);
    //////// cout << w_new << "<-" << w << endl;
    float w_pad = (w_new - w) / 2;
    //////// cout << w_pad << endl;
    int left = (int) w_pad;
    int right = (int) w_pad;
    cv::copyMakeBorder(ret, ret, 0, 0, left, right, cv::BORDER_CONSTANT, value);
    cv::resize(ret, ret, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);
  } else {
    float h_new = w * (3.0 / 4.0);
    //////// cout << h_new << h << endl;
    float h_pad = (h_new - h) / 2;
    //////// cout << h_pad << endl;
    int top = (int) h_pad;
    int bottom = (int) h_pad;
    cv::copyMakeBorder(ret, ret, top, bottom, 0, 0, cv::BORDER_CONSTANT, value);
    cv::resize(ret, ret, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);
  }
}
map<string, at::Tensor> CryptsImplFRVT1N::segment_and_circApprox(cv::Mat image) {
  int w = image.cols;
  int h = image.rows;
  image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

  cv::resize(image, image, cv::Size(resolution[0], resolution[1]), 0, 0, 1);

  // float w_mult = float(w/resolution[0]);
  // float h_mult = float(h/resolution[1]);

  auto input_tensor = torch::from_blob(image.data, {1, resolution[1], resolution[0], 1}, torch::kCPU);
  input_tensor = input_tensor.permute({0, 3, 1, 2});
 
  vector<torch::jit::IValue> *inputs = new vector<torch::jit::IValue>;
  // Mask
  at::Tensor mask_input_tensor = ((input_tensor.clone().detach() - norm_params_mask[0]) / norm_params_mask[1]);
  ////// cout << "Tensor pointer defined" << endl;
  mask_input_tensor = mask_input_tensor.to(torch::kCPU).to(torch::kFloat32);
  ////// cout << "input moved to torch::kCPU." << mask_input_tensor.sizes() << endl;
  inputs->push_back(mask_input_tensor);
  ////// cout << "vector populated with input." << endl;
  at::Tensor out_tensor = mask_model.forward(*inputs).toTensor();
  // mask_input_tensor = mask_input_tensor.to(torch::kCPU);
  inputs->clear();
  ////// cout << "mask model inference done." << endl;
  out_tensor = out_tensor.to(torch::kCPU);
  out_tensor = torch::nn::functional::interpolate(
      out_tensor, torch::nn::functional::InterpolateFuncOptions().size(vector<int64_t> {h, w}).mode(torch::kNearest)
  );
  out_tensor = out_tensor.squeeze();
  at::Tensor mask = at::where(out_tensor > 0.5, 255, 0);
  mask = mask.to(torch::kU8).to(torch::kCPU);
  ////// cout << "mask found." << endl;
  delete inputs;

  // Circle params
  at::Tensor circ_input_tensor = ((input_tensor.clone().detach() - norm_params_circle[0]) / norm_params_circle[1]);

  circ_input_tensor = circ_input_tensor.repeat({1, 3, 1, 1});
  circ_input_tensor = circ_input_tensor.to(torch::kCPU).to(torch::kFloat32);

  inputs = new vector<torch::jit::IValue>;
  inputs->push_back(circ_input_tensor);

  at::Tensor inp_xyr_t = circle_model.forward(*inputs).toTensor();
  // circ_input_tensor = circ_input_tensor.to(torch::kCPU);
  inputs->clear();

  inp_xyr_t = inp_xyr_t.to(torch::kCPU);
  at::Tensor inp_xyr = inp_xyr_t.squeeze();

  float diagonal = sqrt(w * w + h * h);
  inp_xyr[0] *= w;
  inp_xyr[1] *= h;
  inp_xyr[2] *= diagonal * 0.8 * 0.5;
  inp_xyr[3] *= w;
  inp_xyr[4] *= h;
  inp_xyr[5] *= diagonal * 0.5;
  ////// cout << "circle parameters found." << endl;
  delete inputs;

  map<string, at::Tensor> ret;
  ret["pupil_xyr"] = inp_xyr.index({Slice(None, 3)});
  ret["iris_xyr"] = inp_xyr.index({Slice(3, None)});
  ret["mask"] = mask;

  return ret;
}
map<string, at::Tensor>
CryptsImplFRVT1N::cartToPol(cv::Mat &image, at::Tensor &mask, at::Tensor &pupil_xyr, at::Tensor &iris_xyr) {
  int w = image.cols;
  int h = image.rows;
  image.convertTo(image, CV_32FC3);
  auto image_tensor = torch::from_blob(image.data, {1, h, w, 1});
  image_tensor = image_tensor.permute({0, 3, 1, 2});

  mask = mask.unsqueeze({0}).unsqueeze({0}).to(torch::kFloat32);
 
  float width = (float) image_tensor.sizes()[3];
  float height = (float) image_tensor.sizes()[2];

  pupil_xyr = pupil_xyr.unsqueeze({0}).to(torch::kFloat32);
  iris_xyr = iris_xyr.unsqueeze({0}).to(torch::kFloat32);
  at::Tensor theta = 2 * M_PI * (at::linspace(0, polar_width - 1, polar_width) / polar_width);
  theta = theta.to(torch::kFloat32).to(torch::kCPU);

  at::Tensor pxCirclePoints =
      pupil_xyr.index({Slice(None, None), 0}).reshape({-1, 1}) +
      at::matmul(pupil_xyr.index({Slice(None, None), 2}).reshape({-1, 1}), torch::cos(theta).reshape({1, polar_width}));
  at::Tensor pyCirclePoints =
      pupil_xyr.index({Slice(None, None), 1}).reshape({-1, 1}) +
      at::matmul(pupil_xyr.index({Slice(None, None), 2}).reshape({-1, 1}), torch::sin(theta).reshape({1, polar_width}));
  ////// cout << "pxCirclePoints pyCirclePoints defined" << endl;
  at::Tensor ixCirclePoints =
      iris_xyr.index({Slice(None, None), 0}).reshape({-1, 1}) +
      at::matmul(iris_xyr.index({Slice(None, None), 2}).reshape({-1, 1}), torch::cos(theta).reshape({1, polar_width}));
  at::Tensor iyCirclePoints =
      iris_xyr.index({Slice(None, None), 1}).reshape({-1, 1}) +
      at::matmul(iris_xyr.index({Slice(None, None), 2}).reshape({-1, 1}), torch::sin(theta).reshape({1, polar_width}));
  ////// cout << "ixCirclePoints iyCirclePoints defined" << endl;
  at::Tensor radius = (at::linspace(0, polar_height - 1, polar_height) / polar_height).reshape({-1, 1}).to(torch::kCPU);
  // at::Tensor radius = (at::linspace(1, polar_height, polar_height)/polar_height).reshape({-1, 1}).to(torch::kCPU);

  at::Tensor pxCoords = at::matmul((1 - radius), pxCirclePoints.reshape({-1, 1, polar_width}));
  at::Tensor pyCoords = at::matmul((1 - radius), pyCirclePoints.reshape({-1, 1, polar_width}));

  at::Tensor ixCoords = at::matmul(radius, ixCirclePoints.reshape({-1, 1, polar_width}));
  at::Tensor iyCoords = at::matmul(radius, iyCirclePoints.reshape({-1, 1, polar_width}));

  at::Tensor x = (pxCoords + ixCoords).to(torch::kFloat32).to(torch::kCPU);
  at::Tensor x_norm = (x / (width - 1)) * 2 - 1;

  at::Tensor y = (pyCoords + iyCoords).to(torch::kFloat32).to(torch::kCPU);
  at::Tensor y_norm = (y / (height - 1)) * 2 - 1;

  at::Tensor grid_sample_mat = at::cat({x_norm.unsqueeze({-1}), y_norm.unsqueeze({-1})}, -1).to(torch::kCPU);
  at::Tensor image_polar = grid_sample(image_tensor, grid_sample_mat, "bilinear");
  image_polar = at::clamp(at::round(image_polar.index({0, 0, Slice(None, None), Slice(None, None)})), 0, 255);
  at::Tensor mask_polar = grid_sample(mask, grid_sample_mat, "nearest");
  mask_polar = at::where(mask_polar.index({0, 0, Slice(None, None), Slice(None, None)}) < 127.5, 0, 255);

  map<string, at::Tensor> ret;
  ret["image_polar"] = image_polar.clamp(0, 255).squeeze().squeeze().to(torch::kU8);
  ret["mask_polar"] = mask_polar.squeeze().squeeze().to(torch::kU8);

  return ret;
}
at::Tensor CryptsImplFRVT1N::grid_sample(at::Tensor input, at::Tensor grid, string interp_mode) {
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

  // grid sample options
  if (interp_mode.compare("nearest") == 0) {
    auto gridsampleoptions = torch::nn::functional::GridSampleFuncOptions()
                                 .mode(torch::kNearest)
                                 .padding_mode(torch::kZeros)
                                 .align_corners(false);
    at::Tensor ret =
        torch::nn::functional::grid_sample(input.to(torch::kCPU), newgrid.to(torch::kCPU), gridsampleoptions);
    return ret;
  } else {
    auto gridsampleoptions = torch::nn::functional::GridSampleFuncOptions()
                                 .mode(torch::kBilinear)
                                 .padding_mode(torch::kZeros)
                                 .align_corners(false);
    at::Tensor ret =
        torch::nn::functional::grid_sample(input.to(torch::kCPU), newgrid.to(torch::kCPU), gridsampleoptions);
    return ret;
  }
}

void CryptsImplFRVT1N::detectCrypts_emd(cv::Mat *outputCryptsMask, cv::Mat *inputIris, cv::Mat *inputIrisMask) {
  // holds the detected crypts
  // background removal
  cv::Mat R = inputIris->clone();
  R.convertTo(R, CV_32FC1, 1.0f / 255.0f);
  cv::Mat R1 = GaussianBlur(R, this->CRYPTS_BACKGROUND_GAUSS_SIZE, this->CRYPTS_BACKGROUND_GAUSS_SIGMA, "replicate");
  cv::Mat R_diff;
  cv::subtract(R, R1, R_diff);
  cv::normalize(R_diff, R_diff, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);
  // R1 = mat2gray(R1);
  // cv::normalize(R1, R1, 0.0, 1.0, cv::NORM_MINMAX);

  // smooth
  // cv::GaussianBlur(*R1, *I, cv::Size(this->CRYPTS_SMOOTH_GAUSS_SIZE, this->CRYPTS_SMOOTH_GAUSS_SIZE),
  // this->CRYPTS_SMOOTH_GAUSS_SIGMA);
  cv::Mat I = GaussianBlur(R_diff, this->CRYPTS_SMOOTH_GAUSS_SIZE, this->CRYPTS_SMOOTH_GAUSS_SIGMA, "constant");

  // hierarchical segmentation
  cv::Mat *cryptMask = new cv::Mat();
  cv::Mat *irisMask = new cv::Mat(inputIrisMask->clone());
  cv::Mat *invertedIrisMask = new cv::Mat(cv::Scalar::all(255) - inputIrisMask->clone());
  // R1_diff.convertTo(R1, CV_32FC1, 1.0f / 255.0f);
  // I.convertTo(I, CV_32FC1, 1.0f / 255.0f);
  irisMask->convertTo(*irisMask, CV_32FC1, 1.0f / 255.0f);
  invertedIrisMask->convertTo(*invertedIrisMask, CV_32FC1, 1.0f / 255.0f);
  
  /// this->CRYPTS_MAX_CRYP_AREA_LB << ", " << this->CRYPTS_MIN_CRYP_AREA_UB << ", " << this->CRYPTS_CUT_STD << endl;
  this->hierarchicalSegmentation(
      &R_diff,
      &I,
      invertedIrisMask,
      irisMask,
      this->CRYPTS_INITIAL_ITERATIVE_STR_EL_RADIUS,
      this->CRYPTS_MIN_CRYPT_AREA_HS,
      this->CRYPTS_MAX_CRYP_AREA_LB,
      this->CRYPTS_MIN_CRYP_AREA_UB,
      this->CRYPTS_CUT_STD,
      cryptMask
  );
  // delete R1;
  // delete I;
  delete invertedIrisMask;
  delete irisMask;

  // Post processing
  cryptMask->row(0) = cv::Mat::zeros(1, cryptMask->cols, cryptMask->type());
  cryptMask->row(cryptMask->rows - 1) = cv::Mat::zeros(1, cryptMask->cols, cryptMask->type());
  cryptMask->col(0) = cv::Mat::zeros(cryptMask->rows, 1, cryptMask->type());
  cryptMask->col(cryptMask->cols - 1) = cv::Mat::zeros(cryptMask->rows, 1, cryptMask->type());
  cv::Mat output;
  *cryptMask *= 255.0;
  cryptMask->convertTo(*cryptMask, CV_8UC1);
  cryptMask->clone().copyTo(output, *inputIrisMask);
  delete cryptMask;
 
  // break weak connections
  cv::morphologyEx(
      output, output, cv::MORPH_OPEN, getSE(1)
  ); // cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

  // reject small regions
  bwareaopen(output, output, this->CRYPTS_MIN_CRYPT_AREA, 4);

  output.clone().copyTo(*outputCryptsMask);
}
/** Implements the hierarchical detection process of iris crypts.
 *  Returns a mask containing the crypts.
 *  Based on Mr. Jianxu Chen's MATLAB implementation. */
void CryptsImplFRVT1N::hierarchicalSegmentation(
    cv::Mat *bs,
    cv::Mat *Raw,
    cv::Mat *om,
    cv::Mat *mk,
    int strElRad,
    float minArea,
    float maxAreaLowerBound,
    float maxAreaUpperBound,
    float cutStd,
    cv::Mat *output
) {

  /*********************************/
  /* segmentation at current level */
  /*********************************/
  // morphological reconstruction using the scale of current level
  cv::Mat *L1 = new cv::Mat();
  morphOperator(Raw, strElRad, om, L1);

  // apply the mask of previous level
  // cv::Mat *maskedLi = new cv::Mat();
  cv::multiply(*L1, *mk, *L1);
  // if (level <= 6) {
  //     cv::Mat L1im((*L1) * 255);
  //     L1im.convertTo(L1im, CV_8UC1);
  //     cv::imwrite("./hs_output/"+to_string(level)+"/L1_.png", L1im);
  // }

  // clear boundary
  L1->row(0) = cv::Mat::zeros(1, L1->cols, L1->type());
  L1->row(L1->rows - 1) = cv::Mat::zeros(1, L1->cols, L1->type());
  L1->col(0) = cv::Mat::zeros(L1->rows, 1, L1->type());
  L1->col(L1->cols - 1) = cv::Mat::zeros(L1->rows, 1, L1->type());

  // break weak connection
  cv::Mat *R = new cv::Mat();
  cv::morphologyEx(*L1, *R, cv::MORPH_OPEN, getSE(1));
  // cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
  delete L1;
  cv::Mat R_U8((*R) * 255);
  R_U8.convertTo(R_U8, CV_8UC1);
  // if (level <= 6) {
  //     cv::imwrite("./hs_output/"+to_string(level)+"/R.png", R_U8);
  // }

  // remove small regions
  cv::Mat *fea = new cv::Mat(R->size(), CV_8UC1);
  bwareaopen(R_U8, *fea, minArea, 8);
  delete R;
  // if (level <= 6) {
  //     cv::Mat feaim(*fea);
  //     feaim.convertTo(feaim, CV_8UC1);
  //     cv::imwrite("./hs_output/"+to_string(level)+"/fea.png", feaim);
  // }

  // fill holes
  cv::Mat filledFea(fea->size(), CV_8UC1);
  maskfill(fea, &filledFea);
  delete fea;
  // if (level <= 6) {
  //     cv::Mat filledFeaim(filledFea);
  //     filledFeaim.convertTo(filledFeaim, CV_8UC1);
  //     cv::imwrite("./hs_output/"+to_string(level)+"/filledFea.png", filledFeaim);
  // }

  /*********************/
  /* iterative process */
  /*********************/
  cv::Mat iterRegion = cv::Mat::zeros(Raw->size(), CV_8UC1);
  bool flag = false;
  vector<float> areas;
  vector<cv::Mat> componentMasks;
  bwconncomp(filledFea, areas, componentMasks, 8);
  for (int i = 0; i < areas.size(); i++) {
    if (areas[i] > maxAreaLowerBound) {
      if (areas[i] > maxAreaUpperBound) {
        cv::Mat componentMaskInv;
        cv::bitwise_not(componentMasks[i], componentMaskInv);
        cv::bitwise_and(filledFea, componentMaskInv, filledFea);
        cv::bitwise_or(iterRegion, componentMasks[i], iterRegion);
        flag = true;
      } else {
        cv::Mat mean;
        cv::Mat std;
        cv::Mat bs_32FC1(*bs);
        bs_32FC1.convertTo(bs_32FC1, CV_32FC1);
        cv::meanStdDev(bs_32FC1, mean, std, componentMasks[i]);
        if (std.at<float>(0) > this->CRYPTS_CUT_STD) {
          cv::Mat componentMaskInv;
          cv::bitwise_not(componentMasks[i], componentMaskInv);
          cv::bitwise_and(filledFea, componentMaskInv, filledFea);
          cv::bitwise_or(iterRegion, componentMasks[i], iterRegion);
          flag = true;
        }
      }
    }
  }

  cv::Mat new_mask(iterRegion);
  new_mask.convertTo(new_mask, CV_32FC1, 1.0f / 255.0f);

  // need to perform segmentation on the next level
  if (flag) {
    if (strElRad > this->CRYPTS_MIN_ITERATIVE_STR_EL_RADIUS) {
      cv::Mat nextLevelMask;
      hierarchicalSegmentation(
          bs, Raw, om, &new_mask, strElRad - 1, minArea, maxAreaLowerBound, maxAreaUpperBound, cutStd, &nextLevelMask
      );
      cv::bitwise_or(filledFea, nextLevelMask, *output);
    } else
      cv::bitwise_or(filledFea, iterRegion, *output);
  } else
    *output = filledFea.clone();
  // filledFea.copyTo(*output);
  // cv::threshold(filledFea, *output, 0, 255, CV_8UC1);

  // if (level == 0) {
  //     cv::Mat result_check(*output);
  //     result_check *= 255.0;
  //     result_check.convertTo(result_check, CV_8UC1);
  //     cv::imwrite("./hs_output/result.png", result_check);
  // }
}

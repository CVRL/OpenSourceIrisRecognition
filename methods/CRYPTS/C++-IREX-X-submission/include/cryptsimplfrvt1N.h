/*
 * This software was developed at the National Institute of Standards and
 * Technology (NIST) by employees of the Federal Government in the course
 * of their official duties. Pursuant to title 17 Section 105 of the
 * United States Code, this software is not subject to copyright protection
 * and is in the public domain. NIST assumes no responsibility whatsoever for
 * its use by other parties, and makes no guarantees, expressed or implied,
 * about its quality, reliability, or any other characteristic.
 */

#ifndef CRYPTSIMPLFRVT1N_H_
#define CRYPTSIMPLFRVT1N_H_
#define _USE_MATH_DEFINES
#include <ATen/ATen.h>
#include <iostream>
#include <map>
#include <math.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
using namespace std;
using namespace torch::indexing;

#include <frvt1N.h>

/*
 * Declare the implementation class of the FRVT 1:N Interface
 */
namespace FRVT_1N {
  class CryptsImplFRVT1N : public FRVT_1N::Interface {

    private:
      struct PrepDatabaseEntry_emd {
          string id;
          map<FRVT::Image::IrisLR, vector<cv::Mat>> searchCrypts;
          PrepDatabaseEntry_emd() {
            vector<cv::Mat> emptyvec1;
            searchCrypts[FRVT::Image::IrisLR::Unspecified] = emptyvec1;
            vector<cv::Mat> emptyvec2;
            searchCrypts[FRVT::Image::IrisLR::RightIris] = emptyvec2;
            vector<cv::Mat> emptyvec3;
            searchCrypts[FRVT::Image::IrisLR::LeftIris] = emptyvec3;
          }
      };
      int CRYPTS_INITIAL_ITERATIVE_STR_EL_RADIUS;
      int CRYPTS_MIN_ITERATIVE_STR_EL_RADIUS;
      int CRYPTS_BACKGROUND_GAUSS_SIZE;
      int CRYPTS_BACKGROUND_GAUSS_SIGMA;
      int CRYPTS_SMOOTH_GAUSS_SIZE;
      int CRYPTS_SMOOTH_GAUSS_SIGMA;
      float CRYPTS_MIN_CRYPT_AREA;
      float CRYPTS_MIN_CRYPT_AREA_HS;
      float CRYPTS_MAX_CRYP_AREA_LB;
      float CRYPTS_MIN_CRYP_AREA_UB;
      float CRYPTS_CUT_STD;
      float CRYPTS_MATCH_COINCIDENCE_TOLERANCE;
      float CRYPTS_MATCH_DIST_ALPHA;
      int CRYPTS_MATCH_PDIST;
      int CRYPTS_MATCH_MAX_SHIFT;
      int CRYPTS_GD_MAX_SHIFT;
      int CRYPTS_MASK_THRESHOLD;
      float CRYPTS_FRACTION_TOTAL_PAIRS;

      int polar_height;
      int polar_width;
      int cuda; // 0 for false and 1 for true
      vector<int> resolution;

      torch::jit::script::Module mask_model;
      torch::jit::script::Module circle_model;

      vector<float> norm_params_mask;
      vector<float> norm_params_circle;
      
      at::Tensor grid_sample(at::Tensor input, at::Tensor grid, string interp_mode = "bilinear");

      void load_from_cfg(map<string, string> &cfg);
      void fix_image(cv::Mat &ret);
      at::Tensor segment(cv::Mat image);
      map<string, at::Tensor> segment_and_circApprox(cv::Mat image);
      map<string, at::Tensor> cartToPol(cv::Mat &image, at::Tensor &mask, at::Tensor &pupil_xyr, at::Tensor &iris_xyr);

      bool init;
      map<string, string> cfg;
      // vector<FRVT_1N::DatabaseEntry> templates;
      // map<FRVT_1N::IrisImage::Label, vector<at::Tensor>>
      vector<PrepDatabaseEntry_emd> templates_emd;

      void load_cfg(string cfg_path);
      cv::Mat get_cv2_image(const std::shared_ptr<uint8_t> &data, uint16_t width, uint16_t height, bool isRGB);
      bool hasEnding(const string &fullString, const string &ending);
      void convert_uint8_to_emd_crypts(
          vector<FRVT::Image::IrisLR> &labels, vector<cv::Mat> &cryptsList, const vector<uint8_t> &vec
      );
      void hierarchicalSegmentation(
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
      );
      void detectCrypts_emd(cv::Mat *outputCryptsMask, cv::Mat *inputIris, cv::Mat *inputIrisMask);

      float matchCrypts_emd(const cv::Mat &cryptMask1, const cv::Mat &cryptMask2);
      float match_emd(const vector<cv::Mat> &cryptsLis1, const vector<cv::Mat> &cryptsList2);

      float GroundDistance(const cv::Mat &obj1, const cv::Mat &obj2, bool checkShift);

      int getPolarHeight();
      int getPolarWidth();

    public:
      CryptsImplFRVT1N();
      ~CryptsImplFRVT1N() override;

      FRVT::ReturnStatus initializeTemplateCreation(const std::string &configDir, FRVT::TemplateRole role) override;

      FRVT::ReturnStatus createFaceTemplate(
          const std::vector<FRVT::Image> &faces,
          FRVT::TemplateRole role,
          std::vector<uint8_t> &templ,
          std::vector<FRVT::EyePair> &eyeCoordinates
      ) override;

      FRVT::ReturnStatus createFaceTemplate(
          const FRVT::Image &image,
          FRVT::TemplateRole role,
          std::vector<std::vector<uint8_t>> &templs,
          std::vector<FRVT::EyePair> &eyeCoordinates
      ) override;

      virtual FRVT::ReturnStatus createIrisTemplate(
          const std::vector<FRVT::Image> &irises,
          FRVT::TemplateRole role,
          std::vector<uint8_t> &templ,
          std::vector<FRVT::IrisAnnulus> &irisLocations
      ) override;

      virtual FRVT::ReturnStatus createFaceAndIrisTemplate(
          const std::vector<FRVT::Image> &facesIrises, FRVT::TemplateRole role, std::vector<uint8_t> &templ
      ) override;

      FRVT::ReturnStatus finalizeEnrollment(
          const std::string &configDir,
          const std::string &enrollmentDir,
          const std::string &edbName,
          const std::string &edbManifestName,
          FRVT_1N::GalleryType galleryType
      ) override;

      FRVT::ReturnStatus
      initializeIdentification(const std::string &configDir, const std::string &enrollmentDir) override;

      FRVT::ReturnStatus identifyTemplate(
          const std::vector<uint8_t> &idTemplate,
          const uint32_t candidateListLength,
          std::vector<FRVT_1N::Candidate> &candidateList
      ) override;

      static std::shared_ptr<Interface> getImplementation();

    private:
      std::map<std::string, std::vector<uint8_t>> templates;

      const std::string edb {"mei.edb"};
      const std::string manifest {"mei.manifest"};
  };
} // namespace FRVT_1N

#endif /* CRYPTSIMPLFRVT1N_H_ */

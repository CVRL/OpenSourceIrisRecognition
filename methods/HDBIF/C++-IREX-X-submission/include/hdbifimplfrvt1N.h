/*
 * This software was developed at the National Institute of Standards and
 * Technology (NIST) by employees of the Federal Government in the course
 * of their official duties. Pursuant to title 17 Section 105 of the
 * United States Code, this software is not subject to copyright protection
 * and is in the public domain. NIST assumes no responsibility whatsoever for
 * its use by other parties, and makes no guarantees, expressed or implied,
 * about its quality, reliability, or any other characteristic.
 */

#ifndef HdbifImplFRVT1N_H_
#define HdbifImplFRVT1N_H_
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

#include "frvt1N.h"

/*
 * Declare the implementation class of the FRVT 1:N Interface
 */
namespace FRVT_1N {
  class HdbifImplFRVT1N : public FRVT_1N::Interface {
    private:
      struct PrepDatabaseEntry {
          string id;
          map<FRVT::Image::IrisLR, vector<at::Tensor>> searchCodes;
          map<FRVT::Image::IrisLR, vector<at::Tensor>> searchMasks;
          PrepDatabaseEntry() {
            vector<at::Tensor> emptyvec1;
            searchCodes[FRVT::Image::IrisLR::Unspecified] = emptyvec1;
            vector<at::Tensor> emptyvec2;
            searchCodes[FRVT::Image::IrisLR::RightIris] = emptyvec2;
            vector<at::Tensor> emptyvec3;
            searchCodes[FRVT::Image::IrisLR::LeftIris] = emptyvec3;
            vector<at::Tensor> emptyvec4;
            searchMasks[FRVT::Image::IrisLR::Unspecified] = emptyvec4;
            vector<at::Tensor> emptyvec5;
            searchMasks[FRVT::Image::IrisLR::RightIris] = emptyvec5;
            vector<at::Tensor> emptyvec6;
            searchMasks[FRVT::Image::IrisLR::LeftIris] = emptyvec6;
          }
      };
      int debug_count = 0;
      int polar_height;
      int polar_width;
      int filter_size;
      int num_filters;
      int max_shift;
      int cuda; // 0 for false and 1 for true
      string fine_mask_model_path;
      string circle_param_model_path;
      string bsif_dir;
      at::Tensor filter;
      vector<int> resolution;
      torch::jit::script::Module mask_model;
      torch::jit::script::Module circle_model;
      vector<double> norm_params_mask;
      vector<double> norm_params_circle;
      at::Tensor grid_sample(at::Tensor input, at::Tensor grid, string interp_mode = "bilinear");

      void load_from_cfg(map<string, string> &cfg);
      void fix_image(cv::Mat &ret);
      void segment_and_circApprox(cv::Mat image, map<string, at::Tensor>* seg_im);
      void cartToPol(cv::Mat image, at::Tensor mask, at::Tensor pupil_xyr, at::Tensor iris_xyr, map<string, at::Tensor>* c2p_im);
      at::Tensor extractCode(at::Tensor image_polar);
      double matchCodes(at::Tensor code1, at::Tensor code2, at::Tensor mask1_inp, at::Tensor mask2_inp);

      bool init;
      int codeSize0;
      int codeSize1;
      int codeSize2;
      int maskSize0;
      int maskSize1;
      map<string, string> cfg;
      vector<PrepDatabaseEntry> templates;
      void load_cfg(string cfg_path);
      cv::Mat get_cv2_image(const std::shared_ptr<uint8_t> &data, uint16_t width, uint16_t height, bool isRGB);
      bool hasEnding(const string &fullString, const string &ending);
      double
      match(vector<at::Tensor> codes1, vector<at::Tensor> masks1, vector<at::Tensor> codes2, vector<at::Tensor> masks2);
      void convert_uint8_to_tensorvector(
          vector<FRVT::Image::IrisLR> &labels,
          vector<at::Tensor> &codes,
          vector<at::Tensor> &masks,
          const vector<uint8_t> &vec
      );

    public:
      HdbifImplFRVT1N();
      ~HdbifImplFRVT1N() override;

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

      // Testing Routines
      FRVT::ReturnStatus
      insertEnrollment(const uint32_t id, const std::vector<uint8_t> &tmplData);

      std::string
      getConfigValue(const std::string& key);

      // Constants
      const std::string edb {"mei.edb"};
      const std::string manifest {"mei.manifest"};
  };
} // namespace FRVT_1N

#endif /* HdbifImplFRVT1N_H_ */

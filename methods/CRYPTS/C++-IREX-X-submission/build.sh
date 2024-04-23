export ROOT="$PWD"

# Install system tools and dependencies
sudo apt install unzip cmake cmake-curses-gui

# Get Pytorch Dependency
if ! test -d "$ROOT/libtorch/"; then
  wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcpu.zip
  unzip libtorch-cxx11-abi-shared-with-deps-2.2.0+cpu.zip
  rm libtorch-cxx11-abi-shared-with-deps-2.2.0+cpu.zip
fi
export Torch_DIR="$ROOT/libtorch/"

# Get OpenCV Dependency
if ! test -d "$ROOT/opencv/build"; then
  git clone --depth 1 https://github.com/opencv/opencv
  cd "$ROOT/opencv/"
  mkdir build && cd build
  cmake -D_GLIBCXX_USE_CXX11_ABI=1 -DBUILD_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DINSTALL_C_EXAMPLES=OFF -DBUILD_LIST=core,imgproc,imgcodecs \
        -DWITH_GTK=OFF -DWITH_V4L=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_MSMF=OFF -DWITH_CUDA=OFF -DWITH_PNG=OFF -DWITH_JPEG=OFF -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF \
        -DWITH_DSHOW=OFF -DWITH_AVFOUNDATION=OFF -DWITH_1394=OFF -DWITH_ANDROID-MEDIANDK=OFF -DVIDEOIO_ENABLE_PLUGINS=OFF -DWITH_PROTOBUF=OFF -DBUILD_PROTOBUF=OFF -DOPENCV_DNN_OPENCL=OFF -DENABLE_PYLINT=OFF -DENABLE_FLAKE8=OFF  \
        -DBUILD_JAVA=OFF -DBUILD_FAT_JAVA_LIB=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DWITH_PTHREADS_PF=OFF -DPARALLEL_ENABLE_PLUGINS=OFF -DWITH_WIN32UI=OFF -DHIGHGUI_ENABLE_PLUGINS=OFF -DCMAKE_INSTALL_PREFIX=./lib -DCMAKE_BUILD_TYPE=Release ..
  cmake --build . --config Release
fi
export OpenCV_DIR="$ROOT/opencv/build"

cd "$ROOT"

# Clone FRVT
if ! test -d "$ROOT/frvt"; then
  git clone --depth 1 https://github.com/usnistgov/frvt "$ROOT/frvt"
  export FRVT_DIR="$ROOT/frvt"
else
  if ! test -d "$ROOT/frvt/1N/build"; then
    rm -vrf "$ROOT/frvt/1N/build"
  fi
  if ! test -d "$ROOT/frvt/1N/doc"; then
    rm -vrf "$ROOT/frvt/1N/doc"
  fi
  if ! test -d "$ROOT/frvt/1N/bin"; then
    rm -vrf "$ROOT/frvt/1N/bin"
  fi
fi

# Copy FRVT headers
cp "$ROOT/frvt/1N/src/include/frvt1N.h" "$ROOT/NIST-IREX-X/crypts/include/"
cp "$ROOT/frvt/common/src/include/frvt_structs.h" "$ROOT/NIST-IREX-X/crypts/include/"

# Clone or-tools
if ! test -d "$ROOT/or-tools"; then
  git clone --depth 1 --branch v9.2 https://github.com/google/or-tools
fi

# Remove previous build
if test -d "$ROOT/build"; then
  rm -vrf "$ROOT/build"
fi

# Configure and build the library
mkdir build && cd build
cmake -DFRVT_VER="$FRVT_VER" -DFRVT_DIR="$ROOT/frvt" -DTorch_DIR="$ROOT/libtorch/" -DOpenCV_DIR="$ROOT/opencv/build" -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_DEPS:BOOL=ON -DBUILD_EXAMPLES=OFF -DBUILD_SAMPLES=OFF -DUSE_SCIP=OFF ..
cmake --build . --config Release

# Allow override of the checked OS Version
if test -n "$FRVT_OS_VER"; then
   sed -i "s/20.04.3/$FRVT_OS_VER/g"  "$FRVT_DIR/common/scripts/utils.sh"
fi

# Decode Images
if ! test -d "$ROOT/frvt/common/images/iris/images/"; then
  cd "$ROOT/frvt/common/images/iris"
  echo "$PASSWORD" | gpg --pinentry-mode loopback --passphrase-fd 0 --output "NIST_validation_images.tar" --decrypt NIST_validation_images.tar.gz.gpg
  tar xvf NIST_validation_images.tar
fi

mkdir "$ROOT/frvt/1N/doc" || true
echo "$FRVT_VER" > "$ROOT/frvt/1N/doc/version.txt"

cd "$ROOT"

# Copy library to FRVT repo
if ! test -d "$ROOT/frvt/1N/lib/"; then
  mkdir "$ROOT/frvt/1N/lib/"
fi
cp "$ROOT/build/libfrvt_1N_nd_cvrl_crypts_${FRVT_VER}.so" "$ROOT/frvt/1N/lib/"
cp "$ROOT/build/lib/"* "$ROOT/frvt/1N/lib/"
cp "$ROOT/libtorch/lib/"* "$ROOT/frvt/1N/lib/"
cp "$ROOT/opencv/build/lib/"libopencv_imgproc* "$ROOT/frvt/1N/lib/"
cp "$ROOT/opencv/build/lib/"libopencv_imgcodecs* "$ROOT/frvt/1N/lib/"
cp "$ROOT/opencv/build/lib/"libopencv_core* "$ROOT/frvt/1N/lib/"

# Move to FRVT 1N directory to run validation
cd "$ROOT/frvt/1N"

# Copy HDBIF config
if ! test -d "$ROOT/frvt/1N/config/"; then
  cp -r "$ROOT/config/" .
fi

# Run Validation
./run_validate_1N.sh

## Setting up the compilation environment

We recommend to set up an environment with [Ubuntu 20.04.3 as provided by NIST](https://nigos.nist.gov/evaluations/ubuntu-20.04.3-live-server-amd64.iso). It is best if a clean installation of the given ISO is used. The IREX validation routine has a check for the OS version and it will not run unless you are working on Ubuntu 20.04.3. However, if you absolutely have to compile it with any other version of Ubuntu, you can circumvent this OS version check by using:

```sh
export FRVT_OS_VER=<your_ubuntu_version>
# export FRVT_OS_VER=20.04.6 if your ubuntu version is 20.04.6
```
 
For example, if you are using Ubuntu 20.04.6, you can set the FRVT_OS_VER to your Ubuntu version 20.04.6. While we have compiled the library in Ubuntu 20.04.6 and found it to work fine on our test bench run on Ubuntu 20.04.3, we do not recommend doing this.

## Compiling the libraries

The steps for building the libraries using our provided script file are the same for both libraries and are provided below. The description of what each script file does is provided in the corresponding library folder.

IREX requires any library submitted to have a version number for each submission. This version number must be saved in a '.txt' file and saved into the 'doc' directory inside the validation routine folder. This is done by our script file but we have to set this library version number by:

```sh
export FRVT_VER=<version_number>
# export FRVT_VER=000 for your first submission to NIST
```

This library version is the current version for your submission, e.g. if you are submitting it for the first time then, you should set the version number as 000 and therefore, you should run 'export FRVT_VER=000'. 

To run our segmentation models we need (a) the model weights and (b) to set the parameters for our iris recognition model we need the appropriate config file (should be named 'cfg.yaml') inside a directory named 'config' in our library source directory. You can download the zip file containing this 'config' directory from [config-directory-zip](https://notredame.box.com/shared/static/blm6oq6cv41mr12e5gwgoitn7w7qeucz.zip).

Extract the downloaded zip file and copy the 'config' folder into the corresponding library source directory you want to compile ('C++-IREX-X-submission').

```sh
wget -O config.zip https://notredame.box.com/shared/static/blm6oq6cv41mr12e5gwgoitn7w7qeucz.zip
unzip config.zip
cp -r config C++-IREX-X-submission/
rm -r config.zip config
```

Next, we have to extract the IREX validation images. To do so, we need the password for the encrypted compressed file. Please send an email to irex@nist.gov for the password. Set the password to the 'PASSWORD' variable by using:

```sh
export PASSWORD=<password_from_nist>
```

Now, you can build each shared library file by navigating to the appropriate library source directory and running the corresponding script:

```sh
# cd C++-IREX-X-submission
./build.sh
```

The script file also runs the IREX validation routine to generate the package that must be submitted to NIST. From our experience, the IREX validation routine requires a good amount of RAM as it loads all the templates into RAM before matching.

## Ensuring that the library works before official submission

After the validation completes, it should generate the package for submission. One important thing to note is that if you have not copied all the dependencies into the 'lib' folder before running the validation routine, the package might still be generated as the routine can find the default installations for these dependencies if the dependencies are already installed in the system. It is therefore wise to take this generated package, extract the library files, and run the FRVT validation routine on a fresh installation of the [iso provided by NIST (Ubuntu 20.04.3)](https://nigos.nist.gov/evaluations/ubuntu-20.04.3-live-server-amd64.iso). If you want, you can skip running the validation routine in your compilation machine (you would have to delete the appropriate lines from the script file), and simply copy over the required library files and run the validation routine on the fresh machine.

## Description of the `build.sh` script file

Here's an overall description of everything that is done by the script file provided. You can follow the instructions below instead of running the script file if you want.

1. We set the current working directory as 'ROOT' as it helps in navigating.

   ```sh
   export ROOT="$PWD"
   ```

2. We build our library using CMake. We also utilize unzip to extract downloaded zip files. As such, the first instruction in the script is to install CMake and unzip.

   ```sh
   sudo apt install unzip cmake cmake-curses-gui
   ```

3. We download and extract the Libtorch library:

   ```sh
   wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-static-with-deps-2.0.0%2Bcpu.zip
   unzip libtorch-cxx11-abi-static-with-deps-2.0.0+cpu.zip
   rm libtorch-cxx11-abi-static-with-deps-2.0.0+cpu.zip
   ```

4. We get the OpenCV library, configure and compile it for our use-case:

   ```sh
   git clone --depth 1 https://github.com/opencv/opencv
   cd "$ROOT/opencv/"
   mkdir build && cd build
   cmake -DBUILD_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DINSTALL_C_EXAMPLES=OFF -DBUILD_LIST=core,imgproc,imgcodecs \
        -DWITH_GTK=OFF -DWITH_V4L=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_MSMF=OFF -DWITH_CUDA=OFF -DWITH_PNG=OFF \
        -DWITH_JPEG=OFF -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF \
        -DWITH_DSHOW=OFF -DWITH_AVFOUNDATION=OFF -DWITH_1394=OFF -DWITH_ANDROID-MEDIANDK=OFF -DVIDEOIO_ENABLE_PLUGINS=OFF \
        -DWITH_PROTOBUF=OFF -DBUILD_PROTOBUF=OFF -DOPENCV_DNN_OPENCL=OFF -DENABLE_PYLINT=OFF -DENABLE_FLAKE8=OFF  \
        -DBUILD_JAVA=OFF -DBUILD_FAT_JAVA_LIB=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DWITH_PTHREADS_PF=OFF \
        -DPARALLEL_ENABLE_PLUGINS=OFF -DWITH_WIN32UI=OFF -DHIGHGUI_ENABLE_PLUGINS=OFF -DCMAKE_INSTALL_PREFIX=./lib -DCMAKE_BUILD_TYPE=Release ..
   cmake --build . --config Release
   export OpenCV_DIR="$ROOT/opencv/build"
   ```

5. We clone the FRVT repo for the validation routine. 

   ```sh
   cd "$ROOT"
   git clone --depth 1 https://github.com/usnistgov/frvt "$ROOT/frvt"
   export FRVT_DIR="$ROOT/frvt"
   ```

   Optional: If you have an Ubuntu version different from 20.04.3, you can still run the validation routine by setting the FRVT_OS_VER variable by:

   ```sh
   export FRVT_OS_VER=<your_ubuntu_version>
   sed -i "s/20.04.3/$FRVT_OS_VER/g"  "$FRVT_DIR/common/scripts/utils.sh"
   ```

7. We configure and compile the HDBIF library:

   ```sh
   mkdir build && cd build
   cmake -DFRVT_VER="$FRVT_VER" -DFRVT_DIR="$ROOT/frvt" -DTorch_DIR="$ROOT/libtorch/" -DOpenCV_DIR="$ROOT/opencv/build" -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release ..
   cmake --build . --config Release
   ```

   This will compile our library, now we have to run the validation routine using the library file generated.

8. Next, we extract the validation images:

   ```sh
   cd "$ROOT/frvt/common/images/iris"
   echo "$PASSWORD" | gpg --pinentry-mode loopback --passphrase-fd 0 --output "NIST_validation_images.tar" --decrypt NIST_validation_images.tar.gz.gpg
   tar xvf NIST_validation_images.tar
   ```

9. We set the library version number for the FRVT validation routine:

   ```sh
   mkdir "$ROOT/frvt/1N/doc" || true
   echo "$FRVT_VER" > "$ROOT/frvt/1N/doc/version.txt"
   ```

10. We copy the library and the required dependencies to the FRVT repo.

    ```sh
    mkdir "$ROOT/frvt/1N/lib"
    cp "$ROOT/build/libfrvt_1N_nd_cvrl_hdbif_${FRVT_VER}.so" "$ROOT/frvt/1N/lib/"
    cp "$ROOT/libtorch/lib/"lib* "$ROOT/frvt/1N/lib/"
    cp "$ROOT/opencv/build/lib/"libopencv_imgproc* "$ROOT/frvt/1N/lib/"
    cp "$ROOT/opencv/build/lib/"libopencv_imgcodecs* "$ROOT/frvt/1N/lib/"
    cp "$ROOT/opencv/build/lib/"libopencv_core" "$ROOT/frvt/1N/lib/"
    ```

11. We copy the config directory into the FRVT repo:

    ```sh
    cp -r "$ROOT/config" "$ROOT/frvt/1N/"
    ```

11. Finally, we move to the FRVT 1N directory and run the validation routine:

    ```sh
    cd "$ROOT/frvt/1N"
    ./run_validate_1N.sh
    ```

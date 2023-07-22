# AMH_DL_frontend

Let me know if anyone has any questions!

Also tested on:

Backlit (channel 0):
\\nif-prgms-srvr\NIF-Metrology\Microscopy\2022\October 2022\October 12 2022\221012222348385\715297_221012222348385_N220721-001-000_Nikon
 
Toplit (channel 1):
\\nif-prgms-srvr\NIF-Metrology\Microscopy\2022\October 2022\October 12 2022\221012222349233\715297_221012222349233_N220721-001-000_Nikon
 
Ringlit (channel 2):
\\nif-prgms-srvr\NIF-Metrology\Microscopy\2022\October 2022\October 12 2022\221012222348833\715297_221012222348833_N220721-001-000_Nikon



## Building on Pascal (special case) otherwise you just need libtorch, CUDA, and OpenCV installed and for cmake your -DCMAKE_PREFIX_PATH for libtorch below (not all 3):

Currently Loaded Modules:
  1) texlive/2016       3) cuda//11.3.0   5) mvapich2/2.3
  2) StdEnv       (S)   4) gcc/8.3.1      6) cmake/3.14.5
 


module keyword "category: compiler"
module load cuda/11.3.0
ml cuda/11.3.0

### modify the path to openCV and libtorch (of course double check OS and nvcc version etc for proper libtorch download)

cmake -DCMAKE_PREFIX_PATH=/usr/workspace/yancey5/stuff/code/project/libtorch2/libtorch -DCUDNN_LIBRARY_PATH=/collab/usr/global/tools/nvidia/cudnn/toss_3_x86_64_ib/cudnn-8.1.1/lib64 -DCUDNN_INCLUDE_PATH=/collab/usr/global/tools/nvidia/cudnn/toss_3_x86_64_ib/cudnn-8.1.1/include/ cmake -DCMAKE_PREFIX_PATH=/usr/workspace/yancey5/spack/lee218/install/lib64/cmake/opencv4/ -DTorch_DIR=/usr/workspace/yancey5/stuff/code/project/libtorch2/libtorch/share/cmake/Torch ..

cmake --build . --config Release

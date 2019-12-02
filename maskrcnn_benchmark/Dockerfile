ARG CUDA="10.0"
ARG CUDNN="7"
ARG TORCH_NAME="pytorch-1.3.0-py3.7_cuda10.1.243_cudnn7.6.3_0.tar.bz2"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update
RUN apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++
RUN apt-get install -y --no-install-recommends \
      build-essential \
      git \
      libgoogle-glog-dev \
      libgtest-dev \
      libiomp-dev \
      libleveldb-dev \
      liblmdb-dev \
      libopencv-dev \
      libopenmpi-dev \
      libsnappy-dev \
      libprotobuf-dev \
      openmpi-bin \
      openmpi-doc \
      protobuf-compiler \
      libgflags-dev \
      cmake \
      vim

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh
ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.7 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name mask python=3.7.4 \
 && /miniconda/bin/conda clean -ya

# install torch
SHELL ["/bin/bash", "-c"]
RUN source activate mask
RUN conda install -y ipython mkl
RUN pip install ninja yacs cython matplotlib tqdm opencv-python requests numpy
COPY pytorch-1.3.0-py3.7_cuda10.1.243_cudnn7.6.3_0.tar.bz2 /home/pytorch-1.3.0-py3.7_cuda10.1.243_cudnn7.6.3_0.tar.bz2
RUN conda install --name mask --use-local "/home/pytorch-1.3.0-py3.7_cuda10.1.243_cudnn7.6.3_0.tar.bz2"
RUN rm -rf /home/pytorch-1.3.0-py3.7_cuda10.1.243_cudnn7.6.3_0.tar.bz2
RUN pip install torchvision

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
  && cd cocoapi/PythonAPI \
  && python setup.py build_ext install

# install pycocotools
RUN git clone https://github.com/mcordts/cityscapesScripts.git \
  && cd cityscapesScripts \
  && python setup.py build_ext install

# install apex
# 如果本机的cuda版本与编译pytorch的cuda版本不同，此句会出错
RUN git clone https://github.com/NVIDIA/apex.git \
  && cd apex \
  && python setup.py install --cuda_ext --cpp_ext

# install PyTorch
RUN git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
  && cd maskrcnn-benchmark \
  && python setup.py build develop

RUN rm -rf /apex /cityscapesScripts /cocoapi

# install jupyter
RUN pip install jupyter jupyterlab
RUN jupyter notebook --generate-config
COPY "jupyter_notebook_config.py" "/root/.jupyter/jupyter_notebook_config.py"

#docker stop pytorchLearn
#docker rm pytorchLearn
#nvidia-docker run --name pytorchLearn -ti \
#-p 14271:14271 -e PASSWORD=54321 \
#-v /home:/home \
#-v /data:/data  anxu829/pytorch_detectron:test   \
#/bin/bash -c "cd / && jupyter lab --port 14279 --ip 0.0.0.0 --allow-root"
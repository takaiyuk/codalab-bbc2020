# docker build . -t jupyter -f ./docker/jupyter/Dockerfile
# docker run -d --name ${PROJECT_NAME} -p 8888:8888 -v ${PWD}:/workspace/ -v ${HOME}/.kaggle:/root/.kaggle/ jupyter
# docker exec -it ${CONTAINER_ID} /bin/bash
# jupyter notebook list
FROM python:3.7.7-slim
ENV APP_HOME /workspace
WORKDIR $APP_HOME
ADD requirements.txt $APP_HOME
ADD setup.py $APP_HOME
RUN mkdir /root/.kaggle \
  && apt-get -y update \
  && apt-get -y install clang cmake git nodejs npm vim \
  && git clone --recursive https://github.com/microsoft/LightGBM && cd LightGBM && mkdir build && cd build && cmake .. && make -j4 && cd ../python-package && python setup.py install && cd ../.. && rm -r LightGBM \
  && git clone --recursive https://github.com/dmlc/xgboost && cd xgboost && mkdir build && cd build && cmake .. && make -j$(nproc) && cd ../python-package && python setup.py install && cd ../.. && rm -r xgboost \
  && pip install -r requirements.txt \
  && jupyter labextension install @jupyterlab/toc \
  && jupyter nbextension enable --py widgetsnbextension
CMD jupyter lab \
  --no-browser \
  --port=8888 \
  --ip=0.0.0.0 \
  --allow-root

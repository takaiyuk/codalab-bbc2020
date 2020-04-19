# docker build . -t mlflow -f ./docker/mlflow/Dockerfile
# docker run -d --name ${PROJECT_NAME}-mlflow -p 5000:5000 -v ${PWD}:/workspace/ mlflow
# http://0.0.0.0:5000
FROM python:3.7.7-slim
ENV APP_HOME /workspace
WORKDIR $APP_HOME
RUN pip install mlflow==1.7.2
CMD mlflow ui \
  --port=5000 \
  --host=0.0.0.0

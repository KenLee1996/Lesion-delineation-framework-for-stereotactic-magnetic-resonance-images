FROM pytorch/pytorch:latest
RUN pip install --upgrade pip setuptools &&\
pip install nibabel pandas scikit-learn scikit-image matplotlib openpyxl xlrd simpleitk antspyx flwr jupyter ultralytics opencv-python-headless pydicom
CMD jupyter notebook --allow-root --no-browser --port 8888 --ip 0.0.0.0 --NotebookApp.token='' --NotebookApp.password=''
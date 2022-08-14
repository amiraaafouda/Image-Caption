FROM python:3.8

# create a work directory
RUN mkdir /app

# navigate to this work directory
WORKDIR /app

# Copy requirements
COPY ./requirements.txt ./requirements.txt

# Install dependencies
RUN python -m pip install --upgrade pip
RUN pip install tensorflow==2.6.0 -f https://tf.kmtea.eu/whl/stable.html

RUN apt-get update && apt-get -y install sudo
RUN sudo apt-get install -y libhdf5-dev

# RUN sudo apt install espeak

RUN pip install -r requirements.txt

# RUN pip install cython
# RUN pip install h5py==3.2.1

#Copy all files
COPY . .

# Run
CMD ["python","upload.py"]


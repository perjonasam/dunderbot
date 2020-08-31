FROM python:3.7.3

# mkdir /code && cd /code
WORKDIR /code

COPY install-systemlibs.sh .

# Install System Libraries
RUN bash install-systemlibs.sh

# Install packages
RUN pip install poetry

# Add package dependencies
COPY pyproject.toml pyproject.toml

# Lock versions (but upgrade pip for OpenCV) and install
RUN poetry lock
RUN poetry run pip install --upgrade pip
RUN poetry run pip install --upgrade setuptools
RUN poetry install 

# To get stable baselines importing opencv-python to work
RUN apt-get update
RUN apt-get install libgl1-mesa-glx libjpeg62 -y

# Copy source code
COPY . ./

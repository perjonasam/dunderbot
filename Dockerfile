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
#COPY poetry.lock poetry.lock

RUN poetry lock
RUN poetry run pip install --upgrade pip
RUN poetry install 


# Copy source code
COPY . ./

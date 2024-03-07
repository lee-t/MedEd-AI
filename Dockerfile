FROM python:3.11.4

# Set the working directory in the container
WORKDIR /usr/src/app

RUN mkdir /Code
RUN mkdir /Credentials
# Copy the current directory contents into the container at /usr/src/app
COPY ./Code /Code
COPY ./Credentials /Credentials

ENV PYTHONUNBUFFERED=1

RUN python3 -m pip install --upgrade pip

COPY /Code/requirements.txt /usr/src/app/
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /Code/requirements.txt

# Install any needed packages specified in requirements.txt
# RUN python3 -m pip install -r /Code/requirements.txt
# RUN python3 -m pip install --upgrade google-api-python-client oauth2client

# Run main.py when the container launches
CMD ["python3", "/Code/createQuiz.py"]
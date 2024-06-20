FROM chromadb/chroma

# Set the working directory in the container
WORKDIR /usr/src/app

RUN mkdir /Code
RUN mkdir /Data/studentfiles
RUN mkdir /Credentials

# Copy the current directory contents into the container at /usr/src/app
COPY ./Code /Code
COPY ./Credentials /Credentials
COPY /Data/meded_ai_dev.duckdb /usr/src/app/Code/
COPY /Data /usr/src/app/Data
COPY /Code/requirements.txt /usr/src/app/

ENV PYTHONUNBUFFERED=1

RUN python3 -m pip install --upgrade pip

RUN apt-get update --fix-missing && apt-get install -y --fix-missing \
    build-essential \
    gcc \
    g++ \
    musl-dev \
    python3-dev \
    docker-compose \
    cmake \
    libffi-dev \
    autoconf && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /install

RUN python3 -m pip install -r /Code/requirements.txt

EXPOSE 8787

ENTRYPOINT ["jupyter-lab", "/Code/autovectorRAG.ipynb", "--ip", "0.0.0.0","--port", "8787", "--allow-root"]
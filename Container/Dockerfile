FROM python:3.11.4

# Set the working directory in the container
WORKDIR /usr/src/app

RUN mkdir /Code
RUN mkdir /Data
RUN mkdir /Credentials

WORKDIR /Data
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY ../Code/pwyRAG /usr/src/app/Code/
COPY ../Code/requirements.txt /usr/src/app/Code/
COPY ../Credentials /usr/src/app/Credentials
COPY ../Data/meded_ai_dev.duckdb /usr/src/app/Data/
COPY ../Data /usr/src/app/Data

RUN python3 -m pip install -r /usr/src/app/Container/requirements.txt

RUN python3 -m pip install --upgrade duckdb docx2txt

EXPOSE 8787

ENTRYPOINT ["jupyter-lab", "/usr/src/app/Code/pwy100chat.ipynb", "--ip", "0.0.0.0","--port", "8787", "--allow-root"]
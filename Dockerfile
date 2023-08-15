FROM python:3.11

WORKDIR /app

RUN apt update && apt upgrade -y && apt install -y libgdal-dev && apt install -y gdal-bin && apt install -y python3-venv

COPY requirements.in .
RUN sed -i "s/<GDAL_VERSION>/`gdalinfo --version | awk '{print $2}' | sed 's/,//'`/g"  requirements.in 

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip && pip install pip-tools && pip-compile requirements.in && pip install -r requirements.txt

COPY scripts/ scripts/

RUN groupadd stdl -g 9999
USER 9999:9999



FROM python:3.11

RUN apt update && apt upgrade -y && apt install -y libgdal-dev && apt install -y gdal-bin && apt install -y python3-venv

RUN groupadd stdl-docker -g 9999
RUN mkdir /opt/venv && chown -R :9999 /opt/venv && chmod g+rwx /opt/venv

USER 9999:9999

WORKDIR /app

COPY requirements.in .
RUN sed -i "s/<GDAL_VERSION>/`gdalinfo --version | awk '{print $2}' | sed 's/,//'`/g"  requirements.in 

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3 -m pip install --upgrade pip && python3 -m pip install --no-cache pip-tools && pip-compile --cache-dir=/tmp/pip-cache requirements.in && python3 -m pip install --no-cache -r requirements.txt && rm -rf /tmp/pip-cache

COPY scripts/ scripts/
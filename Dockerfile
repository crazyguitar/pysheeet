FROM python:2-alpine
MAINTAINER Blake VandeMerwe <blakev@null.net>

USER root

# use git to clone from source
RUN apk add --update git

RUN mkdir /source \
    && cd /source \
    && git clone https://github.com/crazyguitar/pysheeet.git . \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && cd docs \
    && sphinx-build -b html -d _build/doctrees . _build/html \
    && rm -rf /var/cache/apk/*

WORKDIR /source

# control how many workers the HTTP server should use
ENV WORKERS=2
ENV WORKER_TIMEOUT=60

# map to port 5000
EXPOSE 5000

CMD ["/bin/sh", "-c", "/usr/local/bin/gunicorn \
     -b 0.0.0.0:5000 -w ${WORKERS} -t ${WORKER_TIMEOUT} app:app"]
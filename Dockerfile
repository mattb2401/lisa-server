FROM tensorflow/tensorflow

RUN apt-get update && apt-get install -y git

WORKDIR /go/src/lisa-server

COPY . .

ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH
RUN apt-get install curl
RUN apt-get install -y gnutls-bin

RUN git config --global http.sslVerify false
RUN git config --global http.postBuffer 1048576000

RUN curl -fsSL https://dl.google.com/go/go1.11.1.linux-amd64.tar.gz -o golang.tar.gz && \
    echo "2871270d8ff0c8c69f161aaae42f9f28739855ff5c5204752a8d92a1c9f63993 golang.tar.gz" | sha256sum -c - && \
    tar -C /usr/local -xzf golang.tar.gz && \
    rm golang.tar.gz && \
    mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"
WORKDIR "/go"


ENV TENSORFLOW_LIB_GZIP libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz
ENV TARGET_DIRECTORY /usr/local
RUN  curl -fsSL "https://storage.googleapis.com/tensorflow/libtensorflow/$TENSORFLOW_LIB_GZIP" -o $TENSORFLOW_LIB_GZIP && \
     tar -C $TARGET_DIRECTORY -xzf $TENSORFLOW_LIB_GZIP && \
     rm -Rf $TENSORFLOW_LIB_GZIP
ENV LD_LIBRARY_PATH $TARGET_DIRECTORY/lib
ENV LIBRARY_PATH $TARGET_DIRECTORY/lib
RUN go get -d github.com/tensorflow/tensorflow/tensorflow/go


RUN cd $GOPATH/src/github.com/tensorflow/tensorflow/tensorflow/go && git checkout r1.12

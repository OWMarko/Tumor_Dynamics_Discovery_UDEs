FROM julia:1.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    unzip \
    xorg-dev \
    libgl1-mesa-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

ENV GKSwstype=100

COPY Project.toml Manifest.toml* /app/

RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.precompile()'

COPY . /app

CMD ["bash"]

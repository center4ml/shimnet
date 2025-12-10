FROM python@sha256:e2c7fb05741c735679b26eda7dd34575151079f8c615875fbefe401972b14d85 AS build

RUN useradd -m -u 1000 user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# copy installation files
COPY --chown=user shimnet shimnet/
COPY --chown=user pyproject.toml ./
# install shimnet (cpu version + GUI)
RUN pip install --no-cache-dir .[cpu,gui] --extra-index-url https://download.pytorch.org/whl/cpu 

FROM build AS final

# copy dirs used at inference time
COPY --chown=user configs configs
COPY --chown=user shimnet shimnet
# copy files used at inference time
COPY --chown=user predict-gui.py .
COPY --chown=user download_files.py .

# download weights
RUN python download_files.py --overwrite

CMD [ "python", "./predict-gui.py", "--server_name", "0.0.0.0" ]


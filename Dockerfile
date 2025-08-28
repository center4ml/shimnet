FROM python:3.10 AS build

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

COPY --chown=user . .

# download weights
RUN python download_files.py --overwrite

CMD [ "python", "./predict-gui.py", "--server_name", "0.0.0.0" ]


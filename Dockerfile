FROM python:3.10 as build

RUN useradd -m -u 1000 user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app


COPY --chown=user requirements-cpu.txt requirements-gui.txt ./
RUN pip install --no-cache-dir -r requirements-cpu.txt -r requirements-gui.txt --extra-index-url https://download.pytorch.org/whl/cpu 

FROM build as final

COPY --chown=user . .

# download weights
RUN python download_files.py --overwrite

CMD [ "python", "./predict-gui.py"]


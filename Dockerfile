FROM python:3.10

WORKDIR /usr/src/app

COPY requirements-cpu.txt requirements-gui.txt ./
RUN pip install --no-cache-dir -r requirements-cpu.txt -r requirements-gui.txt --extra-index-url https://download.pytorch.org/whl/cpu 

COPY . .

RUN python download_files.py --overwrite

CMD [ "python", "./predict-gui.py" ]


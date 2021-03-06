FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /

COPY requirements.txt .

RUN python -m pip install --upgrade pip

RUN pip3 install -r requirements.txt

RUN pip3 install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY ./app /app

COPY ./models/model.torch /models/model.torch
COPY ./models/label_encoder.sav /models/label_encoder.sav
COPY ./models/pipe.sav /models/pipe.sav

COPY ./src /src

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "app.main:app"]
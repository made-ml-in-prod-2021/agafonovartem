FROM python:3.6-slim
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY src /src
COPY app.py /app.py
COPY models /models
COPY transformers /transformers

WORKDIR .

ENV PATH_TO_MODEL="models/model.pkl"
ENV PATH_TO_TRANSFORMER="transformers/transformer.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
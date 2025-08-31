
FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py app.py
COPY tourism_project/model_building/logistic_regression_model.joblib ./model_building/logistic_regression_model.joblib
COPY tourism_project/data_registration/X_train.csv ./data_registration/X_train.csv

EXPOSE 7860

CMD ["streamlit", "run", "app.py"]

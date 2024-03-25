
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'nlp09',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 28),  # DAG 시작 날짜
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id='model_training',
    default_args = default_args,
    description = "model data training",
    schedule_interval=timedelta(days=1),  # 매일 실행 
    tags = ['daily', 'model_training'],
)

def run_classifier_training():
  pass

def run_chatbot_training():
  pass

def run_summary_training():
  pass

classifier_training = PythonOperator(
    task_id='classifier_training',
    python_callable=run_classifier_training,
    dag=dag,
)

chatbot_training = PythonOperator(
    task_id='chatbot_training',
    python_callable=run_chatbot_training,
    dag=dag,
)

summary_training = PythonOperator(
    task_id='summary_training',
    python_callable=run_summary_training,
    dag=dag,
)
classifier_training >> chatbot_training >> summary_training
# scripts/pipeline_launcher.py
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
import boto3
import os

sagemaker_session = sagemaker.Session()
role = os.environ.get('SAGEMAKER_ROLE_ARN')  # Set this as env var in CodeBuild
bucket = os.environ.get('S3_BUCKET')         # Set this as env var too

estimator = SKLearn(
    entry_point='code/train.py',
    role=role,
    instance_type='m8g.medium',
    framework_version='0.23-1',
    py_version='py3',
    sagemaker_session=sagemaker_session,
    output_path=f's3://{bucket}/model-output'
)

estimator.fit({'train': f's3://{bucket}/data/train'})


import sagemaker
from sagemaker.sklearn.estimator import SKLearn
import boto3
import os

# Create SageMaker session
sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name
role = os.environ.get('SAGEMAKER_ROLE_ARN')
bucket = os.environ.get('S3_BUCKET')
model_package_group_name = os.environ.get('MODEL_PACKAGE_GROUP', 'default-model-group')

# Define the estimator
estimator = SKLearn(
    entry_point='code/train.py',
    role=role,
    instance_type='ml.t3.large',
    framework_version='0.23-1',
    py_version='py3',
    sagemaker_session=sagemaker_session,
    output_path=f's3://{bucket}/model-output'
)

# Fit the model
estimator.fit({'train': f's3://{bucket}/data/train'})

# Create model package group if not already exists
sm_client = boto3.client("sagemaker", region_name=region)
try:
    sm_client.create_model_package_group(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageGroupDescription="Model group for Random Forest models"
    )
    print(f"✅ Created model package group: {model_package_group_name}")
except sm_client.exceptions.ResourceInUse:
    print(f"ℹ️ Model package group already exists: {model_package_group_name}")

# Register model
model_package = estimator.register(
    content_types=["text/csv"],
    response_types=["text/csv"],
    approval_status="PendingManualApproval",
    model_package_group_name=model_package_group_name
)

print(f"📦 Model registered to group: {model_package_group_name}")
print(f"🔗 Model package ARN: {model_package.model_package_arn}")

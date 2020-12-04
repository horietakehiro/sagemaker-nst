#!/bin/bash 

STACK_NAME=sagemaker-nst-stack
BUCKET_NAME=sagemaker-nst

# create stack
sam deploy \
    --stack-name ${STACK_NAME} \
    --template-file cloudformation.yml \
    --capabilities=CAPABILITY_IAM \
    --parameter-overrides BucketName=${BUCKET_NAME}

# upload prerequired artifacts
aws s3 sync ./images/ s3://${BUCKET_NAME}/train/images/

echo "Done!!"

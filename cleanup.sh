#!/bin/bash
STACK_NAME=sagemaker-nst-stack
BUCKET_NAME=sagemaker-nst

# buckups all training data if neccesary
answer=""
while [[ ${answer} != [yn] ]]
do
    read -n1 -p "Do you need to download all training data from S3 bucket? [y/n]:" answer; echo 
done
if [ ${answer} = "y" ]
then
    DIRNAME=`date "+%Y%m%d-%H%M%S"`
    mkdir -p ./backups/${DIRNAME}/
    aws s3 sync s3://${BUCKET_NAME}/ ./backups/${DIRNAME}/ 
fi

# delete bucket
echo "delete all objects in the bucket : ${BUCKET_NAME}"
aws s3 rb s3://${BUCKET_NAME} --force

# delete cloudformation stack
echo "delete stack : ${STACK_NAME}"
aws cloudformation delete-stack --stack-name ${STACK_NAME}
aws cloudformation wait stack-delete-complete --stack-name ${STACK_NAME}
echo "Done!!"

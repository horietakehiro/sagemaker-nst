# Amazon SageMaker Trial with Neural Style Transfer

## Requirements
- macOS or Linux
- [Docker](https://docs.docker.com/get-docker/) and [docker-compose](https://docs.docker.com/compose/install/)
- [AWS CLI version2](https://docs.aws.amazon.com/ja_jp/cli/latest/userguide/install-cliv2.html)
- [AWS SAM CLI](https://docs.aws.amazon.com/ja_jp/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html)
- [jq](https://stedolan.github.io/jq/download/)

## Environment SetUp
```bash
# clone this repository
git clone https://github.com/horietakehiro/sagemaker-nst.git
cd sagemaker-nst
```

- Notebook Container
```
# build docker image
docker-compose -f docker-compose-cpu.yml build
# or
docker-compose -f docker-compose-gpu.yml build


# launch the notebook container
docker-compose -f docker-compose-cpu.yml up -d 
# or
docker-compose -f docker-compose-gpu.yml up -d 

# now you can access the notebook at "http://localhost:8888/"
# default access password is "secret"
```

- AWS Resources
```bash
# write your aws configuration at ~/.aws/
aws configure

# create aws resources
./deploy.sh
# this script create s3 bucket and sagemaker service role,
# and upload images in ./images/ to s3 bucket
```

- Secret Information
Create `secrets.json` at the same direcotry as `secrets_template.json`. `secrets.json` should contain some secret information.
    - RoleArn : ARN of SageMaker's service role
    - S3Bucket : S3 bucket name which will be used by SageMaker.


## Environment TearDown
- Notebook Container
```bash
# terminate notebook container
docker-compose -f docker-compose-cpu.yml down
# or
docker-compose -f docker-compose-gpu.yml down
```
- AWS Resources
```bash
# delete aws resources
./cleanup.sh
# you can choese whether you download all objects 
# stored at s3 bucket to ./backups/YYYYMMDD-hhmmss/
# by answering the prompt with [y/n]
```

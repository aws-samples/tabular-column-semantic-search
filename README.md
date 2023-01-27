# Tabular Column Semantic Search

### This app creates the following:
1. An automated pipeline for embedding column data from CSVs and indexing the embeddings to OpenSearch.
2. A web app enabling users to search for the k-nearest neighbors to a provided input.

Services used: Step Functions, Glue, SageMaker Processing, Lambda, OpenSearch, S3, ECR, Fargate, ALB
Embeddings are created using [SentenceTransformers](https://www.sbert.net/).
By default the following models are used: 
- `all-MiniLM-L6-v2`, 
- `all-distilroberta-v1`, 
- `average_word_embeddings_glove.6B.300d`

![Architecture](architecture.png)

### Deployment prerequisites
1. [AWS CDK](https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html)
2. [Docker](https://docs.docker.com/get-docker/) running in the background

### How do I use this pipeline and web app?
1. Customize username, password, and any other desired configs in 'config.yaml'.
2. Deploy the pipeline by following the steps below. Deploying from a cloud based instance (e.g. EC2, Cloud9) is recommended.
3. Once deployed, upload CSV files with column headings to the 'data/csv/input/file' or 'data/csv/input/batch' paths of the S3 bucket created during deployment. Files uploaded to 'data/csv/input/file' will be individually processed automatically upon upload. Files uploaded to 'data/csv/input/batch' will be processed in batch when the pipeline is manually triggered. During pipeline execution, input data will be automatically embedded and indexed to OpenSearch. After successful indexing, input data is moved to 'data/csv/processed/'. You can track the pipeline status in the Step Function State Machine console.
    * Note the following helper scripts provided in the 'utils' folder.
        * invoke_sfn_batch.py - Use this script to trigger the pipeline to process batch input data uploaded at 'data/csv/input/batch'. From the cli, run 'python utils/invoke_sfn_batch.py'.
        * get_sample_data.py - Use this script to upload sample data, file or batch, and automatically trigger the pipeline to process uploaded data. From the cli, run 'python utils/get_sample_data.py --destination_bucket <DESTINATION_BUCKET> --input_type <file | batch>'.)
4. Open the web app URL (ALBFargateServiceServiceURL) output by 'cdk deploy' or found in the CloudFormation stack Outputs.
5. Log in using credentials specified in 'config.yaml'.
6. Use the web app to query OpenSearch and visualize results.

### Steps to deploy
Create a virtual environment:

```
$ python3 -m venv .venv
```

Activate your virtualenv:

```
$ source .venv/bin/activate
```

Install the required dependencies:

```
$ pip install -r requirements.txt
```

At this point you can synthesize the CloudFormation template for this code:

```
$ cdk synth
```

Bootstrap your default AWS account/region. Note you may incur AWS charges for data stored in the bootstrapped resources.

```
$ cdk bootstrap
```

Deploy the pipeline to your default AWS account/region. Note Docker needs to be running in the background. Resources will incur charges in your account while the pipeline is deployed.

```
$ cdk deploy
```

To tear down the pipeline, run the following aptly named command.

```
$ cdk destroy
```

## Useful commands

 * `cdk ls`          list all stacks in the app
 * `cdk synth`       emits the synthesized CloudFormation template
 * `cdk deploy`      deploy this stack to your default AWS account/region
 * `cdk diff`        compare deployed stack with current state
 * `cdk docs`        open CDK documentation
 * `cdk destroy`     destroy existing stack

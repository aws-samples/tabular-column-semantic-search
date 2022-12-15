import aws_cdk as cdk
from aws_cdk import (
    Duration,
    Size,
    Stack,
    aws_iam as iam,
    aws_s3 as s3,
    aws_s3_deployment as s3_deploy,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as sfn_tasks,
    aws_opensearchservice as opensearch,
    aws_ec2 as ec2,
    aws_lambda as lambda_,
    aws_lambda_event_sources as event_sources,
    aws_ecr_assets as ecr_assets,
    aws_glue as glue,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_route53 as route53,
    aws_elasticloadbalancingv2 as elb,
    aws_logs as logs
)
from constructs import Construct
import yaml
import streamlit_authenticator as stauth

#####################################
# Get pipeline configs
account_id = cdk.Aws.ACCOUNT_ID
region = cdk.Aws.REGION

with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)
        
resources_name_prefix = config['resources_name_prefix']
parquet_s3_path = config['parquet_s3_path']
processed_csv_s3_path = config['processed_csv_s3_path']
embeddings_s3_path = config['embeddings_s3_path']
glue_max_concurrent_runs = config['glue_max_concurrent_runs']
sm_processing_instance_count = config['sm_processing_instance_count']
sm_processing_instance_type = config['sm_processing_instance_type']
opensearch_instance_type = config['opensearch_instance_type']
opensearch_domain_name = config['opensearch_domain_name']
opensearch_volume_size = config['opensearch_volume_size']
models = config['models']
max_batches = config['max_batches']
local_cpu_architecture = config['local_cpu_architecture']

s3_bucket_name = f'{resources_name_prefix}-{account_id}-{region}'
model_list = models.replace(' ','').split(",")

#####################################
# Hash and store user creds for web app authentication
auth_dict = {
    config['username']: {
        'email': 'email',
        'name': 'name',
        'password': stauth.Hasher([config['password']]).generate()[0]
    }
}

filename = './streamlit_app/auth.yaml'
with open(filename) as file:
        auth = yaml.load(file, Loader=yaml.SafeLoader)
auth['credentials']['usernames'] = auth_dict
with open(filename, 'w') as file:
    file.write(yaml.dump(auth))


#####################################
# Define CDK stack  
class CdkSemanticSearchPipelineStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        #####################################
        # Create IAM roles

        # Role for StepFunctions
        sfn_role = iam.Role(self, "Step Functions IAM role",
            assumed_by=iam.ServicePrincipal("states.amazonaws.com")
        )
        # Add managed policies to sfn_role
        managedPolicies=[
                iam.ManagedPolicy.from_aws_managed_policy_name('CloudWatchEventsFullAccess'),
                iam.ManagedPolicy.from_aws_managed_policy_name('CloudWatchLogsFullAccess'), 
                iam.ManagedPolicy.from_aws_managed_policy_name('AWSLambda_FullAccess'),
                iam.ManagedPolicy.from_aws_managed_policy_name('AmazonS3FullAccess'),
                iam.ManagedPolicy.from_aws_managed_policy_name('AmazonSageMakerFullAccess')
            ]
        for policy in managedPolicies: sfn_role.add_managed_policy(policy) 

        # Role for SageMaker
        sm_role = iam.Role(self, "SageMaker IAM role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com")
        )
        sm_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('AmazonSageMakerFullAccess')) 

        # Role for Glue
        glue_role = iam.Role(self, "Glue IAM role",
            assumed_by=iam.ServicePrincipal("glue.amazonaws.com")
        )
        glue_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('CloudWatchLogsFullAccess'))       

        #####################################
        # Create and configure S3 bucket

        s3_bucket = s3.Bucket(self, "S3 Bucket",
            bucket_name = s3_bucket_name,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=True,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )

        # Upload local 's3' folder to bucket
        s3_deploy.BucketDeployment(self, "S3 deploy assets",
            sources=[s3_deploy.Source.asset('./assets/s3')],
            destination_bucket=s3_bucket,
        )

        # Add S3 permissions for SM and Glue
        s3_bucket.grant_read_write(sm_role)
        s3_bucket.grant_read_write(glue_role)
        
        # Include bucket name in Cfn output
        cdk.CfnOutput(self, 'S3 Bucket Name',
            value=s3_bucket.bucket_name
        )
        
        # Include bucket console url in Cfn output
        cdk.CfnOutput(self, 'S3 Bucket Console URL',
            value=f'https://{region}.console.aws.amazon.com/s3/buckets/{s3_bucket.bucket_name}'
        )

        #####################################
        # Define resources and steps for Step Functions State Machine

        # Define Glue job
        glue_transform_job = glue.CfnJob(self, 'Glue Transform Job',
            name=f'{resources_name_prefix}-parquet-transform',
            command=glue.CfnJob.JobCommandProperty(
                name='glueetl',
                python_version="3",
                script_location=f's3://{s3_bucket.bucket_name}/scripts/csv_to_parquet.py'
            ),
            role=glue_role.role_name,
            glue_version="3.0",
            timeout=7200, # sec
            worker_type="G.2X",
            number_of_workers=10,
            execution_property=glue.CfnJob.ExecutionPropertyProperty(
                max_concurrent_runs=glue_max_concurrent_runs
            ),
            default_arguments={
                '--enable-auto-scaling': 'true',
                '--enable-job-insights': 'true'
            }
        )
        # Set cdk removal policy
        glue_transform_job.apply_removal_policy(cdk.RemovalPolicy.DESTROY)

        # Create Sfn task for Glue job
        task0_glue_parquet_transform = sfn_tasks.GlueStartJobRun(self, 'Glue Transform CSV to Parquet',
            glue_job_name=glue_transform_job.name,
            arguments=sfn.TaskInput.from_object({
                '--input_s3URI': sfn.JsonPath.string_at('$.sfn_input.input_s3URI'),
                '--glue_output_s3URI': sfn.JsonPath.string_at('$.sfn_input.glue_output_s3URI')
                }
            ),
            result_selector= {
                "Arguments.$": "$.Arguments",
                "JobName.$": "$.JobName",
                "Id.$": "$.Id",
                "JobRunState.$": "$.JobRunState"
            },
            # Set key for result
            result_path= '$.glue_transform_result',
            # Wait for job to complete before progressing to next task 
            integration_pattern=sfn.IntegrationPattern.RUN_JOB
        )

        # Build and push SM processing container
        processing_image_asset = ecr_assets.DockerImageAsset(self, 'ECR Image - SM Processing create embeddings',
            directory='assets/docker/create_embeddings'
        )

        # Create parallel Sfn task for SM processing create embeddings jobs
        task1_parallel_create_embeddings = sfn.Parallel(self, "SageMaker Processing Create Embeddings",
            # Set key for result
            result_path= '$.sm_processing_results'
        )

        # For each embedding model, create parallel SM Processing jobs embed column data
        input_data_localpath = '/opt/ml/processing/input/data'
        for model_name in model_list:

            # Define SageMaker Processing job to embed column data
            sm_processing_json = {
                'Type': 'Task',
                'Resource': 'arn:aws:states:::sagemaker:createProcessingJob.sync',
                'Parameters': {
                    'ProcessingJobName.$': f"States.Format('{model_name.replace('_','-').replace('.','-')}-{{}}', $.sfn_input.sm_processing_jobname)",
                    'ProcessingInputs':[
                        {
                            'InputName': 'data',
                            'S3Input': {
                                'S3Uri.$': '$.sfn_input.glue_output_s3URI',
                                'LocalPath': input_data_localpath,
                                'S3DataType': 'S3Prefix',
                                'S3InputMode': 'File',
                                'S3DataDistributionType': 'FullyReplicated',
                                'S3CompressionType': 'None'
                            }
                        },
                        {
                            'InputName': 'code',
                            'S3Input': {
                                'S3Uri': f's3://{s3_bucket.bucket_name}/scripts/create_embeddings.py',
                                'LocalPath': '/opt/ml/processing/input/code',
                                'S3DataType': 'S3Prefix',
                                'S3InputMode': 'File',
                                'S3DataDistributionType': 'FullyReplicated',
                                'S3CompressionType': 'None'
                            }
                        }
                    ],
                    'ProcessingOutputConfig':{
                        'Outputs': [
                            {
                                'OutputName': 'output-1',
                                'S3Output': {
                                    'S3Uri.$': '$.sfn_input.sm_processing_output_s3URI',
                                    'LocalPath': '/opt/ml/processing/output',
                                    'S3UploadMode': 'EndOfJob'
                                }
                            },
                        ]
                    },
                    'ProcessingResources':{
                        'ClusterConfig': {
                            'InstanceCount': sm_processing_instance_count,
                            'InstanceType': sm_processing_instance_type,
                            'VolumeSizeInGB': 30
                        }
                    },
                    'StoppingCondition':{
                        'MaxRuntimeInSeconds': 3600
                    },
                    'AppSpecification':{
                        'ImageUri': processing_image_asset.image_uri,
                        'ContainerEntrypoint': [
                            "python3",
                            "/opt/ml/processing/input/code/create_embeddings.py"
                        ],
                        'ContainerArguments': [
                            '--input_file_or_path', input_data_localpath,
                            '--model_name', model_name,
                            '--stop_after_n', str(max_batches),
                            '--output_dir', '/opt/ml/processing/output'
                        ]
                    },
                    'RoleArn': sm_role.role_arn
                },
                # Filter desired result from task output
                'ResultSelector': {
                    "ProcessingJobName.$": "$.ProcessingJobName",
                    "ProcessingJobStatus.$": "$.ProcessingJobStatus",
                    "ProcessingInputsS3Uri.$": "$.ProcessingInputs[0].S3Input.S3Uri",
                    "ProcessingOutputS3Uri.$": "$.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri"
                },
                # Set key for result
                'ResultPath': '$.result',
                # Select portion of the state output to pass to the next state
                'OutputPath': '$.result',
                "Retry": [{
                    "ErrorEquals": ["SageMaker.AmazonSageMakerException"],
                    "IntervalSeconds": 5,
                    "MaxAttempts": 4,
                    "BackoffRate": 2.0
                }],
            }

            # Create Sfn task for SM processing create embeddings job
            sm_processing_create_embeddings = sfn.CustomState(self, model_name,
                state_json=sm_processing_json
            )

                # Add parallel Sfn task branch
            task1_parallel_create_embeddings.branch(sm_processing_create_embeddings)

        # Create Lambda to index OpenSearch
        lambda_index_opensearch = lambda_.DockerImageFunction(self, "Lambda to index OpenSearch",
            function_name= resources_name_prefix + "-index-opensearch",
            code=lambda_.DockerImageCode.from_image_asset('./assets/lambda/index_opensearch/'),
            environment={
                'domain_name': opensearch_domain_name,
                'processed_csv_s3_path': processed_csv_s3_path
            },
            memory_size=2084,
            timeout=Duration.minutes(5)
        )
        
        # Set cdk removal policy for function
        lambda_index_opensearch.apply_removal_policy(cdk.RemovalPolicy.DESTROY)
        # Add permissions for function
        lambda_index_opensearch.role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('AmazonOpenSearchServiceReadOnlyAccess')) 
        s3_bucket.grant_read(lambda_index_opensearch)
        
        # Create Step Functions task for lambda_index_opensearch
        task2_lambda_index_opensearch = sfn_tasks.LambdaInvoke(self, "Lambda Index OpenSearch",
            lambda_function=lambda_index_opensearch,
            result_selector= {
                "Payload.$": "$.Payload",
                "StatusCode.$": "$.StatusCode",
                "RequestId.$": "$.SdkResponseMetadata.RequestId"
            },
            # Set key for result
            result_path= '$.lambda_index_opensearch_result'
        )

        # Create Lambda move S3 objects to processed locations
        lambda_cleanup = lambda_.Function(self, "Lambda for Cleanup",
            runtime=lambda_.Runtime.PYTHON_3_9,
            function_name= resources_name_prefix + "-cleanup",
            code=lambda_.Code.from_asset('./assets/lambda/cleanup'),
            handler="lambda_cleanup.handler",
            timeout=Duration.minutes(5)
        )
        # Set cdk removal policy for function
        lambda_cleanup.apply_removal_policy(cdk.RemovalPolicy.DESTROY)
        # Add permissions for function
        s3_bucket.grant_read_write(lambda_cleanup)

        # Create Step Functions task for lambda_cleanup
        task3_lambda_cleanup = sfn_tasks.LambdaInvoke(self, "Lambda Cleanup",
            lambda_function=lambda_cleanup,
            result_selector= {
                "Payload.$": "$.Payload",
                "StatusCode.$": "$.StatusCode",
                "RequestId.$": "$.SdkResponseMetadata.RequestId"
            },
            # Set key for result
            result_path= '$.lambda_cleanup_result'
        )

        # Define order of State Machine steps
        definition = task0_glue_parquet_transform \
            .next(task1_parallel_create_embeddings) \
            .next(task2_lambda_index_opensearch) \
            .next(task3_lambda_cleanup)

        # Create State Machine
        sfn_state_machine = sfn.StateMachine(self, "Step Functions State Machine",
            state_machine_name = f'{resources_name_prefix}-state-machine',
            definition=definition,
            role=sfn_role,
            timeout=Duration.hours(4),
            logs=sfn.LogOptions(
                destination=logs.LogGroup(self, f'{resources_name_prefix}-state-machine-logs'),
                level=sfn.LogLevel.ALL
            )
        )

        # Include State Machine console url in Cfn output
        cdk.CfnOutput(self, 'State Machine Console URL',
                value=f'https://{region}.console.aws.amazon.com/states/home?region={region}#/statemachines/view/{sfn_state_machine.state_machine_arn}'
        )

        #####################################
        # Create Lambda to invoke Sfn Sate Machine
        lambda_invoke_sfn = lambda_.Function(self, "Lambda invoke Step Functions",
            runtime=lambda_.Runtime.PYTHON_3_9,
            function_name= resources_name_prefix + "-invoke-step-functions",
            code=lambda_.Code.from_asset('./assets/lambda/invoke_stepfunctions'),
            handler="lambda_invoke_stepfunctions.handler",
            environment={
                'account_id': account_id,
                'state_machine_arn': sfn_state_machine.state_machine_arn,
                'resources_name_prefix': resources_name_prefix,
                'parquet_s3_path': parquet_s3_path,
                'processed_csv_s3_path': processed_csv_s3_path,
                'embeddings_s3_path': embeddings_s3_path,
                'bucket': s3_bucket.bucket_name
            },
            timeout=Duration.seconds(30)
        )
        # Set cdk removal policy for function
        lambda_invoke_sfn.apply_removal_policy(cdk.RemovalPolicy.DESTROY)
        # Add permissions for lambda_invoke_sfn
        sfn_state_machine.grant_start_execution(lambda_invoke_sfn)

        # Trigger lambda_invoke_sfn when '.csv' objects are uploaded to S3 'data/csv/input/file' path
        lambda_invoke_sfn.add_event_source(
            event_sources.S3EventSource(
                bucket=s3_bucket,
                events=[s3.EventType.OBJECT_CREATED],
                filters=[s3.NotificationKeyFilter(
                    prefix='data/csv/input/file',
                    suffix='.csv'
                    )
                ]
            )
        )

        ####################################
        # Create Lambda to query OpenSearch
        lambda_query_opensearch = lambda_.DockerImageFunction(self, "Lambda query OpenSearch",
            function_name= resources_name_prefix + "-query-opensearch",
            code=lambda_.DockerImageCode.from_image_asset('./assets/lambda/query_opensearch/'),
            environment={
                'domain_name': opensearch_domain_name
            },
            memory_size=2084,
            timeout=Duration.minutes(1)
        )
        
        # Set cdk removal policy for function
        lambda_query_opensearch.apply_removal_policy(cdk.RemovalPolicy.DESTROY)
        # Add permissions for function
        lambda_query_opensearch.role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name('AmazonOpenSearchServiceReadOnlyAccess')) 

        ####################################
        # Create OpenSearch Domain
        opensearch_domain = opensearch.Domain(self, "OpenSearch Domain",
            domain_name = opensearch_domain_name,
            version=opensearch.EngineVersion.OPENSEARCH_1_3,
            capacity=opensearch.CapacityConfig(
                data_node_instance_type=opensearch_instance_type,
                data_nodes=1
            ),
            ebs=opensearch.EbsOptions(
                volume_size=opensearch_volume_size,
                volume_type=ec2.EbsDeviceVolumeType.GP3
            ),
            enforce_https=True,
            node_to_node_encryption=True,
            encryption_at_rest=opensearch.EncryptionAtRestOptions(
                enabled=True
            ),
            access_policies=[
                # Grant lambda functions read/write permissions to all indicies
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    principals=[
                        lambda_index_opensearch.role,
                        lambda_query_opensearch.role
                        ],
                    actions=[
                        "es:ESHttpPut",
                        "es:ESHttpPost",
                        "es:ESHttpGet",
                        "es:ESHttpHead"
                    ],
                    resources= [f'arn:aws:es:{region}:{account_id}:domain/{opensearch_domain_name}/*']
                )
            ],
            removal_policy=cdk.RemovalPolicy.DESTROY
        )
        
        # Include domain console URL in Cfn output
        cdk.CfnOutput(self, 'OpenSearch Domain Console URL',
            value=f'https://{region}.console.aws.amazon.com/esv3/home?region={region}#opensearch/domains/{opensearch_domain_name}'
        )
        
        ####################################
        # Create Lambda to embed web app payload
        lambda_embed_payload = lambda_.DockerImageFunction(self, "Lambda embed payload",
            function_name= resources_name_prefix + "-embed-payload",
            code=lambda_.DockerImageCode.from_image_asset('./assets/lambda/embed_payload/'),
            memory_size=2084,
            ephemeral_storage_size=Size.mebibytes(1024),
            timeout=Duration.minutes(1)
        )
        
        # Set cdk removal policy for function
        lambda_embed_payload.apply_removal_policy(cdk.RemovalPolicy.DESTROY)
        
        ###################################
        # Deploy Streamlit app on Fargate fronted by ALB
        
        # Create VPC
        vpc = ec2.Vpc(self, "VPC", 
            max_azs = 2
        )

        # Create ECS cluster
        ecs_cluster = ecs.Cluster(self, "ECS Cluster", 
            cluster_name=f'{resources_name_prefix}-cluster',
            vpc=vpc
        )

        if local_cpu_architecture == 'ARM64':
            ecs_cpu_architecture=ecs.CpuArchitecture.ARM64
        else:
            ecs_cpu_architecture=ecs.CpuArchitecture.X86_64

        # Build and push app container to ECR
        app_image = ecs.ContainerImage.from_asset('streamlit_app')

        # Create Application Load Balanced Fargate Service
        load_balanced_fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(self, "ALB Fargate Service",
            service_name = f'{resources_name_prefix}-alb-fargate-service',
            cluster=ecs_cluster,            
            cpu=512,                    
            desired_count=1,           
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=app_image, 
                container_port=8501,
                environment={
                    'lambda_embed_arn': lambda_embed_payload.function_arn,
                    'lambda_query_arn': lambda_query_opensearch.function_arn,
                    'opensearch_domain': opensearch_domain.domain_name
                }
            ),
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs_cpu_architecture
            ),
            memory_limit_mib=2048,      
            public_load_balancer=True,
            load_balancer_name=f'{resources_name_prefix}-alb',
            # # Uncomment and update below if you have a domain to use
            # target_protocol=elb.ApplicationProtocol.HTTPS,
            # protocol=elb.ApplicationProtocol.HTTPS,
            # redirect_http=True,
            # domain_name='app.example.com',
            # domain_zone=route53.HostedZone(self, 'Rt53 Hosted Zone',
                # zone_name='example.com'
        ) 
        
        # Add permissions for fargate service task role to invoke lambda functions
        lambda_embed_payload.grant_invoke(load_balanced_fargate_service.task_definition.task_role)
        lambda_query_opensearch.grant_invoke(load_balanced_fargate_service.task_definition.task_role)

        # Setup task auto-scaling
        scaling = load_balanced_fargate_service.service.auto_scale_task_count(
            max_capacity=10
        )
        
        scaling.scale_on_cpu_utilization("ECS scaling policy",
            target_utilization_percent=50,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60),
        )

from aws_cdk import (
    Duration,
    RemovalPolicy,  
    Size, 
    Stack, 
    aws_certificatemanager as acm,
    aws_cognito as cognito,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_elasticloadbalancingv2 as elb,
    aws_elasticloadbalancingv2_actions as elb_actions,
    aws_lambda as lambda_,
    aws_opensearchservice as opensearch,
)
from constructs import Construct
import yaml
from tools import utils

#####################################
# Get configs
with open("config.yaml") as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

email = config["email"]
app_username = config["app_username"]
resources_name_prefix = config["resources_name_prefix"]
local_cpu_architecture = config["local_cpu_architecture"]
certificate_arn = config["certificate_arn"]

if not certificate_arn:
    # Generate cert and upload to ACM
    key_file = "tools/private.key"
    cert_file = "tools/selfsigned.crt"
    utils.cert_gen(
        key_file=key_file, 
        cert_file=cert_file
    )
    certificate_arn = utils.cert_acm_upload(
        key_file=key_file,
        cert_file=cert_file
    )
    # Add cert arn to config file
    config["certificate_arn"] = certificate_arn
    with open("config.yaml", "w") as file:
        file.write(yaml.dump(config, sort_keys=False))

#####################################
# Define CDK stack
class CdkSemanticSearchFrontEndStack(Stack):
    def __init__(self, scope: Construct, construct_id: str,
                 lambda_query_opensearch: lambda_.DockerImageFunction, 
                 opensearch_domain: opensearch.Domain, 
                 **kwargs
                 ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ####################################
        # Create Lambda to embed payload
        lambda_embed_payload = lambda_.DockerImageFunction(
            self,
            "Lambda embed payload",
            function_name=resources_name_prefix + "-embed-payload",
            code=lambda_.DockerImageCode.from_image_asset("./assets/lambda/embed_payload/"),
            memory_size=2084,
            ephemeral_storage_size=Size.mebibytes(1024),
            timeout=Duration.minutes(1),
        )

        # Set provisioned concurrency
        lambda_embed_payload.add_alias("Live", provisioned_concurrent_executions=1)

        # Set cdk removal policy for function
        lambda_embed_payload.apply_removal_policy(RemovalPolicy.DESTROY)

        ###################################
        # Deploy Streamlit app on Fargate fronted by ALB

        # Create VPC
        vpc = ec2.Vpc(self, "VPC", max_azs=2)

        # Create ECS cluster
        ecs_cluster = ecs.Cluster(self, "ECS Cluster", cluster_name=f"{resources_name_prefix}-cluster", vpc=vpc)

        if local_cpu_architecture == "ARM64":
            ecs_cpu_architecture = ecs.CpuArchitecture.ARM64
        else:
            ecs_cpu_architecture = ecs.CpuArchitecture.X86_64

        # Build and push app container to ECR
        app_image = ecs.ContainerImage.from_asset("streamlit_app")

        # Create Application Load Balanced Fargate Service
        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "ALB Fargate Service",
            service_name=f"{resources_name_prefix}-alb-fargate-service",
            cluster=ecs_cluster,
            certificate=acm.Certificate.from_certificate_arn(self, "Certificate",
                certificate_arn=certificate_arn
            ),
            redirect_http=True,
            cpu=512,
            desired_count=1,
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=app_image,
                container_port=8501,
                environment={
                    "lambda_embed_arn": lambda_embed_payload.function_arn,
                    "lambda_query_arn": lambda_query_opensearch.function_arn,
                    "opensearch_domain": opensearch_domain.domain_name,
                },
            ),
            runtime_platform=ecs.RuntimePlatform(cpu_architecture=ecs_cpu_architecture),
            memory_limit_mib=2048,
            public_load_balancer=True,
            load_balancer_name=resources_name_prefix,
        )

        # Configure the health checks to use /healthcheck endpoint
        fargate_service.target_group.configure_health_check(
            enabled=True,
            path="/healthcheck",
            healthy_http_codes="200"
        )

        # Add permissions for fargate service task role to invoke lambda functions
        lambda_embed_payload.grant_invoke(fargate_service.task_definition.task_role)
        lambda_query_opensearch.grant_invoke(fargate_service.task_definition.task_role)

        # Set task auto-scaling
        scaling = fargate_service.service.auto_scale_task_count(max_capacity=10)

        scaling.scale_on_cpu_utilization(
            "ECS scaling policy",
            target_utilization_percent=50,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60),
        )

        ####################################
        # Create Cognito resources

        # User pool
        user_pool = cognito.UserPool(self, "User Pool",
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            password_policy=cognito.PasswordPolicy(
                min_length=12,
                require_lowercase=True,
                require_uppercase=True,
                require_digits=True,
                require_symbols=True,
            ),
            user_invitation=cognito.UserInvitationConfig(
                email_subject="Column Semantic Search Login",
                email_body= f"<p>Your credentials:</p> \
                    <p> \
                    Username: {{username}}<br />\
                    Password: {{####}} \
                    </p> \
                    <p>\
                    Please wait until the deployent has completed before accessing the website. \
                    </p>\
                    <p>\
                    Website: https://{fargate_service.load_balancer.load_balancer_dns_name} \
                    </p>\
                    ",
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )

        # User Pool Domain
        user_pool_domain = cognito.UserPoolDomain(self, "User Pool Domain",
            user_pool=user_pool,
            cognito_domain=cognito.CognitoDomainOptions(
                domain_prefix=resources_name_prefix
            )
        )

        # User Pool Client
        user_pool_client = cognito.UserPoolClient(self, "User Pool Client",
            user_pool=user_pool,
            user_pool_client_name="AlbAuthentication",
            generate_secret=True,
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(authorization_code_grant=True),
                scopes=[cognito.OAuthScope.OPENID],
                callback_urls=[f"https://{fargate_service.load_balancer.load_balancer_dns_name}/oauth2/idpresponse"]
            ),
            supported_identity_providers=[cognito.UserPoolClientIdentityProvider.COGNITO]
        )

        # User
        user = cognito.CfnUserPoolUser(self, "User Pool User",
            user_pool_id=user_pool.user_pool_id,
            desired_delivery_mediums=["EMAIL"],
            force_alias_creation=False,
            user_attributes=[
                cognito.CfnUserPoolUser.AttributeTypeProperty(
                    name="email",
                    value=email
                )
            ],   
            username=app_username
        )

        ####################################
        # Add Cognito auth to ALB

        # Add HTTPS egress rule to Load Balancer security group to talk to Cognito
        lb_security_group = fargate_service.load_balancer.connections.security_groups[0]

        lb_security_group.add_egress_rule(
            peer=ec2.Peer.any_ipv4(),
            connection=ec2.Port(
                protocol=ec2.Protocol.TCP,
                string_representation="443",
                from_port=443,
                to_port=443
            )
        )

        # Allow 10 seconds for in flight requests before termination
        fargate_service.target_group.set_attribute(
            key="deregistration_delay.timeout_seconds",
            value="10"
        )

        # Add the authentication actions as a rule
        fargate_service.listener.add_action(
            "authenticate-rule",
            action=elb_actions.AuthenticateCognitoAction(
                next=elb.ListenerAction.forward(
                    target_groups=[
                        fargate_service.target_group
                    ]
                ),
                user_pool=user_pool,
                user_pool_client=user_pool_client,
                user_pool_domain=user_pool_domain,

            )
        )

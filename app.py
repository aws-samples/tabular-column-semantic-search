import os
import aws_cdk as cdk
import yaml

from cdk_stacks.cdk_semantic_search_pipeline_stack import CdkSemanticSearchPipelineStack
from cdk_stacks.cdk_semantic_search_frontend_stack import CdkSemanticSearchFrontEndStack

with open("config.yaml") as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

stack_name = config["stack_name"]

env = cdk.Environment(
        account=os.environ["CDK_DEFAULT_ACCOUNT"],
        region=os.environ["CDK_DEFAULT_REGION"]
    )

app = cdk.App()

pipeline = CdkSemanticSearchPipelineStack(app, f"{stack_name}-Pipeline", env=env)

frontend = CdkSemanticSearchFrontEndStack(app, f"{stack_name}-FrontEnd", 
    pipeline.lambda_query_opensearch, 
    pipeline.opensearch_domain,
    env=env
)

for stack in [pipeline, frontend]:
    cdk.Tags.of(stack).add("Creator", "CDK")
    cdk.Tags.of(stack).add("Description", "Column semantic search pipeline")

app.synth()

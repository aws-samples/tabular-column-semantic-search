import aws_cdk as cdk

from cdk_stacks.cdk_semantic_search_pipeline_stack import CdkSemanticSearchPipelineStack

app = cdk.App()
CdkSemanticSearchPipelineStack(app, "CdkSemanticSearchPipelineStack")

app.synth()

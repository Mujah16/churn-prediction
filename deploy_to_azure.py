import mlflow.azureml
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
import uuid

# Example config — replace these with your actual Azure values
subscription_id = "11111111-2222-3333-4444-555555555555"
resource_group = "my-resource-group"
workspace_name = "my-ml-workspace"
model_name = "ChurnLSTM"
model_version = "1"

# Azure ML client
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

# Unique endpoint name to avoid collisions
endpoint_name = f"churn-endpoint-{uuid.uuid4().hex[:6]}"

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="Endpoint for churn prediction model",
    auth_mode="key",
)

print(f"🌀 Creating endpoint: {endpoint.name}...")
ml_client.begin_create_or_update(endpoint).result()

# Create deployment from MLflow model
deployment = ManagedOnlineDeployment(
    name="blue",  # deployment name within the endpoint
    endpoint_name=endpoint.name,
    model=f"azureml:{model_name}:{model_version}",
    instance_type="Standard_DS2_v2",
    instance_count=1,
)

print(f"🚀 Deploying model {model_name}:{model_version} to endpoint...")
ml_client.begin_create_or_update(deployment).result()

# Set as default
ml_client.online_endpoints.begin_update(
    ManagedOnlineEndpoint(name=endpoint.name, traffic={"blue": 100})
).result()

print(f"✅ Deployment complete! Endpoint: {endpoint.name}")

from __future__ import annotations

import argparse
import os
import uuid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy a registered churn model to Azure ML.")
    parser.add_argument("--subscription-id", default=os.getenv("AZURE_SUBSCRIPTION_ID"))
    parser.add_argument("--resource-group", default=os.getenv("AZURE_RESOURCE_GROUP"))
    parser.add_argument("--workspace-name", default=os.getenv("AZURE_ML_WORKSPACE"))
    parser.add_argument("--model-name", default=os.getenv("AZURE_ML_MODEL_NAME", "ChurnLSTM"))
    parser.add_argument("--model-version", default=os.getenv("AZURE_ML_MODEL_VERSION", "1"))
    parser.add_argument("--endpoint-name", default=os.getenv("AZURE_ML_ENDPOINT_NAME"))
    parser.add_argument(
        "--instance-type",
        default=os.getenv("AZURE_ML_INSTANCE_TYPE", "Standard_DS2_v2"),
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=int(os.getenv("AZURE_ML_INSTANCE_COUNT", "1")),
    )
    return parser.parse_args()


def require(value: str | None, name: str) -> str:
    if not value:
        raise SystemExit(f"Missing {name}. Provide it as a CLI argument or environment variable.")
    return value


def main() -> None:
    args = parse_args()

    try:
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:
        raise SystemExit(
            "Azure dependencies are missing. Install them with "
            "`pip install -r requirements-azure.txt`.",
        ) from exc

    subscription_id = require(args.subscription_id, "subscription id")
    resource_group = require(args.resource_group, "resource group")
    workspace_name = require(args.workspace_name, "workspace name")
    endpoint_name = args.endpoint_name or f"churn-endpoint-{uuid.uuid4().hex[:6]}"

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Endpoint for churn prediction model",
        auth_mode="key",
    )

    print(f"Creating endpoint: {endpoint.name}")
    ml_client.begin_create_or_update(endpoint).result()

    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint.name,
        model=f"azureml:{args.model_name}:{args.model_version}",
        instance_type=args.instance_type,
        instance_count=args.instance_count,
    )

    print(f"Deploying model {args.model_name}:{args.model_version} to endpoint {endpoint.name}")
    ml_client.begin_create_or_update(deployment).result()

    ml_client.online_endpoints.begin_update(
        ManagedOnlineEndpoint(name=endpoint.name, traffic={"blue": 100}),
    ).result()

    print(f"Deployment complete. Endpoint: {endpoint.name}")


if __name__ == "__main__":
    main()

"""
Deploy the forecasting model to Azure ML as a managed online endpoint.
Run: python azure/deploy_azure.py
Requires: azure/config.json with subscription_id, resource_group, workspace_name
"""
import os
import json
import pickle
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

CONFIG_PATH = "azure/config.json"


def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"Create {CONFIG_PATH} with subscription_id, resource_group, workspace_name"
        )
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_client(cfg: dict) -> MLClient:
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=cfg["subscription_id"],
        resource_group_name=cfg["resource_group"],
        workspace_name=cfg["workspace_name"],
    )


def register_model(client: MLClient, freq: str) -> Model:
    artifact_path = f"model/artifacts/ensemble_{freq}.pkl"
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Train model first: {artifact_path}")
    model = Model(
        path=artifact_path,
        name=f"sales-forecast-{freq.lower()}",
        description=f"Ensemble sales forecast model ({freq} frequency)",
        type="custom_model",
    )
    return client.models.create_or_update(model)


def deploy(freq: str = "W"):
    cfg = load_config()
    client = get_client(cfg)

    # Register model
    registered_model = register_model(client, freq)
    print(f"Registered model: {registered_model.name} v{registered_model.version}")

    endpoint_name = f"sales-forecast-{freq.lower()}-ep"

    # Create endpoint
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Sales Forecasting Ensemble Endpoint",
        auth_mode="key",
        tags={"freq": freq, "model": "ensemble"},
    )
    client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint created: {endpoint_name}")

    # Environment
    env = Environment(
        name="sales-forecast-env",
        conda_file="azure/conda_env.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )

    # Deployment
    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint_name,
        model=registered_model,
        environment=env,
        code_configuration=CodeConfiguration(
            code="azure/scoring",
            scoring_script="score.py",
        ),
        instance_type="Standard_DS2_v2",
        instance_count=1,
    )
    client.online_deployments.begin_create_or_update(deployment).result()

    # Route 100% traffic to blue
    endpoint.traffic = {"blue": 100}
    client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Deployment complete. Endpoint: {endpoint_name}")

    # Print scoring URI
    ep = client.online_endpoints.get(endpoint_name)
    print(f"Scoring URI: {ep.scoring_uri}")


if __name__ == "__main__":
    import sys
    freq = sys.argv[1] if len(sys.argv) > 1 else "W"
    deploy(freq)

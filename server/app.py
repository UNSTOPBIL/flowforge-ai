from openenv.core import create_fastapi_app
import uvicorn

try:
    from flowforge.env import FlowForgeEnvironment
    from flowforge.models import FlowForgeAction, FlowForgeObservation
except ImportError:
    from env import FlowForgeEnvironment
    from models import FlowForgeAction, FlowForgeObservation

def create_app():
    """Factory function for OpenEnv server deployment.
    
    Returns standard ASGI application wrapping the FlowForge environment,
    with strict typing defined by the Pydantic models.
    """
    return create_fastapi_app(
        FlowForgeEnvironment,          # Class, not instance (for isolated sessions)
        action_cls=FlowForgeAction,
        observation_cls=FlowForgeObservation,
    )

app = create_app()

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

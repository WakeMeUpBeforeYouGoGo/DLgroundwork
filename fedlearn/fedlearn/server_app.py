"""FedLearn: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fedlearn.task import Net, get_weights

# Initialize model parameters
print("Initializing model parameters...")
ndarrays = get_weights(Net())
parameters = ndarrays_to_parameters(ndarrays)

def server_fn(context: Context):
    print("Server function started")

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    print(f"Number of server rounds: {num_rounds}")

    # Define strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)
    print(f"Server configured with strategy: {strategy}")

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
print("Starting ServerApp...")
app = ServerApp(server_fn=server_fn)
print("ServerApp running...")

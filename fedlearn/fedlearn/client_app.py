"""FedLearn: A Flower / PyTorch app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from fedlearn.task import StudentNet, Net, DEVICE, load_data, get_weights, set_weights, train, test

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, student_net, trainloader, valloader, local_epochs, learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75):
        self.net = net
        self.student_net = student_net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.T = T
        self.soft_target_loss_weight = soft_target_loss_weight
        self.ce_loss_weight = ce_loss_weight

    def fit(self, parameters, config):
        print("Client fit method called")
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.student_net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            DEVICE,
            self.T,
            self.soft_target_loss_weight,
            self.ce_loss_weight,
            self.learning_rate,
        )
        print("Client training completed")
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        print("Client evaluate method called")
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        loss_student, accuracy_student = test(self.student_net, self.valloader)
        print(f"Client evaluation completed with accuracy: {accuracy}")
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context):
    print("Client function started")
    # Load model and data
    net = Net().to(DEVICE)
    student_net = StudentNet().to(DEVICE)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, student_net, trainloader, valloader, local_epochs).to_client()

# Flower ClientApp
print("Starting ClientApp...")
app = ClientApp(client_fn)
print("ClientApp running...")

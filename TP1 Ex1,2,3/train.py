"""Train

This script allows to train one federated learning experiment; the dataset name, the algorithm and the
number of clients should be precised alongside with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * train - training class ready for federated learning simulation

"""

from utils.args import *
from utils.utils import *
# from plot_utils import plot_results  # Removed import statement

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd


def init_clients(args_):

    clients = []

    data_dir = get_data_dir(args_.experiment)

    for client_dir in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, client_dir)) and client_dir.startswith('client_'):

            client_id = int(client_dir.replace('client_', ''))
            client_dir = os.path.join(data_dir, f"client_{client_id}")
            logs_dir = os.path.join(args_.logs_dir, f"client_{client_id}")
            os.makedirs(logs_dir, exist_ok=True)
            logger = SummaryWriter(logs_dir)

            client = \
                init_client(
                    args=args_,
                    client_id=client_id,
                    client_dir=client_dir,
                    logger=logger
                )

            clients.append(client)

    return clients


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()

    seed = (args.seed if (("seed" in args) and (args.seed >= 0)) else int(time.time()))
    torch.manual_seed(seed)

    print("==> Initialize Clients..")

    clients = \
        init_clients(args_=args)

    clients_weights = get_clients_weights(
        clients=clients,
        objective_type=args.objective_type,
    )

    print("==> Initialize Aggregator..")

    global_learner = \
        get_learner(
            experiment_name=args.experiment,
            device=args.device,
            optimizer_name=args.server_optimizer,
            lr=args.server_lr,
            seed=seed
        )

    global_logs_dir = os.path.join(args.logs_dir, "global")
    os.makedirs(global_logs_dir, exist_ok=True)
    global_logger = SummaryWriter(global_logs_dir)

    aggregator = \
        get_aggregator(
            aggregator_type=args.aggregator_type,
            clients=clients,
            clients_weights=clients_weights,
            global_learner=global_learner,
            logger=global_logger,
            verbose=args.verbose,
            seed=seed
        )

    print("==> Training Loop..")

    aggregator.write_logs()

    local_epochs_list = [1, 5, 10, 50, 100]  # Các cấu hình số local epochs cần thử nghiệm

    # Tạo dictionary để lưu kết quả độ chính xác của tập test tương ứng cho mỗi cấu hình local epochs.
    test_accuracies_dict = {local_steps: [] for local_steps in local_epochs_list}

    for local_steps in local_epochs_list:
        print(f"==> Training with {local_steps} local epochs..")
        
        # Cập nhật số local epochs cho từng client
        for client in clients:
            client.local_steps = local_steps

        # Reset lại aggregator (ví dụ: reset số vòng giao tiếp) để bắt đầu lại từ đầu cho cấu hình này.
        aggregator.c_round = 0
        aggregator.write_logs()  # Ghi log trạng thái khởi tạo

        for c_round in tqdm(range(args.n_rounds)):
            print(f"Training Round: {c_round + 1}/{args.n_rounds}, Local Epochs: {local_steps}")
            aggregator.mix()
            
            if (c_round % args.log_freq) == (args.log_freq - 1):
                aggregator.write_logs()
            
            # Sau mỗi round, tiến hành đánh giá mô hình trên tập test của client 0
            test_loss, test_metric = clients[0].learner.evaluate_loader(clients[0].test_loader)
            test_accuracies_dict[local_steps].append((c_round, test_metric))  # Lưu lại số round và test accuracy

            # Ghi kết quả vào file để dễ theo dõi
            with open('result.txt', 'a') as f:
                f.write(f'Local Steps: {local_steps}, Round: {c_round}, Test Accuracy: {test_metric}\n')

    # After training loop
    # plot_results(test_accuracies_dict)  # Removed plotting function call

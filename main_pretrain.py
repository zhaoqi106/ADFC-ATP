import time
import argparse
import torch_geometric
import torch.optim as optim
from model.encoder_gat import GATNet
from utils.gat_pre import GATCon
from utils.gat_pretrain import *
from utils.nt_xent import NT_Xent
from torch_geometric.loader import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
"""
模型预训练
"""
# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    feature_graph = torch.Tensor()
    feature_org = torch.Tensor()
    edge_weight = torch.Tensor()
    feature_weight = torch.Tensor()
    for tem in train_bar:
        tem = tem.to(device)

        graph1, out_1, org2, out_2, ew, xw = net(tem)

        feature_graph = torch.cat((feature_graph, graph1.cpu().data), 0)

        feature_org = torch.cat((feature_org, org2.cpu().data), 0)
        edge_weight = torch.cat((edge_weight, ew.cpu().data))
        feature_weight = torch.cat((feature_weight, xw.cpu().data))

        criterion = NT_Xent(out_1.shape[0], temperature, 1)
        loss = criterion(out_1, out_2)
        total_num += len(tem)
        total_loss += loss.item() * len(tem)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.8f}'.format(epoch, warm_epochs, total_loss / total_num))

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

    return total_loss / total_num, feature_graph, feature_org, edge_weight.numpy(), feature_weight.numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--datafile', default='test', help='orginal data for input')
    parser.add_argument('--path', default='pretrain', help='orginal data for input')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=10, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=50, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--augment_ratio', default=0.3, type=float, help='Ratio for data augmentation')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')

    # args parse
    args = parser.parse_args()
    print(args)
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, warm_epochs = args.batch_size, args.epochs

    train_data = TestbedDataset(
        augment_ratio=args.augment_ratio,
        root=args.path,
        dataset=args.datafile,
        patt=f'_aug_{args.augment_ratio}',
        seed=args.seed
    )

    print('use GAT encoder')
    model_encoder1 = GATNet(output_dim=128)
    model_encoder2 = GATNet(output_dim=128)
    model = GATCon(encoder1=model_encoder1, encoder2=model_encoder2)
    model = model.to('cuda:0')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch_geometric.nn.DataParallel(model, device_ids=[0, 1, 2])
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-7)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}'.format(batch_size, warm_epochs, args.datafile, args.augment_ratio)
    if not os.path.exists('results/' + save_name_pre):
        os.mkdir('results/' + save_name_pre)

    for epoch in range(1, warm_epochs + 1):
        start = time.time()

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        train_loss, features, org, ew, xw = train(model, train_loader, optimizer)

        if epoch in list(range(0, warm_epochs + 1, 5)):
            torch.save(model_encoder1.state_dict(),
                       'results/model/' + str(epoch) + '_model_encoder_gat_' + save_name_pre + '.pkl')

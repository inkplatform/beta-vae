import os
import torch
import torch.optim as optim
import multiprocessing
import time
import preprocess as prep
import utils
from config import params
from torchvision.utils import save_image
from models import celeba_model

def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, mu, logvar = model(data)
        loss = model.loss(output, data, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()), epoch, batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    print('Train set Average loss:', train_loss)
    return train_loss


def test(model, device, test_loader, return_images=0, log_interval=None):
    model.eval()
    test_loss = 0

    # two np arrays of images
    original_images = []
    rect_images = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            output, mu, logvar = model(data)
            loss = model.loss(output, data, mu, logvar)
            test_loss += loss.item()

            if return_images > 0 and len(original_images) < return_images:
                original_images.append(data[0].cpu())
                rect_images.append(output[0].cpu())

            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), loss.item()))

    test_loss /= len(test_loader)
    print('Test set Average loss:', test_loss)

    if return_images > 0:
        return test_loss, original_images, rect_images

    return test_loss


if(params['dataset'] == 'CelebA'):
    from models import celeba_model as net_model
elif(params['dataset'] == 'CASIA'):
    from models import casia_model as net_model
elif(params['dataset'] == 'FERET'):
    from models import feret_model as net_model

use_cuda = params['use_cuda'] and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
print('num cpus:', multiprocessing.cpu_count())

IMAGE_PATH = '/home/mophy/DataSet/bate_vae/data/celeba/img_align_celeba/'
data_partition = 'data/list_eval_partition.csv'
data_attr = '/home/mophy/DataSet/bate_vae/data/list_attr_celeba.csv'
if(params['dataset'] == 'CelebA'):
    IMAGE_PATH = '/home/mophy/DataSet/bate_vae/data/celeba/img_align_celeba/'
    data_partition = '/home/mophy/DataSet/bate_vae/data/list_eval_partition_celeba.csv'
    data_attr = '/home/mophy/DataSet/bate_vae/data/list_attr_celeba.csv'
elif(params['dataset'] == 'CASIA'):
    IMAGE_PATH = '/home/mophy/DataSet/bate_vae/data/casia/'
    data_partition = '/home/mophy/DataSet/bate_vae/data/list_eval_partition_casia.csv'
    data_attr = '/home/mophy/DataSet/bate_vae/data/list_attr_casia.csv'
elif(params['dataset'] == 'FERET'):
    IMAGE_PATH = '/home/mophy/DataSet/bate_vae/data/feret/'
    data_partition = '/home/mophy/DataSet/bate_vae/data/list_eval_partition_feret.csv'
    data_attr = '/home/mophy/DataSet/bate_vae/data/list_attr_feret.csv'

# training code
train_ids, test_ids = prep.split_dataset(data_partition)
print('num train_images:', len(train_ids))
print('num test_images:', len(test_ids))

data_train = prep.ImageDiskLoader(train_ids, IMAGE_PATH)
data_test = prep.ImageDiskLoader(test_ids, IMAGE_PATH)

kwargs = {'num_workers': multiprocessing.cpu_count(),'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(data_train, batch_size=params['batch_size'], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=params['test_batch_size'], shuffle=True, **kwargs)

print('latent size:', params['latent_size'])

# model = models.BetaVAE(latent_size=LATENT_SIZE).to(device)
model = net_model.DFCVAE(latent_size=params['latent_size']).to(device)

optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

if __name__ == "__main__":

    start_epoch = model.load_last_model(params['model_path'] + params['dataset']) + 1
    train_losses, test_losses = utils.read_log(params['log_path'] + params['dataset'] + '/log.pkl', ([], []))

    com_path = params['compare_path'] + params['dataset'] + '/'
    if not os.path.exists(os.path.join(com_path)):
        os.makedirs(com_path)

    for epoch in range(start_epoch, params['num_epochs'] + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, params['print_interval'])
        test_loss, original_images, rect_images = test(model, device, test_loader, return_images=5)

        save_image(original_images + rect_images,  com_path + str(epoch) + '.png', padding=0, nrow=len(original_images))

        train_losses.append((epoch, train_loss))
        test_losses.append((epoch, test_loss))
        utils.write_log(params['log_path'] + params['dataset'] + '/log.pkl', (train_losses, test_losses))

        model.save_model(params['model_path'] + params['dataset'] + '/' + '%03d.pt' % epoch)

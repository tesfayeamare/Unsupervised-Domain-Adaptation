import requests

_channel = 'homestead'

def notify(message, title=None):
    if title is None:
        print(message);
        requests.post(f'https://ntfy.sh/{_channel}', data=message)
    else:
        print(f'{title}: {message}');
        requests.post(f'https://ntfy.sh/{_channel}', data=message, headers={
            "Title": title,
        })

def to_cuda(*tensors):
    tensors = list(tensors)
    if len(tensors) == 1:
        return tensors[0].cuda()
    for i, tensor in enumerate(tensors):
        tensors[i] = tensor.cuda()
    return tensors

def get_accuracy(targets, outputs):
    return (outputs.max(dim=1)[1].eq(targets).sum() / outputs.size(dim=0)).item()

def get_accuracy2(outputs, targets):
    return (outputs.max(dim=1)[1].eq(targets).sum() / outputs.size(dim=0)).item()

def show_images(dataLoader, n=3):
    images, lables = next(iter(dataLoader))
    grid = torchvision.utils.make_grid(
      list(map(lambda img: torch.clip(img, min=0, max=1), images[0:n])), nrow=n)
    plt.figure(figsize=(5,5))
    plt.imshow(torch.permute(grid, (1, 2, 0)), vmin=0, vmax=1)
    print('lables:', lables[0:n])
    
def mover(dataLoader, device='cuda:0'):
    for batch in dataLoader:
        for i, t in enumerate(batch):
            batch[i] = t.to(device)
        yield batch
    
'''
from torchsummary import summary
summary(model, (3, 232, 232))
'''
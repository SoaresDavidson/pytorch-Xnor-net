import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
import torchvision
import torchvision.transforms as transforms


from torchvision.utils import make_grid 
import matplotlib.pyplot as plt

from torchinfo import summary
from AlexNet import AlexNet, AlexNetXNOR

parser = argparse.ArgumentParser(
                    prog='pytorch Lenet5 training ',
                    description='implementation of Lenet5 algorithm and its binarization on mnist dataset'
                    )
parser.add_argument('--model', default="lenet5", type=str, help="train lenet5 or alexnet model")
parser.add_argument('--binarized', default=False, type=bool, help="apply xnor or standard model")
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--batch_size', default=64, type=int, metavar='B',
                    help='batch size')


args = parser.parse_args()

batch_size =args.batch_size
num_classes = 10
learning_rate = args.lr
num_epochs = args.epochs

train_dataset = torchvision.datasets.CIFAR10(root = './data',
                                               train = True,
                                               transform = transforms.Compose([
                                                #       transforms.Resize((224, 224)),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                                                      ]),
                                               download = True)
    
    
test_dataset = torchvision.datasets.CIFAR10(root = './data',
                                            train = False,
                                            transform = transforms.Compose([
                                                    # transforms.Resize((224, 224)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                                                    ]),
                                            download=True)
    
    
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            pin_memory=True,
                                            num_workers=16
                                            )
    
    
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            pin_memory=True,
                                            num_workers=16
                                            )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"dispositivo: {device}")


    
model:nn.Module = AlexNet(num_classes) if not args.binarized else AlexNetXNOR(num_classes)

i, _ = next(iter(train_loader))
summary(model, input_size=i.shape) 
# model.compile()
torch.set_float32_matmul_precision('high') 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//4, gamma=0.1)
cost = nn.CrossEntropyLoss()


total_samples = len(train_loader.dataset) #type: ignore
print(total_samples)
print(len(train_loader))

def train(epoch):
    model.train()
    running_loss = 0
    for i, (images, labels) in enumerate(train_loader):  
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = cost(outputs, labels)
        
        loss.backward()

        running_loss += loss.item()

        optimizer.step()

        if (i) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Sample [{i * batch_size}/{total_samples}], Loss: {loss.item():.4f}')
    scheduler.step()


def eval():
    model.eval() 
    times = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images,labels = images.to(device), labels.to(device)

            start_time = time.time()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            end_time = time.time()
            times.append(end_time - start_time)

            total += labels.size(0)
            correct += (predicted == labels).sum().item() 

        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')
        avg_inference_time = sum(times) / len(times) if times else 0
        print(f"Average inference time per batch: {avg_inference_time:.6f} seconds")


for i in range(num_epochs):
    train(i)
    eval()

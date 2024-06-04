import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CUB_dataset, CUB_dataset_Test
from tqdm import tqdm


'''Performs a simple finetuned resnet50 classification to serve as a baseline. LEADERBOARD RESULT: 63% accuracy'''

def main(
    n_epochs,
    batch_size,
    lr,
    device,
):
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for p in resnet.parameters():
        p.requires_grad = False
    resnet.fc = nn.Sequential(*[nn.Dropout(p=0.3), nn.Linear(2048, 30)])
    resnet.fc.requires_grad = True
    resnet.to(device)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 

    train_dataset = CUB_dataset(
        root_dir='data/train_images',
        class_index_file='data/class_indexes.csv',
        transform=train_transform,
        return_id=True)
    val_dataset = CUB_dataset(
        root_dir='data/val_images',
        class_index_file='data/class_indexes.csv',
        transform=transform,
        return_id=True)
    test_dataset = CUB_dataset_Test(
        root_dir="data/test_images",
        transform=transform,
        return_id=True,
        gallery_length=0)
    test_paths = test_dataset.image_paths

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    batch = next(iter(test_loader))
    # print(batch['image'])
    # print(batch['id'])

    for epoch in range(1, n_epochs+1):
        print("Epoch %d"%epoch)
        resnet.train()
        train_loss = 0.
        tqdmloader = tqdm(train_loader, unit="batch")
        for i, batch in enumerate(tqdmloader):
            optimizer.zero_grad()
            images, labels = batch['image'].to(device), batch['label'].to(device)
            logits = resnet(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss = (train_loss * i * batch_size + loss.detach().cpu().item()) / ((i+1) * batch_size)
            tqdmloader.set_description('train_loss %.4f'%train_loss)
        
        resnet.eval()
        val_loss = 0.
        tqdmloader = tqdm(val_loader, unit='batch')
        with torch.no_grad():
            for i, batch in enumerate(tqdmloader):
                images, labels = batch['image'].to(device), batch['label'].to(device)
                logits = resnet(images)
                loss = loss_fn(logits, labels)
                val_loss = (val_loss * i * batch_size + loss.detach().cpu().item()) / ((i+1) * batch_size)
                tqdmloader.set_description('val_loss %.4f'%val_loss)
    
    predictions = torch.zeros(size=(len(test_dataset),), dtype=torch.int32)
    resnet.eval()
    with torch.no_grad():
        tqdmloader = tqdm(test_loader, unit='batch')
        for i, batch in enumerate(tqdmloader):
            images, idx = batch['image'].to(device), batch['id'].to(torch.int32)
            logit = resnet(images)
            pred = torch.argmax(logit, dim=-1).to(torch.int32).cpu()
            predictions[idx] = pred
    
    submissions = []
    for i in range(len(test_dataset)):
        submissions.append([test_paths[i][17:].split(".jpg")[0], predictions[i].item()])
    import pandas
    pandas.DataFrame(data=submissions, columns=['ID','Category'], index=None).to_csv('submissions_baseline.csv', index=False)

        
if __name__ == '__main__':
    args = {
        'n_epochs':20,
        'lr':5e-4,
        'batch_size':64,
        'device':torch.device('cuda'),
    }
    main(**args)

import torch
import torch.nn.parallel
from torchvision import transforms
from dataset import CUB_dataset, CUB_dataset_Test, CUB_full_dataset
from model import AVSL_Similarity
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from train import train
from inference import validate, infer_queries, get_predictions, infer_gallery
import argparse
import os

def main(
        base_model_name,
        lay_to_emb_ids,
        emb_dim, 
        num_classes, 
        topk, 
        momentum,
        momentum_decay,
        p,
        epochs, 
        lr, 
        batch_size_training,
        batch_size_inference,
        device, 
        accumulation_steps,
        CNN_coeffs, 
        sim_coeffs,
        metrics_K,
        model_path,
        name,
        validate_on_train=False,
        validate_on_val=True,
        infer_gallery_to_queries=True,
        infer_full_dataset=False,
        pretrained=False,
        train_model=False,
    ) -> None:
    '''
    Full pipeline to create/train/validate/save model
    

        Parameters
        ----------
        - base_model_name: The base CNN model on which the AVSL model is built upon
        - lay_to_emb_ids: The ids the layers of the CNN model to project into an embedding space. For ResNet50 can be any
            sublist of [1,2,3,4], for EfficientNet_V2_S can be a sublist of [0,1,2,3,4,5,6,7]
        - num_classes: number of classes in the dataset
        - emb_dim: embedding dimension
        - topk: parameter of the AVSL model to compute the overall similarity
        - momemtum: the momentum at which the links are updated. W <- (1-m)*old_W + m*new_W
        - momentum_decay: exponential decay rate of the momentum
        - p: the exponent of the norm used to compute similarities, p=2 means L2 norm.
        - accumulation_steps: number of steps for gradient accumulation (to increase the batch size without using too much GPU memory)
        - batch_size_training: batch_size used for training, the higher the better ProxyAnchorLoss works
        - batch_size_inference: batch_size used for inference, small is faster
        - CNN_coeffs, sim_coeffs: coefficients used for the ProxyAnchorLoss
        - metrics_K: the K used to compute recall@K
        - model_path: if using a pretrained model, the path of the model
        - name: name of the model, is used to create a directory in runs/{name}

    '''
    # ==================== Datasets ====================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 
    # train_transform = A.Compose([
    #     A.Resize((224,224)),
    #     A.HorizontalFlip(p=0.5),
    #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    #     A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
    #     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ToTensorV2(),
    # ])
    # transform = A.Compose([
    #     A.Resize((224,224)),
    #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ToTensorV2(),
    # ])
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
    query_dataset = CUB_dataset_Test(
        root_dir="data/test_images",
        transform=transform,
        return_id=True,
        gallery_length=0)
    
    n_layers = len(lay_to_emb_ids)
    model_name = f"emb{emb_dim}-batch{batch_size_training}-lr{lr}-layers{n_layers}-topk{topk}-m{momentum}-mdk{momentum_decay}.pt"
    save_dir = os.path.join("runs", name)
    
    if not(os.path.exists("runs")):
        os.mkdir("runs")
    if not(os.path.exists(save_dir)):
        os.mkdir(save_dir)

    if pretrained:
        print("Loading pretrained model")
        model = torch.load(model_path, device)
    else:
        model = AVSL_Similarity(base_model_name, lay_to_emb_ids, num_classes, emb_dim, topk, momentum, p).to(device)
    if train_model:
        train(model, train_dataset, val_dataset, n_layers, epochs, lr, batch_size_training, device, accumulation_steps, CNN_coeffs, sim_coeffs, momentum_decay, save_dir, model_name)
    
    # =================== Measuring performance ======================
    if validate_on_train:
        validate(model, train_dataset, batch_size_inference, device, metrics_K, save_matrix=True, name="train", save_dir=save_dir)
    if validate_on_val:
        validate(model, val_dataset, batch_size_inference, device, metrics_K, save_matrix=True, name="val", save_dir=save_dir)

    # ========================= Infering on test set =========================
    if infer_gallery_to_queries:
        gallery_to_query_similarities = infer_queries(model, train_dataset, batch_size_inference, query_dataset, device)
        torch.save(gallery_to_query_similarities, os.path.join(save_dir,"glry_to_query_sim.pt"))
        for metric_K in metrics_K:
            get_predictions(gallery_to_query_similarities, train_dataset.labels, metric_K, query_dataset.image_paths, save_dir=save_dir)
    # ========================= Infering on the whole dataset =========================
    if infer_full_dataset:
        full_dataset = CUB_full_dataset(transform, return_id=True)
        full_similarities = infer_gallery(model, full_dataset, batch_size_inference, device)
        torch.save(full_similarities, os.path.join(save_dir, "full_similiarities.pt"))
        print("Saved full similarities!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse training parameters")

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size_training", type=int, default=100)
    parser.add_argument("--batch_size_inference", type=int, default=30)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_model_name", type=str, default="ResNet50", help="Either ResNet50 or EfficientNet_V2_S")
    parser.add_argument("--lay_to_emb_ids", type=int, nargs='+', default=[2,3,4], help="the base model's layers to project to an embedding space")
    parser.add_argument("--emb_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--topk", type=int, default=128, help="topk value AVSL")
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--momentum_decay", type=float, default=0.9)
    parser.add_argument("--p", type=int, default=2, help="norm degree for embedding distance")
    parser.add_argument("--CNN_coeffs", type=float, nargs=2, default=(32, 0.1), help="Coefficients for CNN loss")
    parser.add_argument("--sim_coeffs", type=float, nargs=2, default=(32, 0.1))
    parser.add_argument("--validate_on_train", action="store_true")
    parser.add_argument("--validate_on_val", action="store_true")
    parser.add_argument("--infer_gallery_to_queries", action="store_true")
    parser.add_argument("--infer_full_dataset", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--train_model", action="store_true", help="Whetheer to train the model")
    parser.add_argument("--model_path", type=str, default=None, help="if pretrained, takes the pretrained model path")
    parser.add_argument("--name", type=str, default="AVSL_v3", help="General name of the model, should not be too long, i.e. AVSL_resnet50")
    parser.add_argument("--metrics_K", type=int, nargs='+', default=[1, 2, 4, 8], help="Values of K for recall and precision")

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['device'] = torch.device(args_dict['device'])
    
    main(**args_dict)
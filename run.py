import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

# user-defined
from Model.UnetModel import UNetModel as unet
from data.preprocess import MyDataset
from utils.config import DefaultConfig


def adjust_learning_rate(learning_rate, learning_rate_decay, optimizer, epoch):
    learning_rate = learning_rate * (learning_rate_decay ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    return learning_rate


def train_unhealthy(args):
    global epoch_losses
    torch.cuda.empty_cache()
    # checkpoint1 = torch.load('model_train8.pth')
    # step1: read data
    # create Dataset
    my_dataset = MyDataset(scale=(args.crop_height, args.crop_width), mode='train', data_type='unhealthy')
    # use DataLoader to load Dataset
    train_have_loader = DataLoader(my_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # step2: model definition
    model_have = unet(
        in_channels=args.in_channels,
        model_channels=args.model_channels,
        out_channels=args.out_channels,
        channel_mult=args.channel_mult,
        attention_resolutions=args.attention_resolutions
    )
    # model_have.load_state_dict(checkpoint1)
    model_have.to(args.device)
    # Calculate parameters
    total_params = sum(p.numel() for p in model_have.parameters())
    print(f"Total number of parameters: {total_params}")

    # step3: optimizer difinition
    optimizer_have = torch.optim.Adam(model_have.parameters(), lr=args.lr)
    # step4: train the model
    epochs = 100
    epoch_losses_all = []
    for epoch in range(epochs):
        print("epoch:", end=' ')
        print(epoch)
        for step_have_train, images_have_train in tqdm(enumerate(train_have_loader), total=len(train_have_loader)):
            epoch_losses = []
            images_have_train = images_have_train.to(args.device)
            optimizer_have.zero_grad()
            t_have_train = torch.randint(199, args.timesteps, (args.batch_size,), device=args.device).long()
            # print(t_have_train)
            # t_have_train = torch.full((args.batch_size,), args.timesteps, dtype=torch.long, device=args.device)
            # print(t_have_train)
            # 计算模型model_have的训练误差(loss_have_train)，去噪后的伪原图(denoise_images_have_train)，预测的噪声(pnoise_have_train),加噪后的图像(denoise_image)
            (loss_have_train, denoise_images_have_train, pnoise_have_train,
             noise_image) = args.gaussian_diffusion.train_losses(
                model_have, images_have_train, t_have_train)

            epoch_losses.append(loss_have_train.item())
            loss_have_train.backward()
            optimizer_have.step()
            if epoch % 10 == 0 and step_have_train == 0:
                denoised_image = denoise_images_have_train[0].permute(1, 2, 0).cpu().detach().numpy()
                denoised_image = (denoised_image + 1) * 127.5
                plt.imshow(denoised_image.astype(int))
                plt.title(f"Epoch {epoch} - Denoised  - with loss:{loss_have_train}")
                plt.axis("off")
                plt.savefig(f'{args.img_output_dir}/unhealthy_epoch_{epoch}_Denoised.png')
                plt.show()
                plt.close()
                noisy_image = noise_image[0].permute(1, 2, 0).cpu().detach().numpy()
                noisy_image = (noisy_image + 1) * 127.5
                plt.imshow(noisy_image.astype(int))
                plt.title(f"Epoch {epoch} - Noisy")
                plt.axis("off")
                plt.savefig(f'{args.img_output_dir}/unhealthy_epoch_{epoch}_Noisy.png')
                plt.show()
                plt.close()
                torch.save(model_have.state_dict(), args.save_have_model_path + f'_{epoch}.pth')
        # adjust_learning_rate(args.lr, args.learning_rate_decay, optimizer_have, epoch)
        epoch_losses_all.append(np.max(epoch_losses))
    plt.plot(epoch_losses_all)
    plt.title('epoch losses')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f'{args.img_output_dir}/unhealthy_epoch losses.png')
    plt.show()
    plt.close()

    print("model_have saved successfully!")


def train_healthy(args):
    checkpoint2 = torch.load('model_train9.pth')
    dataset_no_train = datasets.ImageFolder('images_isic2016_have', transform=args.transform)
    train_no_loader = torch.utils.data.DataLoader(dataset_no_train, batch_size=args.batch_size, shuffle=False)
    dataset_no_train_target = datasets.ImageFolder('images_isic2016_no', transform=args.transform)
    train_no_target_loader = torch.utils.data.DataLoader(dataset_no_train_target, batch_size=args.batch_size,
                                                         shuffle=False)
    model_no = unet(
        in_channels=3,
        model_channels=96,
        out_channels=3,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    )
    optimizer_no = torch.optim.Adam(model_no.parameters(), lr=7e-4)
    model_no.load_state_dict(checkpoint2)
    model_no.to(args.device)
    epochs = 1
    for epoch in range(epochs):
        print("epoch:", end=' ')
        print(epoch)
        for step_no_train, (
                (images_no_train, labels_no_train), (images_no_train_target, labels_no_train_target)) in enumerate(
            zip(train_no_loader, train_no_target_loader)):
            optimizer_no.zero_grad()
            # print(labels_no_train)
            # batch_size = images_no_train.shape[0]
            images_no_train = images_no_train.to(args.device)
            images_no_train_target = images_no_train_target.to(args.device)
            t_no_train = torch.randint(50, 60, (args.batch_size,), device=args.device).long()
            (loss_no_train, denoise_images_no_train, pnoise_no_train, x_noisy) = args.gaussian_diffusion.train_losses(
                model_no, images_no_train, t_no_train, images_no_train_target)

            print("Loss:", loss_no_train.item())
            print(step_no_train)

            loss_no_train.backward()
            optimizer_no.step()
    torch.save(model_no.state_dict(), 'model_no8.pth')


def test(args):
    # step1: load model
    model_have_use = unet(
        in_channels=3,
        model_channels=96,
        out_channels=3,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    )
    model_have_use.to(args.device)
    model_no_use = unet(
        in_channels=3,
        model_channels=96,
        out_channels=3,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    )
    model_no_use.to(args.device)
    checkpoint1 = torch.load('model_train8.pth', map_location=torch.device('cpu'))
    checkpoint2 = torch.load('model_no8.pth', map_location=torch.device('cpu'))
    model_have_use.load_state_dict(checkpoint1)
    model_no_use.load_state_dict(checkpoint2)
    dataset_final = datasets.ImageFolder('testdata', transform=args.transform)
    final_loader = torch.utils.data.DataLoader(dataset_final, batch_size=4, shuffle=False)
    dataset_mask = datasets.ImageFolder('testdata', transform=args.transform)
    mask_loader = torch.utils.data.DataLoader(dataset_mask, batch_size=4, shuffle=False)

    with torch.no_grad():
        for step, ((images, labels), (masks, labels_mask)) in enumerate(zip(final_loader, mask_loader)):
            images = images.to(args.device)
            masks = masks.to(args.device)
            t = torch.randint(0, 1, (4,), device=args.device).long()
            print(t)
            (loss, denoise_images, pnoise, x_noisy) = args.gaussian_diffusion.train_losses(model_have_use, images, t,
                                                                                           x_target=None)
            (loss, denoise_images2, pnoise2, x_noisy2) = args.gaussian_diffusion.train_losses(model_no_use, images, t,
                                                                                              x_target=None)
            plt.figure(figsize=(6, 6))
            for idx, image in enumerate(x_noisy):
                image = (image.squeeze().permute(1, 2, 0) + 1) * 127.5
                image = image.to("cpu").numpy().astype(int)
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(image, aspect='auto')
                plt.axis("off")
            plt.show()
            plt.figure(figsize=(6, 6))
            for idx, image in enumerate(denoise_images):
                image = (image.squeeze().permute(1, 2, 0) + 1) * 127.5
                image = image.to("cpu").numpy().astype(int)
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(image, aspect='auto')
                plt.axis("off")
            plt.show()
            plt.figure(figsize=(6, 6))
            for idx, image in enumerate(pnoise):
                image = (image.squeeze().permute(1, 2, 0) + 1) * 127.5
                image = image.to("cpu").numpy().astype(int)
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(image, aspect='auto')
                plt.axis("off")
            plt.show()
            for idx, image in enumerate(denoise_images2):
                image = (image.squeeze().permute(1, 2, 0) + 1) * 127.5
                image = image.to("cpu").numpy().astype(int)
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(image, aspect='auto')
                plt.axis("off")
            plt.show()

            plt.figure(figsize=(6, 6))
            for idx, image in enumerate(pnoise2):
                image = (image.squeeze().permute(1, 2, 0) + 1) * 127.5
                image = image.to("cpu").numpy().astype(int)
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(image, aspect='auto')
                plt.axis("off")
            plt.show()

            plt.figure(figsize=(6, 6))
            for idx, mask_batch in enumerate(masks):
                mask_batch = (mask_batch.squeeze().permute(1, 2, 0) + 1)
                mask_batch = mask_batch.to("cpu").numpy()
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(mask_batch, cmap='gray', aspect='auto')
                plt.axis("off")
            plt.show()

            plt.figure(figsize=(6, 6))
            for idx, (d1, d2) in enumerate(zip(pnoise, pnoise2)):
                # image=d2-d1
                d1 = (d1.squeeze().permute(1, 2, 0) + 1) * 127.5
                d1 = d1.to("cpu").numpy().astype(int)
                d1_tensor = torch.from_numpy(d1)
                d2 = (d2.squeeze().permute(1, 2, 0) + 1) * 127.5
                d2 = d2.to("cpu").numpy().astype(int)
                d2_tensor = torch.from_numpy(d2)
                import numpy as np

                d1_tensor_float = d1_tensor.float()
                d2_tensor_float = d2_tensor.float()
                gray_d1 = torch.sum(d1_tensor_float * args.weights, dim=-1, keepdim=True)
                gray_d1 = gray_d1.cpu().numpy()
                gray_d1 = gray_d1 - np.min(gray_d1)
                gray_d2 = torch.sum(d2_tensor_float * args.weights, dim=-1, keepdim=True)
                gray_d2 = gray_d2.cpu().numpy()
                gray_d2 = gray_d2 - np.min(gray_d2)
                gray_image2 = gray_d2 - gray_d1 * 1.5
                gray_image2 = (gray_image2 - np.min(gray_image2)) * (255 / np.max(gray_image2 - np.min(gray_image2)))

                gray_image2_brightened = np.clip(gray_image2 * 1.2, 0, 255)
                threshold = 180
                binary_image = np.where(gray_image2_brightened < threshold, 1, 0)
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(gray_image2_brightened.squeeze(), cmap='gray', aspect='auto')
                plt.axis("off")
                # print(binary_image2)
            plt.show()
            plt.figure(figsize=(6, 6))
            for idx, (d1, d2) in enumerate(zip(pnoise, pnoise2)):
                # image=d2-d1
                d1 = (d1.squeeze().permute(1, 2, 0) + 1) * 127.5
                d1 = d1.to("cpu").numpy().astype(int)
                d1_tensor = torch.from_numpy(d1)
                d2 = (d2.squeeze().permute(1, 2, 0) + 1) * 127.5
                d2 = d2.to("cpu").numpy().astype(int)
                d2_tensor = torch.from_numpy(d2)
                import numpy as np
                d1_tensor_float = d1_tensor.float()
                d2_tensor_float = d2_tensor.float()
                gray_d1 = torch.sum(d1_tensor_float * args.weights, dim=-1, keepdim=True)
                gray_d1 = gray_d1.cpu().numpy()
                gray_d1 = gray_d1 - np.min(gray_d1)
                gray_d2 = torch.sum(d2_tensor_float * args.weights, dim=-1, keepdim=True)
                gray_d2 = gray_d2.cpu().numpy()
                gray_d2 = gray_d2 - np.min(gray_d2)
                gray_image2 = gray_d2 - gray_d1 * 1.5
                gray_image2 = (gray_image2 - np.min(gray_image2)) * (255 / np.max(gray_image2 - np.min(gray_image2)))
                gray_image2_brightened = np.clip(gray_image2 * 1.2, 0, 255)

                threshold = 189
                binary_image = np.where(gray_image2_brightened < threshold, 1, 0)
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(binary_image.squeeze(), cmap='gray', aspect='auto')
                plt.axis("off")
            plt.show()
            intersection = np.logical_and(binary_image, mask_batch) * 255
            has_number_greater_than_255 = (intersection > 255).any()
            dice = (2.0 * np.sum(intersection)) / (np.sum(binary_image) * 255 + np.sum(mask_batch) * 255)
            print(f"Step {step}, Image {idx}, Dice Score: {dice}")

    return None


if __name__ == '__main__':
    args = DefaultConfig()
    # train_unhealthy(args)
    # train_healthy(args)
    # test2(args)

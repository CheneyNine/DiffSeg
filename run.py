import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
# user-defined
from Model.UnetModel import UNetModel as unet
from utils.config import DefaultConfig
from data.preprocess import MyDataset
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
import matplotlib.pyplot as plt
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from skimage.color import gray2rgb
from skimage.transform import resize
import random

def adjust_learning_rate(learning_rate, learning_rate_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR multiplied by learning_rate_decay(set 0.98, usually) every epoch"""
    learning_rate = learning_rate * (learning_rate_decay ** epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    return learning_rate

def train_unhealthy(args):
    global epoch_losses
    torch.cuda.empty_cache()
    checkpoint1 = torch.load('model_train8.pth')

    # step1: 读取数据
    # 创建 Dataset 实例
    my_dataset = MyDataset(scale=(args.crop_height, args.crop_width), mode='train', data_type='unhealthy')
    # 使用 DataLoader 来包装 Dataset
    train_have_loader = DataLoader(my_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # step2: 定义模型
    model_have = unet(
        in_channels=args.in_channels,
        model_channels=args.model_channels,
        out_channels=args.out_channels,
        channel_mult=args.channel_mult,
        attention_resolutions=args.attention_resolutions
    )
    model_have.load_state_dict(checkpoint1)
    model_have.to(args.device)
    # 计算模型的参数量
    total_params = sum(p.numel() for p in model_have.parameters())
    print(f"Total number of parameters: {total_params}")

    # step3: 定义优化器
    optimizer_have = torch.optim.Adam(model_have.parameters(), lr=args.lr)
    # step4: 训练模型
    epochs = 1
    # 准备一个列表来存储每个epoch的损失
    epoch_losses_all = []
    for epoch in range(epochs):
        print("epoch:", end=' ')
        print(epoch)
        for step_have_train, images_have_train in tqdm(enumerate(train_have_loader), total=len(train_have_loader)):
            epoch_losses = []
            images_have_train = images_have_train.to(args.device)
            # 梯度清零
            optimizer_have.zero_grad()
            # 对每个样本进行随机t_have步加噪
            t_have_train = torch.randint(199, args.timesteps, (args.batch_size,), device=args.device).long()
            # print(t_have_train)
            # t_have_train = torch.full((args.batch_size,), args.timesteps, dtype=torch.long, device=args.device)
            # print(t_have_train)
            # 计算模型model_have的训练误差(loss_have_train)，去噪后的伪原图(denoise_images_have_train)，预测的噪声(pnoise_have_train),加噪后的图像(denoise_image)
            (loss_have_train, denoise_images_have_train, pnoise_have_train,noise_image) = args.gaussian_diffusion.train_losses(
                model_have, images_have_train, t_have_train)

            epoch_losses.append(loss_have_train.item())
            # 反向传播
            loss_have_train.backward()
            # 更新参数
            optimizer_have.step()
            if epoch % 10 == 0 and step_have_train == 0:  # 每隔10个epoch的第一个step
                # 假设 denoise_images_have_train 是一个四维张量 (batch_size, height, width, channels)
                # 展示去噪后的图像
                denoised_image = denoise_images_have_train[0].permute(1, 2, 0).cpu().detach().numpy()
                denoised_image = (denoised_image + 1) * 127.5
                plt.imshow(denoised_image.astype(int))
                plt.title(f"Epoch {epoch} - Denoised  - with loss:{loss_have_train}")
                plt.axis("off")
                plt.savefig(f'{args.img_output_dir}/unhealthy_epoch_{epoch}_Denoised.png')
                plt.show()
                plt.close()

                # 展示原始加噪图像
                noisy_image = noise_image[0].permute(1, 2, 0).cpu().detach().numpy()
                noisy_image = (noisy_image + 1) * 127.5
                plt.imshow(noisy_image.astype(int))
                plt.title(f"Epoch {epoch} - Noisy")
                plt.axis("off")
                plt.savefig(f'{args.img_output_dir}/unhealthy_epoch_{epoch}_Noisy.png')
                plt.show()
                plt.close()

                # 保存模型
                torch.save(model_have.state_dict(), args.save_have_model_path + f'_{epoch}.pth')
        # 在每个epoch结束后更新学习率
        # adjust_learning_rate(args.lr, args.learning_rate_decay, optimizer_have, epoch)
        epoch_losses_all.append(np.max(epoch_losses))
    # 画出每轮epoch损失图
    plt.plot(epoch_losses_all)
    plt.title('epoch losses')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f'{args.img_output_dir}/unhealthy_epoch losses.png')
    plt.show()
    plt.close()

    print("model_have saved successfully!")


def train_healthy(args):
    # 训练加噪后生成没有病症图片的模型
    # step1: 读取数据
    checkpoint2 = torch.load('model_train9.pth')
    dataset_no_train = datasets.ImageFolder('images_isic2016_have', transform=args.transform)
    train_no_loader = torch.utils.data.DataLoader(dataset_no_train, batch_size=args.batch_size, shuffle=False)
    dataset_no_train_target = datasets.ImageFolder('images_isic2016_no', transform=args.transform)
    train_no_target_loader = torch.utils.data.DataLoader(dataset_no_train_target, batch_size=args.batch_size, shuffle=False)
    # step2: 定义模型
    model_no = unet(
        in_channels=3,
        model_channels=96,
        out_channels=3,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    )
    # step3: 定义优化器
    optimizer_no = torch.optim.Adam(model_no.parameters(), lr=7e-4)
    # step4: 训练模型
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
    # step5: 保存模型
    torch.save(model_no.state_dict(), 'model_no8.pth')

def test(args):
    # 进行噪声差异化计算
    # step1: 载入模型
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
    # 加载之前保存的模型参数
    checkpoint1 = torch.load('model_train8.pth', map_location=torch.device('cpu'))
    checkpoint2 = torch.load('model_no8.pth', map_location=torch.device('cpu'))
    # 将参数加载到新模型中
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
            (loss, denoise_images, pnoise, x_noisy) = args.gaussian_diffusion.train_losses(model_have_use, images, t,x_target=None)
            (loss, denoise_images2, pnoise2, x_noisy2) = args.gaussian_diffusion.train_losses(model_no_use, images, t,x_target=None)
            # 在显示之前，设置图像的显示大小
            plt.figure(figsize=(6, 6))  # 12x6英寸的图像大小，可以根据需要调整
            # 绘制原图
            for idx, image in enumerate(x_noisy):
                image = (image.squeeze().permute(1, 2, 0) + 1) * 127.5
                image = image.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(image, aspect='auto')
                plt.axis("off")
            plt.show()  # 显示所有的图像
            # 绘制有病症方向的噪声
            plt.figure(figsize=(6, 6))  # 12x6英寸的图像大小，可以根据需要调整
            for idx, image in enumerate(denoise_images):
                image = (image.squeeze().permute(1, 2, 0) + 1) * 127.5
                image = image.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(image, aspect='auto')
                plt.axis("off")
            plt.show()  # 显示所有的图像
            plt.figure(figsize=(6, 6))  # 12x6英寸的图像大小，可以根据需要调整
            for idx, image in enumerate(pnoise):
                image = (image.squeeze().permute(1, 2, 0) + 1) * 127.5
                image = image.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(image, aspect='auto')
                plt.axis("off")
            plt.show()  # 显示所有的图像
            for idx, image in enumerate(denoise_images2):
                image = (image.squeeze().permute(1, 2, 0) + 1) * 127.5
                image = image.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(image, aspect='auto')
                plt.axis("off")
            plt.show()  # 显示所有的图像
            # 绘制无病症方向的噪声
            plt.figure(figsize=(6, 6))  # 12x6英寸的图像大小，可以根据需要调整
            for idx, image in enumerate(pnoise2):
                image = (image.squeeze().permute(1, 2, 0) + 1) * 127.5
                image = image.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(image, aspect='auto')
                plt.axis("off")
            plt.show()  # 显示所有的图像
            # 绘制专家标注
            plt.figure(figsize=(6, 6))  # 12x6英寸的图像大小，可以根据需要调整
            for idx, mask_batch in enumerate(masks):
                mask_batch = (mask_batch.squeeze().permute(1, 2, 0) + 1)
                mask_batch = mask_batch.to("cpu").numpy()
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(mask_batch, cmap='gray', aspect='auto')
                plt.axis("off")
            plt.show()  # 显示所有的图像

            # 噪声差异
            plt.figure(figsize=(6, 6))  # 12x6英寸的图像大小，可以根据需要调整
            for idx, (d1, d2) in enumerate(zip(pnoise, pnoise2)):
                # image=d2-d1
                d1 = (d1.squeeze().permute(1, 2, 0) + 1) * 127.5
                d1 = d1.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
                d1_tensor = torch.from_numpy(d1)
                d2 = (d2.squeeze().permute(1, 2, 0) + 1) * 127.5
                d2 = d2.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
                d2_tensor = torch.from_numpy(d2)
                import numpy as np

                # 将gray_image2转换为8位无符号整型（即灰度图像）
                # gray_image2 = gray_image2.astype(np.uint8)

                # 将d1_tensor的数据类型转换为float
                d1_tensor_float = d1_tensor.float()
                d2_tensor_float = d2_tensor.float()

                # 使用加权求和获取灰度图像
                gray_d1 = torch.sum(d1_tensor_float * args.weights, dim=-1, keepdim=True)
                gray_d1 = gray_d1.cpu().numpy()  # 转换为NumPy数组
                gray_d1 = gray_d1 - np.min(gray_d1)
                gray_d2 = torch.sum(d2_tensor_float * args.weights, dim=-1, keepdim=True)
                gray_d2 = gray_d2.cpu().numpy()  # 转换为NumPy数组
                gray_d2 = gray_d2 - np.min(gray_d2)
                gray_image2 = gray_d2 - gray_d1 * 1.5
                gray_image2 = (gray_image2 - np.min(gray_image2)) * (255 / np.max(gray_image2 - np.min(gray_image2)))


                gray_image2_brightened = np.clip(gray_image2 * 1.2, 0, 255)

                # 二值化灰度图像
                threshold = 180  # 阈值，可以根据需要调整
                binary_image = np.where(gray_image2_brightened < threshold, 1, 0)
                # 对二值图像进行外部白色区域的填充处理
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(gray_image2_brightened.squeeze(), cmap='gray', aspect='auto')  # 使用灰度颜色映射
                plt.axis("off")
                # print(binary_image2)
            plt.show()  # 显示所有的图像
            # 噪声差异
            plt.figure(figsize=(6, 6))  # 12x6英寸的图像大小，可以根据需要调整
            for idx, (d1, d2) in enumerate(zip(pnoise, pnoise2)):
                # image=d2-d1
                d1 = (d1.squeeze().permute(1, 2, 0) + 1) * 127.5
                d1 = d1.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
                d1_tensor = torch.from_numpy(d1)
                d2 = (d2.squeeze().permute(1, 2, 0) + 1) * 127.5
                d2 = d2.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
                d2_tensor = torch.from_numpy(d2)
                import numpy as np
                # 将d1_tensor的数据类型转换为float
                d1_tensor_float = d1_tensor.float()
                d2_tensor_float = d2_tensor.float()
                # 使用加权求和获取灰度图像
                gray_d1 = torch.sum(d1_tensor_float * args.weights, dim=-1, keepdim=True)
                gray_d1 = gray_d1.cpu().numpy()  # 转换为NumPy数组
                gray_d1 = gray_d1 - np.min(gray_d1)
                gray_d2 = torch.sum(d2_tensor_float * args.weights, dim=-1, keepdim=True)
                gray_d2 = gray_d2.cpu().numpy()  # 转换为NumPy数组
                gray_d2 = gray_d2 - np.min(gray_d2)
                gray_image2 = gray_d2 - gray_d1 * 1.5
                gray_image2 = (gray_image2 - np.min(gray_image2)) * (255 / np.max(gray_image2 - np.min(gray_image2)))
                gray_image2_brightened = np.clip(gray_image2 * 1.2, 0, 255)
                # 二值化灰度图像
                # 创建一个空列表来存储转换后的灰度图像
                # 假设 gray_image2_brightened 是一个包含四张图像的列表
                threshold = 189  # 阈值，可以根据需要调整
                binary_image = np.where(gray_image2_brightened < threshold, 1, 0)

                # 对二值图像进行外部白色区域的填充处理
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(binary_image.squeeze(), cmap='gray', aspect='auto')  # 使用灰度颜色映射
                plt.axis("off")
                # print(binary_image2)
            plt.show()  # 显示所有的图像
            # 计算 Dice 系数
            intersection = np.logical_and(binary_image, mask_batch) * 255
            has_number_greater_than_255 = (intersection > 255).any()
            dice = (2.0 * np.sum(intersection)) / (np.sum(binary_image) * 255 + np.sum(mask_batch) * 255)
            print(f"Step {step}, Image {idx}, Dice Score: {dice}")

    return None


def test2(args):
    # 进行噪声差异化计算
    # step1: 载入模型
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
    # 加载之前保存的模型参数
    checkpoint1 = torch.load('model_train8.pth', map_location=torch.device('cpu'))
    checkpoint2 = torch.load('model_no8.pth', map_location=torch.device('cpu'))
    # 将参数加载到新模型中
    model_have_use.load_state_dict(checkpoint1)
    model_no_use.load_state_dict(checkpoint2)
    dataset_final = datasets.ImageFolder('testdata', transform=args.transform)
    final_loader = torch.utils.data.DataLoader(dataset_final, batch_size=1, shuffle=False)
    dataset_mask = datasets.ImageFolder('testdata', transform=args.transform)
    mask_loader = torch.utils.data.DataLoader(dataset_mask, batch_size=1, shuffle=False)

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
            # 在显示之前，设置图像的显示大小
            plt.figure(figsize=(6, 6))  # 12x6英寸的图像大小，可以根据需要调整
            # 绘制原图
            for idx, image in enumerate(x_noisy):
                image = (image.squeeze().permute(1, 2, 0) + 1) * 127.5
                image = image.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(image, aspect='auto')
                plt.axis("off")
            plt.show()  # 显示所有的图像
            # 绘制有病症方向的噪声
            plt.figure(figsize=(6, 6))  # 12x6英寸的图像大小，可以根据需要调整
            for idx, image in enumerate(denoise_images):
                image = (image.squeeze().permute(1, 2, 0) + 1) * 127.5
                image = image.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(image, aspect='auto')
                plt.axis("off")
            plt.show()  # 显示所有的图像
            plt.figure(figsize=(6, 6))  # 12x6英寸的图像大小，可以根据需要调整
            for idx, image in enumerate(pnoise):
                image = (image.squeeze().permute(1, 2, 0) + 1) * 127.5
                image = image.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(image, aspect='auto')
                plt.axis("off")
            plt.show()  # 显示所有的图像
            for idx, image in enumerate(denoise_images2):
                image = (image.squeeze().permute(1, 2, 0) + 1) * 127.5
                image = image.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(image, aspect='auto')
                plt.axis("off")
            plt.show()  # 显示所有的图像
            # 绘制无病症方向的噪声
            plt.figure(figsize=(6, 6))  # 12x6英寸的图像大小，可以根据需要调整
            for idx, image in enumerate(pnoise2):
                image = (image.squeeze().permute(1, 2, 0) + 1) * 127.5
                image = image.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(image, aspect='auto')
                plt.axis("off")
            plt.show()  # 显示所有的图像
            # 绘制专家标注
            plt.figure(figsize=(6, 6))  # 12x6英寸的图像大小，可以根据需要调整
            for idx, mask_batch in enumerate(masks):
                mask_batch = (mask_batch.squeeze().permute(1, 2, 0) + 1)
                mask_batch = mask_batch.to("cpu").numpy()
                plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                plt.imshow(mask_batch, cmap='gray', aspect='auto')
                plt.axis("off")
            plt.show()  # 显示所有的图像

            # 噪声差异
            plt.figure(figsize=(6, 6))  # 12x6英寸的图像大小，可以根据需要调整
            for idx, (d1, d2) in enumerate(zip(pnoise, pnoise2)):
                # image=d2-d1
                d1 = (d1.squeeze().permute(1, 2, 0) + 1) * 127.5
                d1 = d1.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
                d1_tensor = torch.from_numpy(d1)
                d2 = (d2.squeeze().permute(1, 2, 0) + 1) * 127.5
                d2 = d2.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
                d2_tensor = torch.from_numpy(d2)
                import numpy as np

                # 将gray_image2转换为8位无符号整型（即灰度图像）
                # gray_image2 = gray_image2.astype(np.uint8)

                # 将d1_tensor的数据类型转换为float
                d1_tensor_float = d1_tensor.float()
                d2_tensor_float = d2_tensor.float()

                # 使用加权求和获取灰度图像
                gray_d1 = torch.sum(d1_tensor_float * args.weights, dim=-1, keepdim=True)
                gray_d1 = gray_d1.cpu().numpy()  # 转换为NumPy数组
                gray_d1 = gray_d1 - np.min(gray_d1)
                gray_d2 = torch.sum(d2_tensor_float * args.weights, dim=-1, keepdim=True)
                gray_d2 = gray_d2.cpu().numpy()  # 转换为NumPy数组
                gray_d2 = gray_d2 - np.min(gray_d2)
                gray_image2 = gray_d2 - gray_d1 * 1.5
                gray_image2 = (gray_image2 - np.min(gray_image2)) * (255 / np.max(gray_image2 - np.min(gray_image2)))
                ################################OPTIMIZEAREBELOW#############################################
                # 步骤1: 扩展通道数
                gray_image2_expanded = np.repeat(gray_image2, 3, axis=-1)  # 从(1, 128, 128, 1)扩展到(1, 128, 128, 3)
                # 步骤2: 调整维度顺序以匹配images的形状(torch.Size([1, 3, 128, 128]))
                gray_image2_expanded = np.transpose(gray_image2_expanded,
                                                    (0, 3, 1, 2))  # 从(1, 128, 128, 3)变换到(1, 3, 128, 128)
                # 将NumPy数组转换为torch张量，并确保数据类型匹配
                gray_image2_torch = torch.from_numpy(gray_image2_expanded).float()  # 确保转换后的张量与images的数据类型一致

                output_list = [gray_image2_torch]*10  # 这个列表可以动态添加任意数量的二维数组
                combined_output = np.stack(output_list)
                print(combined_output.shape)
                print(images.shape)
                densecrf_optimize(combined_output, images, K=10, H=5)
                # gray_image2_brightened = np.clip(gray_image2 * 1.2, 0, 255)
                #
                # # 二值化灰度图像
                # threshold = 180  # 阈值，可以根据需要调整
                # binary_image = np.where(gray_image2_brightened < threshold, 1, 0)
                # # 对二值图像进行外部白色区域的填充处理
                # plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
                # plt.imshow(gray_image2_brightened.squeeze(), cmap='gray', aspect='auto')  # 使用灰度颜色映射
                # plt.axis("off")
                # print(binary_image2)
            plt.show()  # 显示所有的图像
            # 噪声差异
            # plt.figure(figsize=(6, 6))  # 12x6英寸的图像大小，可以根据需要调整
            # for idx, (d1, d2) in enumerate(zip(pnoise, pnoise2)):
            #     # image=d2-d1
            #     d1 = (d1.squeeze().permute(1, 2, 0) + 1) * 127.5
            #     d1 = d1.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
            #     d1_tensor = torch.from_numpy(d1)
            #     d2 = (d2.squeeze().permute(1, 2, 0) + 1) * 127.5
            #     d2 = d2.to("cpu").numpy().astype(int)  # 将图像从GPU移回CPU并转换为NumPy数组
            #     d2_tensor = torch.from_numpy(d2)
            #     import numpy as np
            #     # 将d1_tensor的数据类型转换为float
            #     d1_tensor_float = d1_tensor.float()
            #     d2_tensor_float = d2_tensor.float()
            #     # 使用加权求和获取灰度图像
            #     gray_d1 = torch.sum(d1_tensor_float * args.weights, dim=-1, keepdim=True)
            #     gray_d1 = gray_d1.cpu().numpy()  # 转换为NumPy数组
            #     gray_d1 = gray_d1 - np.min(gray_d1)
            #     gray_d2 = torch.sum(d2_tensor_float * args.weights, dim=-1, keepdim=True)
            #     gray_d2 = gray_d2.cpu().numpy()  # 转换为NumPy数组
            #     gray_d2 = gray_d2 - np.min(gray_d2)
            #     gray_image2 = gray_d2 - gray_d1 * 1.5
            #     gray_image2 = (gray_image2 - np.min(gray_image2)) * (255 / np.max(gray_image2 - np.min(gray_image2)))
            #     gray_image2_brightened = np.clip(gray_image2 * 1.2, 0, 255)
            #     # 二值化灰度图像
            #     # 创建一个空列表来存储转换后的灰度图像
            #     # 假设 gray_image2_brightened 是一个包含四张图像的列表
            #     threshold = 189  # 阈值，可以根据需要调整
            #     binary_image = np.where(gray_image2_brightened < threshold, 1, 0)
            #
            #     # 对二值图像进行外部白色区域的填充处理
            #     plt.subplot(len(final_loader), 4, step * 4 + idx + 1)
            #     plt.imshow(binary_image.squeeze(), cmap='gray', aspect='auto')  # 使用灰度颜色映射
            #     plt.axis("off")
            #     # print(binary_image2)
            plt.show()  # 显示所有的图像
            # 计算 Dice 系数
            # intersection = np.logical_and(binary_image, mask_batch) * 255
            # has_number_greater_than_255 = (intersection > 255).any()
            # dice = (2.0 * np.sum(intersection)) / (np.sum(binary_image) * 255 + np.sum(mask_batch) * 255)
            # print(f"Step {step}, Image {idx}, Dice Score: {dice}")

    return None



def densecrf_optimize(segmentations, original_image, K=10, H=5):
    """
    使用DenseCRF优化分割结果

    :param segmentations: 分割结果集合，假设是一个形状为[N, H, W]的numpy数组
    :param original_image: 原始图像，形状为[H, W, C]
    :param K: 迭代次数
    :param H: 每次迭代随机选取的分割结果数量
    :return: 优化后的分割结果
    """
    if len(original_image.shape) == 2:
        original_image = gray2rgb(original_image)
    original_image = resize(original_image, (segmentations.shape[1], segmentations.shape[2]),
                            preserve_range=True, anti_aliasing=True).astype(np.uint8)

    final_results = []

    for k in range(K):
        # 随机选择H个分割结果
        selected_indices = random.sample(range(segmentations.shape[0]), H)
        selected_segmentations = segmentations[selected_indices]

        # 计算平均分割结果作为数据项
        avg_segmentation = np.mean(selected_segmentations, axis=0)
        avg_segmentation_softmax = np.stack((avg_segmentation, 1 - avg_segmentation), axis=-1)  # 转换为Softmax概率

        # 创建DenseCRF模型
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], 2)
        U = unary_from_softmax(avg_segmentation_softmax)
        d.setUnaryEnergy(U)

        # 添加颜色相关的平滑项
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image, compat=10)

        # 添加空间相关的平滑项
        d.addPairwiseGaussian(sxy=(3, 3), compat=3)

        # 进行能量最小化
        Q = d.inference(5)
        map_soln = np.argmax(Q, axis=0).reshape((original_image.shape[0], original_image.shape[1]))

        final_results.append(map_soln)

    # 计算所有迭代结果的平均值
    final_avg_result = np.mean(final_results, axis=0)

    # 显示最终平均结果
    plt.imshow(final_avg_result, cmap='gray')
    plt.title('Final Average Result')
    plt.axis('off')
    plt.show()

    return final_avg_result


if __name__ == '__main__':
    args = DefaultConfig()
    train_unhealthy(args)
    #train_healthy(args)
    #test2(args)


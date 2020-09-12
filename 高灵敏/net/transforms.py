import torch
import torchvision.transforms as transforms
import numpy as np
import random
import torchvision.transforms.functional as TF



def image_transforms(mode='train', augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                     do_augmentation=True, transformations=None,  size=(256, 512)):
    if mode == 'train':
        data_transform = transforms.Compose([
            # ResizeImage(train=True, size=size),            
            
            # RandomFlipHor(do_augmentation),
            # RandomFlipVer(do_augmentation),
            # randomRotation(do_augmentation),
            ToTensor(train=True),
            
            # AugmentImagePair(augment_parameters, do_augmentation),
            
        ])
        return data_transform

    elif mode == 'test':
        data_transform = transforms.Compose([
            # ResizeImage(train=False, size=size),
            ToTensor(train=False), 
 
        
        ])
        return data_transform
    elif mode == 'custom':
        data_transform = transforms.Compose(transformations)
        return data_transform
    else:
        print('Wrong mode')
        
# class ResizeImage(object):
#     def __init__(self, train=True, size=(256, 512)):
#         self.train = train
#         self.transform = transforms.Resize(size)
       

#     def __call__(self, sample):
#         if self.train:
#             left_image = sample['left_image']
#             right_image = sample['right_image']
#             # right_image = self.transform(right_image)
#             # left_image = self.transform(left_image)
#             w, h = left_image.size
#             th, tw = 256, 512
    
#             x1 = random.randint(0, w - tw)
#             y1 = random.randint(0, h - th)

#             left_image = left_image.crop((x1, y1, x1 + tw, y1 + th))
#             right_image = right_image.crop((x1, y1, x1 + tw, y1 + th))

#             sample = {'left_image': left_image, 'right_image': right_image}
#         else:
#             # left_image = sample
#             # new_left_image = self.transform(left_image)
#             # sample = new_left_image
#             left_image = sample['left_image']
#             right_image = sample['right_image']
#             # left_image = self.transform(left_image)
#             # right_image = self.transform(right_image)
#             sample = {'left_image': left_image, 'right_image': right_image}
#         return sample


class DoTest(object):
    def __call__(self, sample):
        new_sample = torch.stack((sample, torch.flip(sample, [2])))
        return new_sample


class ToTensor(object):
    def __init__(self, train):
        self.train = train
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        if self.train:
            left_image = sample['left_image']
            bg_image = sample['bg_image']


            left_image = self.transform(left_image)
            bg_image =self.transform(bg_image)


            
           
            
            sample = {'left_image': left_image,'bg_image': bg_image,}
            # sample = {'left_image': new_left_image, 'mask_image': new_mask_image,'D2_image': new_D2_image}
        else:

            left_image = sample['left_image']
            bg_image = sample['bg_image']
            
            left_image = self.transform(left_image)
            bg_image =self.transform(bg_image)


            
           
            
            sample = {'left_image': left_image,'bg_image': bg_image}
            # sample = {'left_image': new_left_image, 'mask_image': new_mask_image,'D2_image': new_D2_image}
        

        return sample





class AugmentImagePair(object):
    def __init__(self, augment_parameters, do_augmentation):
        self.do_augmentation = do_augmentation
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.5
        self.brightness_high = augment_parameters[3]  # 2.0
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2

    def __call__(self, sample):
        left_image = sample['left_image']
        bg_image = sample['bg_image']
        # mask_image = sample['mask_image']
        # D7_image = sample['D7_image']
        # right_image = sample['right_image']
        p = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if p > 0.5:
                # randomly shift gamma
                random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
                left_image_aug = left_image ** random_gamma
                # right_image_aug = right_image ** random_gamma

                # randomly shift brightness
                random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
                left_image_aug = left_image_aug * random_brightness
                # right_image_aug = right_image_aug * random_brightness

                # randomly shift color
                random_colors = np.random.uniform(self.color_low, self.color_high, 1)
                for i in range(1):
                    left_image_aug[i, :, :] *= random_colors[i]
                    # right_image_aug[i, :, :] *= random_colors[i]
                # left_image_aug*= random_colors
                # saturate
                left_image_aug = torch.clamp(left_image_aug, 0, 1)
                # right_image_aug = torch.clamp(right_image_aug, 0, 1)

                sample = {'left_image': left_image_aug,'bg_image': bg_image}
                

        else:
            sample = {'left_image': left_image,'bg_image': bg_image}
        return sample



class RandomFlipHor(object):
    def __init__(self, do_augmentation):
        self.transform = transforms.RandomHorizontalFlip(p=1)
        self.do_augmentation = do_augmentation

    def __call__(self, sample):
        left_image = sample['left_image']
        bg_image = sample['bg_image']
        # mask_image = sample['mask_image']
        # D7_image = sample['D7_image']
        # right_image = sample['right_image']
        k = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if k > 0.5:
                # right_image = self.transform(right_image)
                left_image = self.transform(left_image)
                bg_image = self.transform(bg_image)
                D7_image = self.transform(D7_image)
                mask_image = self.transform(mask_image)
                # random_angle = np.random.randint(1, 360)
                # left_image.rotate(random_angle)
                sample = {'left_image': left_image,'bg_image': bg_image}
        else:
            sample = {'left_image': left_image,'bg_image': bg_image}
        return sample

class RandomFlipVer(object):
    def __init__(self, do_augmentation):
        self.transform = transforms.RandomVerticalFlip(p=1)
        self.do_augmentation = do_augmentation

    def __call__(self, sample):
        left_image = sample['left_image']
        bg_image = sample['bg_image']
        mask_image = sample['mask_image']
        D7_image = sample['D7_image']
        # right_image = sample['right_image']
        k = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if k > 0.5:
                # right_image = self.transform(right_image)
                left_image = self.transform(left_image)
                bg_image = self.transform(bg_image)
                D7_image = self.transform(D7_image)
                mask_image = self.transform(mask_image)
                # random_angle = np.random.randint(1, 360)
                # left_image.rotate(random_angle)
                sample = {'left_image': left_image,'bg_image': bg_image}
        else:
            sample = {'left_image': left_image,'bg_image': bg_image}
        return sample


class randomRotation(object):
    def __init__(self, do_augmentation):
        self.transform = transforms.RandomRotation(90,90)
        self.do_augmentation = do_augmentation

    def __call__(self, sample):
        left_image = sample['left_image']
        bg_image = sample['bg_image']
        mask_image = sample['mask_image']
        D7_image = sample['D7_image']
        # right_image = sample['right_image']
        k = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if k > 0.5:
                # right_image = self.transform(right_image)
                angle = 90
                left_image1 = TF.rotate(left_image, angle)
                bg_image1 = TF.rotate(bg_image, angle)
                D7_image1 = TF.rotate(D7_image, angle)
                mask_image1 = TF.rotate(mask_image, angle)
                # left_image1 = self.transform(left_image)
                # bg_image1 = self.transform(bg_image)
                # D7_image1 = self.transform(D7_image)
                # mask_image1 = self.transform(mask_image)
                sample = {'left_image': left_image1,'bg_image': bg_image1}
        else:
            sample = {'left_image': left_image,'bg_image': bg_image}
        return sample

# class RandomFlip(object):
#     def __init__(self, do_augmentation):
#         self.transform = transforms.RandomHorizontalFlip(p=1)
#         self.do_augmentation = do_augmentation

#     def __call__(self, sample):
#         left_image = sample['left_image']
#         # right_image = sample['right_image']
#         k = np.random.uniform(0, 1, 1)
#         if self.do_augmentation:
#             if k > 0.5:

                
#                 random_angle = np.random.randint(1, 360)
#                 left_image.rotate(random_angle)
#                 sample = {'left_image': left_image}
#         else:
#             sample = {'left_image': left_image}
#         return sample
 
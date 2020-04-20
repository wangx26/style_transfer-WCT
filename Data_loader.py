from torch.utils.data import Dataset
import os
from PIL import Image
import numpy
import torchvision.transforms as transforms

supported_img_formats = ('.png', '.jpg', 'jpeg')

class Data(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()

        self.finesize = args.finesize

        self.image_list = []#图片列表，分输入文件和文件夹两种，输入文件夹为批量处理
        if os.path.isfile(args.content) and os.path.isfile(args.style):
            if args.content.endswith(supported_img_formats) and args.style.endswith(supported_img_formats):
                self.image_list = [(args.content, args.style)]
        elif os.path.isdir(args.content) and os.path.isdir(args.style):
            for c in os.listdir(args.content):
                for s in os.listdir(args.style):
                    self.image_list.append((os.path.join(args.content, c), os.path.join(args.style, s)))
        else:
            print('The path is not right.')

    def __getitem__(self, index):
        #读入第index图片
        contentImg = Image.open(self.image_list[index][0]).convert('RGB')
        styleImg = Image.open(self.image_list[index][1]).convert('RGB')

        #resize图片尺寸
        if(self.finesize):
            w, h = contentImg.size
            neww, newh = w, h
            if(w > h):
                if(w != self.finesize):
                    neww = self.finesize
                    newh = int(h * neww / w)
            else:
                if(h != self.finesize):
                    newh = self.finesize
                    neww = int(w * newh / h)
            contentImg = contentImg.resize((neww, newh))
            styleImg = styleImg.resize((neww, newh))

        contentImg = transforms.ToTensor()(contentImg)
        styleImg = transforms.ToTensor()(styleImg)

        return contentImg.squeeze(0), styleImg.squeeze(0), self.image_list[index]

    def __len__(self):
        return len(self.image_list)
import os
import json
import random
import datetime
from functools import partial

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# visualization
import matplotlib.pyplot as plt
from skimage import color

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2' 
device = torch.device('cuda')

IMAGE_ROOT = "/data2/kdg_datasets/segmentation/deta/train/DCM"
LABEL_ROOT = "/data2/kdg_datasets/segmentation/deta/train/outputs_json"

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
CLASS2IND.items()
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

BATCH_SIZE = 8
LR = 1e-4
RANDOM_SEED = 21

NUM_EPOCHS = 20
VAL_EVERY = 20

SAVED_DIR = "/home/"

if not os.path.exists(SAVED_DIR):                                                           
    os.makedirs(SAVED_DIR)

for root, _dirs, files in os.walk(IMAGE_ROOT): # os.walk를 통해 폴더 내부 재귀적 탐색 root: dir, file이 있는 path / dirs: root 하위 폴더들, files: root 하위 파일들
    print(f'root: {root}')
    print(f'files: {files}')

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT) # relpath 로 상대경로
    for root, _dirs, files in os.walk(IMAGE_ROOT) # os.walk로 특정 경로 하위 디렉토리 탐색
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png" # splitext로 파일의 확장자 추출 [0]은 파일명 [1]은 확장자
}

jsons = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
}

jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

assert len(jsons_fn_prefix - pngs_fn_prefix) == 0 # assert로 해당 조건이 참이 아니면 에러발생
assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

pngs = sorted(pngs)
jsons = sorted(jsons)

filename=np.array(pngs)

groups=[os.path.dirname(fname) for fname in filename] # os.path.dirname 으로 상위 디렉토리 반환
groups # 각 png 파일의 상위 폴더 ID001... 을 반환

_filenames = np.array(pngs)
groups = [os.path.dirname(fname) for fname in _filenames]

ys = [0 for fname in _filenames]
gkf = GroupKFold(n_splits=5) 

for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
    print(f'x: {x}')
    print(f'y: {y}')

class XRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None):
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)

        groups = [os.path.dirname(fname) for fname in _filenames] # 각 png 파일에 대한 상위 폴더
        # os.path.dirname 으로 파일이 들어있는 폴더 이름 반환

        # dummy label
        ys = [0 for fname in _filenames]
        
        gkf = GroupKFold(n_splits=5) 
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):  # x: train data, y: validation data (x,y)는 index로 출력되며 각 fold의 y는 중복되지 않음
            if is_train:  # is_train 이라면
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue # 0 번 fold는 제외
                    
                filenames += list(_filenames[y]) # 0번째 fold를 제외한 y값을 모두 취합
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])  # train이 아니면 0번째 fold의 y 값을 저장 나머지는 저장 안 함
                labelnames = list(_labelnames[y])
                
                # skip i > 0
                break
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # (H, W, NC) 모양의 label을 생성
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )  # 2048*2048*29
        label = np.zeros(label_shape, dtype=np.uint8)

        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        

        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            

            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label # ...은 생략부호 ...부분은 (2048*2048) 의미 해당 부분에 fillpoly를 통해 값 할당
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image} # train 이면 image와 label 반환 val, test면 
            result = self.transforms(**inputs) # dict 형태 가변인자 입력으로
            
            image = result["image"]
            label = result["mask"] if self.is_train else label


        image = image.transpose(2, 0, 1)    
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label

tf = A.Resize(1024, 1024)

train_dataset = XRayDataset(is_train=True, transforms=tf)
valid_dataset = XRayDataset(is_train=False, transforms=tf)

image, label = train_dataset[0]

print(image.shape, label.shape)

label_np = label.cpu().numpy()

#visualization
PALETTE=[(220, 20, 60), (119, 11, 32), (0,0,142), (0,0,230), (106,0,228),
         (0,60,100), (0,80,100), (0,0,70), (0,0,192), (250, 170,30),
         (100,170,30), (220,220,0), (175, 116, 175), (250,0,30), (165, 42, 42),
         (255, 77, 255), (0, 226, 252), (182, 182, 255), (0,82,0), (120, 166, 157),
         (110, 76, 0), (174, 57, 255), (199,100,0), (72, 0, 118), (255, 179, 240),
         (0, 125, 92), (209, 0, 151), (188,208,182), (0,220,176)]

def label2rgb(label):
    image_size=label.shape[1:] +(3, )
    image=np.zeros(image_size, dtype=np.uint8)

    for i, class_label in enumerate(label):
        image[class_label==1]=PALETTE[i]

    return image

fig, ax = plt.subplots(1, 2, figsize=(24, 12))
ax[0].imshow(image[0])   
ax[1].imshow(label2rgb(label_np))

plt.show()

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True, # 마지막 batch는 사용하지 않음
)

valid_loader = DataLoader(
    dataset=valid_dataset, 
    batch_size=1,
    shuffle=False,
    num_workers=0,
    drop_last=False
)

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def save_model(model, file_name='fcn_resnet50_best_model.pt'):
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save(model, output_path)

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

def train(model, data_loader, val_loader, criterion, optimizer):
    print(f'Start training..')
    
    n_class = len(CLASSES)
    best_dice = 0.
    
    for epoch in range(NUM_EPOCHS):
        model.train()

        for step, (images, masks) in enumerate(data_loader):            
            # gpu 연산을 위해 device 할당합니다.
            # images, masks = images.cuda(), masks.cuda()
            # model = model.cuda()
            
            images, masks = images.to(device), masks.to(device)
            model = model.to(device)


            outputs = model(images)['out']
            
            # loss를 계산합니다.
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
             
        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model)

def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # images, masks = images.cuda(), masks.cuda()         
            # model = model.cuda()
            
            images, masks = images.to(device), masks.to(device)         
            model = model.to(device)

            outputs = model(images)['out']
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    return avg_dice

model = models.segmentation.fcn_resnet50(pretrained=True)

# output class 개수를 dataset에 맞도록 
model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)
model = nn.DataParallel(model) 
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6)

set_seed()
train(model, train_loader, valid_loader, criterion, optimizer)

model = torch.load(os.path.join(SAVED_DIR, "fcn_resnet50_best_model.pt"))

IMAGE_ROOT = "/data2/kdg_datasets/segmentation/deta/test/DCM"

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}


def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)

class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  
        
        image = torch.from_numpy(image).float()
            
        return image, image_name
    
def test(model, data_loader, thr=0.5):
    # model = model.cuda()
    model = model.to(device)
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # images = images.cuda()
            images = images.to(device)    
            outputs = model(images)['out']
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class

tf = A.Resize(1024, 1024)

test_dataset = XRayInferenceDataset(transforms=tf)

test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
)

rles, filename_and_class = test(model, test_loader)

# result visualization

filename_and_class[0]
image = cv2.imread(os.path.join(IMAGE_ROOT, filename_and_class[0].split("_")[1]))

preds = []
for rle in rles[:len(CLASSES)]:
    pred = decode_rle_to_mask(rle, height=2048, width=2048)
    preds.append(pred)

preds = np.stack(preds, 0)

fig, ax = plt.subplots(1, 2, figsize=(24, 12))
ax[0].imshow(image)    # remove channel dimension
ax[1].imshow(label2rgb(preds))

plt.show()

classes, filename = zip(*[x.split("_") for x in filename_and_class])

image_name = [os.path.basename(f) for f in filename]

df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
})

df.head(10)

df.to_csv("output.csv", index=False)

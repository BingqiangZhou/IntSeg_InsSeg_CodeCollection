import socket
import timeit
from datetime import datetime
import os
import glob
from collections import OrderedDict

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import upsample

# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import coco
from networks import deeplab_resnet as resnet
from layers.loss import class_balanced_cross_entropy_loss
from dataloaders import custom_transforms as tr
from dataloaders.utils import generate_param_report

gpu_id = 0
print('Using GPU: {} '.format(gpu_id))
# Setting parameters
nEpochs = 200  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume

p = OrderedDict()  # Parameters to include in report
classifier = 'psp'  # Head classifier to use
p['trainBatch'] = 4  # Training batch size
testBatch = 4  # Testing batch size
useTest = True  # See evolution of the test set when training
nTestInterval = 10  # Run on test set every nTestInterval epochs
snapshot = 1  # Store a model every snapshot epochs
nInputChannels = 4  # Number of input channels (RGB + Distance Map of bounding box)
zero_pad_crop = True  # Insert zero padding when cropping the image
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 1e-4  # Learning rate
p['wd'] = 5e-4  # Weight decay
p['momentum'] = 0.9  # Momentum

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
if not os.path.exists(os.path.join(save_dir, 'models')):
    os.makedirs(os.path.join(save_dir, 'models'))

# Network definition
modelName = 'deepgc_pascal'
net = resnet.resnet101(1, pretrained=True, nInputChannels=nInputChannels, classifier=classifier)

if resume_epoch == 0:
    print("Initializing from pretrained Deeplab-v2 model")
else:
    print("Initializing weights from: {}".format(
        os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    net.load_state_dict(
        torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage))  # Load all tensors onto the CPU

# TODO: Load model trained on VOC and SBD datasets

train_params = [{'params': resnet.get_1x_lr_params(net), 'lr': p['lr']},
                {'params': resnet.get_10x_lr_params(net), 'lr': p['lr'] * 10}]

if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()

if resume_epoch != nEpochs:
    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # Use the following optimizer
    optimizer = optim.SGD(train_params, lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    p['optimizer'] = str(optimizer)

    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-15, 15), scales=(.75, 1.25)),
        tr.FixedResize(resolutions={'image': (450, 450), 'gt': (450, 450)}),
        tr.DistanceMap(v=0.15, elem='gt'),
        tr.ConcatInputs(elems=('image', 'distance_map')),
        tr.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        tr.FixedResize(resolutions={'image': (450, 450), 'gt': (450, 450)}),
        tr.DistanceMap(v=0.15, elem='gt'),
        tr.ConcatInputs(elems=('image', 'distance_map')),
        tr.ToTensor()])

    coco_train = coco.COCOSegmentation(split='train', transform=composed_transforms_tr)
    coco_val = coco.COCOSegmentation(split='val', transform=composed_transforms_ts)

    trainloader = DataLoader(coco_train, batch_size=p['trainBatch'], shuffle=True, num_workers=4)
    testloader = DataLoader(coco_val, batch_size=testBatch, shuffle=False, num_workers=4)

    generate_param_report(os.path.join(save_dir, exp_name + '.txt'), p)

    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    running_loss_tr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    print("Training Network")

    # Main Training and Testing Loop
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()

        net.train()
        for ii, sample_batched in enumerate(trainloader):

            inputs, gts = sample_batched['concat'], sample_batched['gt']

            # Forward-Backward of the mini-batch
            inputs, gts = Variable(inputs, requires_grad=True), Variable(gts)
            if gpu_id >= 0:
                inputs, gts = inputs.cuda(), gts.cuda()

            output = net.forward(inputs)
            output = upsample(output, size=(450, 450), mode='bilinear', align_corners=True)

            # Compute the losses, side outputs and fuse
            loss = class_balanced_cross_entropy_loss(output, gts, size_average=True, batch_average=True)
            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == num_img_tr - 1:
                running_loss_tr = running_loss_tr / num_img_tr
                writer.add_scalar('train/total_loss_epoch', running_loss_tr, epoch)
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['trainBatch'] + inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                running_loss_tr = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")

            # Backward the averaged gradient
            loss /= p['nAveGrad']
            loss.backward()
            aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % p['nAveGrad'] == 0:
                writer.add_scalar('train/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

        # Save the model
        if (epoch % snapshot) == snapshot - 1 and epoch != 0:
            torch.save(net.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))

        # One testing epoch
        if useTest and epoch % nTestInterval == (nTestInterval - 1):
            net.eval()
            for ii, sample_batched in enumerate(testloader):
                inputs, gts = sample_batched['concat'], sample_batched['gt']

                # Forward pass of the mini-batch
                inputs, gts = Variable(inputs, requires_grad=True), Variable(gts)
                if gpu_id >= 0:
                    inputs, gts = inputs.cuda(), gts.cuda()

                with torch.no_grad():
                    output = net.forward(inputs)
                output = upsample(output, size=(450, 450), mode='bilinear', align_corners=True)

                # Compute the losses, side outputs and fuse
                loss = class_balanced_cross_entropy_loss(output, gts, size_average=True)
                running_loss_ts += loss.item()


                # Print stuff
                if ii % num_img_ts == num_img_ts - 1:
                    running_loss_ts = running_loss_ts / num_img_ts
                    writer.add_scalar('val/total_loss_epoch', running_loss_ts, epoch)
                    print('Validation:')
                    print('[Epoch: %d, numImages: %5d]' % (epoch, ii * testBatch + inputs.data.shape[0]))
                    writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
                    print('Loss: %f' % running_loss_ts)
                    running_loss_ts = 0

    writer.close()

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:52:50 2020

@author: ssridhar
"""

# testing the model
def test_net(net, device, testloader):
    import torch
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
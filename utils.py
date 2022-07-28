from config import *

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def save_torch_model(model, name):
    torch.save(model.state_dict(), name)

def get_test_results(model, dataloader):
    label = np.array([])
    label_predicted = np.array([])

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # append labels
        label = np.append(label, labels.numpy())
        label_predicted = np.append(label_predicted, preds.cpu().numpy())

    return label, label_predicted

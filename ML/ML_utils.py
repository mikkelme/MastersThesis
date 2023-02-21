from module_import import *

def seed_everything(seed: int):
    # import random, os
    # import numpy as np
    # import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# seed_everything(5400)

def get_device(ML_setting):
    # Device
    if ML_setting['use_gpu']:
      device = torch.device('cuda:0')
    else:
      device = torch.device('cpu')

    return device

# def get_outputs(mode, model, data, device):
#     if mode == 0: # Full image (all channels)
#         rgb_inputs = data['image'].to(device)
#         outputs = model(rgb_inputs)
#         return outputs

#     if mode == 1: # Split image into rgb and infrared channels
#         rgb_inputs = data['image'][:, 0:3].to(device)
#         infrared_inputs = data['image'][:, 3:].to(device)
#         outputs = model(rgb_inputs, infrared_inputs)
#         return outputs


def train_epoch(model, trainloader, criterion, optimizer, device):
    model.train() # Put into training mode
    losses = []

    num_batches = len(trainloader)
    progress_bar_length = 8
    
    for batch_idx, data in enumerate(trainloader):
        optimizer.zero_grad() # sets gradients of all optimized torch.Tensor's to zero.


        # Inputs
        config = data['config']
        config_shape = config.size()[1:]
        stretch = torch.tensor(np.array([np.full(config_shape, s, dtype=np.float32) for s in data['stretch']]))
        FN = torch.tensor(np.array([np.full(config_shape, f, dtype=np.float32) for f in data['F_N']]))
        input = torch.stack((config, stretch, FN), 1).to(device) # Gather inputs on multiple channels
        # XXX For some reason I think the .to(device) is nessecary but check up on it
        
        # Labels
        Ff = torch.tensor(np.array(data['Ff_mean'], dtype = np.float32))
        is_ruptured = torch.tensor(np.array(data['is_ruptured'], dtype = np.int32))
        labels = torch.stack((Ff, is_ruptured), 1).to(device) 
        
        # XXX When mergening float32 with int32 I believe it stores both as float32 anyway...
        
       
        output = model(input)
        
        #
        # XXX Working here XXX
        #
        exit()
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # print progress
        progress = int(((batch_idx+1)/num_batches)*progress_bar_length)
        print(f'\rLoss : {np.mean(losses):.4f} |{progress* "="}>{(progress_bar_length-progress)* " "}| {batch_idx+1}/{num_batches} ({100*(batch_idx+1)/num_batches:2.0f}%)', end = '')

    print()
    return np.mean(losses)


# def evaluate_model(mode, model, dataloader, criterion, device, numcl):
#     model.eval()

#     concat_outputs =  np.empty((0, numcl)) # model outputs
#     concat_preds   =  np.empty((0, numcl)) # model predictions (binary)
#     concat_labels  =  np.empty((0, numcl)) # data labels
#     avgprecs = np.zeros(numcl) # average precision for each class
#     filenames = [] # filenames as they come out of the dataloader


#     with torch.no_grad():
#         losses = []
#         for batch_idx, data in enumerate(dataloader):
#             labels = data['label'].to(device)
#             outputs = get_outputs(mode, model, data, device)
#             loss = criterion(outputs, labels.type(torch.float))
#             losses.append(loss.item())

#             cpu_outputs = outputs.to('cpu')
#             cpu_preds = torch.round(outputs).to('cpu')
#             cpu_labels = labels.float().to('cpu')



#             concat_outputs = np.concatenate((concat_outputs, cpu_outputs), axis=0)
#             concat_preds = np.concatenate((concat_preds, cpu_preds), axis=0)
#             concat_labels = np.concatenate((concat_labels, cpu_labels), axis=0)
#             filenames += data['filename']

#         for c in range(numcl):
#             y_true = concat_labels[:,c]
#             y_true = np.where(np.logical_and(y_true != 0, y_true != 1), y_true > 0.5, y_true) # fix small label mistakes
#             y_scores = concat_outputs[:,c]
#             if np.sum(y_true) != 0:
#                 avgprecs[c] = average_precision_score(y_true, y_scores)
#             else:
#                 avgprecs[c] = 0

#             # true_positives = np.sum(np.logical_and(concat_preds[:,c] == 1, concat_labels[:,c] == 1))
#             # total_positives = np.sum(concat_preds[:,c])
#             # if total_positives != 0:
#             #     precision = true_positives / total_positives
#             # else:
#             #     precision = 0

#             # avgprecs[c] = precision


#     return avgprecs, np.mean(losses), concat_outputs, concat_labels, concat_preds, filenames


def train_and_evaluate(mode, dataloader_train, dataloader_test, model, criterion, optimizer, scheduler, num_epochs, device, numcl):
    best_avgprec = 0
    best_epoch = -1

    train_losses = []
    test_losses = []
    test_precs = []

    print('Training model')
    for epoch in range(num_epochs):
        print('-' * 14)
        print('Epoch: {}/{}'.format(epoch+1, num_epochs))

        avgloss = train_epoch(mode, model, dataloader_train, criterion, device, optimizer )
        train_losses.append(avgloss)

        if scheduler is not None:
            scheduler.step()


        avgprecs, avgloss, concat_outputs, concat_labels, concat_preds, filenames  = evaluate_model(mode, model, dataloader_test, criterion, device, numcl)
        test_precs.append(avgprecs)
        test_losses.append(avgloss)

        combined_avgprec = np.mean(avgprecs)
        print(f'avg. precision: {combined_avgprec:.5f}')

        if combined_avgprec > best_avgprec:
            best_weights = model.state_dict()
            best_avgprec = combined_avgprec
            best_epoch = epoch
            best_scores = concat_outputs

    print('-' * 14)
    return best_epoch, best_avgprec, best_weights, best_scores, train_losses, test_losses, test_precs

# def save_training_history(session_name, train_losses, test_losses, test_precs):
#     filename = session_name + '_training_history.txt'
#     outfile = open('./' + filename, 'w')
#     numcl = len(test_precs[0])
#     outfile.write(f'numcl = {numcl}\n')
#     outfile.write('train_loss test_loss test_precs \n')

#     for i in range(len(train_losses)):
#         outfile.write(f'{train_losses[i]} {test_losses[i]}')
#         for j in range(numcl):
#             outfile.write(f' {test_precs[i][j]}')
#         outfile.write('\n')
#     outfile.close()


# def save_best_model(session_name, model, best_weights):
#     modelname = session_name + '_model_dict_state'
#     model.load_state_dict(best_weights)
#     torch.save(model.state_dict(), './' + modelname)



# def save_best_model_val_scores(session_name, best_scores, best_epoch):
#     filename = session_name + '_best_epoch_scores.txt'
#     outfile = open(filename, 'w')
#     outfile.write(f'# best epoch: {best_epoch}\n')
#     outfile.write(f'# shape: {best_scores.shape[0]}, {best_scores.shape[1]}\n')
#     for i in range(best_scores.shape[0]):
#         for j in range(best_scores.shape[1]):
#             outfile.write(f'{best_scores[i,j]} ')
#         outfile.write('\n')
#     outfile.close()

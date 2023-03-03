from module_import import *

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_device(ML_setting):
    # Device
    if ML_setting['use_gpu']:
      device = torch.device('cuda:0')
    else:
      device = torch.device('cpu')

    return device


# def get_inputs(data, device):
#     config = data['config']
#     config_shape = config.size()[1:]
#     stretch = torch.from_numpy(np.array([np.full(config_shape, s, dtype=np.float32) for s in data['stretch']]))
#     FN = torch.from_numpy(np.array([np.full(config_shape, f, dtype=np.float32) for f in data['F_N']]))
#     inputs = torch.stack((config, stretch, FN), 1).to(device) # Gather inputs on multiple channels
#     # XXX For some reason I think the .to(device) is nessecary but check up on it
    
#     return inputs
    

def get_inputs(data, device):
    image = data['config']
    stretch = torch.from_numpy(np.array(data['stretch'], dtype = np.float32))
    FN = torch.from_numpy(np.array(data['F_N'], dtype = np.float32))
    vals = torch.stack((stretch, FN), 1).to(device)
    return image, vals
    
def get_labels(data, device):
    Ff = torch.from_numpy(np.array(data['Ff_mean'], dtype = np.float32))
    is_ruptured = torch.from_numpy(np.array(data['is_ruptured'], dtype = np.int32))
    labels = torch.stack((Ff, is_ruptured), 1).to(device) 
    # XXX When mergening float32 with int32 I believe it stores both as float32 anyway...

    return labels




def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train() # Training mode
    losses = []

    num_batches = len(dataloader)
    progress_bar_length = 8
    
    for batch_idx, data in enumerate(dataloader):
        # Zero gradients of all optimized torch.Tensor's
        optimizer.zero_grad() 

        # --- Evaluate --- #    
        image, vals = get_inputs(data, device)
        labels = get_labels(data, device)
        
        outputs = model(image, vals)
        # print(outputs)
        loss, MSE, BCE = criterion(outputs, labels)
       
        # --- Optimize --- #
        loss.backward()
        optimizer.step()
        losses.append([loss.item(), MSE.item(), BCE.item()])

        # --- print progress --- #
        progress = int(((batch_idx+1)/num_batches)*progress_bar_length)
        print(f'\rLoss : {np.mean(losses):.4f} |{progress* "="}>{(progress_bar_length-progress)* " "}| {batch_idx+1}/{num_batches} ({100*(batch_idx+1)/num_batches:2.0f}%)', end = '')

    print()
    losses = np.array(losses)
    return np.mean(losses, axis = 0)


def evaluate_model(model, dataloader, criterion, device):
    model.eval() # Evaluation mode

    with torch.no_grad():
        losses = []
        abs_error = []
        accuracy = []
        
        for batch_idx, data in enumerate(dataloader):
            # --- Evaluate --- #
            image, vals = get_inputs(data, device)
            labels = get_labels(data, device)
            outputs = model(image, vals)
            loss, MSE, BCE = criterion(outputs, labels)
       
            # --- Analyse --- #
            losses.append([loss.item(), MSE.item(), BCE.item()])
            non_rupture = labels[:, 1] < 0.5
            
            # Additional metrics
            abs_error.append(torch.mean(torch.abs(outputs[non_rupture,0] - labels[non_rupture, 0])))
            rup_pred = torch.round(outputs[:,1])
            accuracy.append((torch.sum(rup_pred == labels[:,1])/len(rup_pred)).item())
            
        
        losses = np.array(losses)
        abs_error = np.array(abs_error)
        accuracy = np.array(accuracy)
        return np.mean(losses, axis = 0), np.mean(abs_error), np.mean(accuracy)



def train_and_evaluate(model, dataloaders, criterion, optimizer, scheduler, ML_setting, device, save_best = False):
    train_val_hist = {'epoch': [],
                      'train_loss_TOT': [],
                      'train_loss_MSE': [],
                      'train_loss_BCE': [],
                      'val_loss_TOT': [],
                      'val_loss_MSE': [],
                      'val_loss_BCE': [],
                      'abs': [],
                      'acc': [] }   
    
    best = {'epoch': -1, 'loss': 1e6, 'weights': None}
    num_epochs = ML_setting['maxnumepochs']

    print('Training model')
    for epoch in range(num_epochs):
        try:
            print('-' * 14)
            print(f'Epoch: {epoch+1}/{num_epochs}')


            avgloss = train_epoch(model, dataloaders['train'], criterion, optimizer, device)

            train_val_hist['train_loss_TOT'].append(avgloss[0])
            train_val_hist['train_loss_MSE'].append(avgloss[1])
            train_val_hist['train_loss_BCE'].append(avgloss[2])
            
            if scheduler is not None: # TODO: Check this
                scheduler.step()

            avgloss, avg_abs_error, avgacc = evaluate_model(model, dataloaders['val'], criterion, device)
            
            train_val_hist['val_loss_TOT'].append(avgloss[0])
            train_val_hist['val_loss_MSE'].append(avgloss[1])
            train_val_hist['val_loss_BCE'].append(avgloss[2])
            train_val_hist['abs'].append(avg_abs_error)
            train_val_hist['acc'].append(avgacc)
            train_val_hist['epoch'].append(epoch)
            
            print(f'val_loss: {avgloss[0]:g}, abs: {avg_abs_error:g}, acc: {avgacc:g}')
            
            
            if save_best:
                if avgloss[0] < best['loss']: # TODO: Best criteria?
                    best['epoch'] = epoch
                    best['loss'] = avgloss[0]
                    best['abs'] = avg_abs_error
                    best['acc'] = avgacc
                    best['weights'] = model.state_dict()
                    
        
        
        except KeyboardInterrupt: break
    print('-' * 14)
    
    
    for key in train_val_hist:
        train_val_hist[key] = np.array(train_val_hist[key])
    
    
    return train_val_hist, best
    # return np.array(train_losses), np.array(validation_losses), best

def save_training_history(name, train_val_hist, ML_setting):
    filename = name + '_training_history.txt'
    outfile = open(filename, 'w')
    
    for key in ML_setting:
        outfile.write(f'# {key} = {ML_setting[key]}\n')
    
    epochs = train_val_hist['epoch']
    
    keys = ''
    for key in train_val_hist:
        keys += f'{key} '
    keys += '\n' # add linebreak
    outfile.write(keys)
    
    for i, epoch in enumerate(epochs):
        for key in train_val_hist:
            data = train_val_hist[key][i]
            outfile.write(f'{data} ')
        outfile.write('\n')
    outfile.close()


def save_best_model_scores(name, best, ML_setting):
    filename = name + '_best_scores.txt'
    
    outfile = open(filename, 'w')
    for key in ML_setting:
        outfile.write(f'# {key} = {ML_setting[key]}\n')
    for key in best:
        if key != 'weights':
            outfile.write(f'{key} = {best[key]}\n')
    outfile.close()


def save_best_model(name, model, best_weights):
    modelname = name + '_model_dict_state'
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), modelname)


def load_weights(model, weight_path, use_gpu = False):
    
    if use_gpu:
        model.load_state_dict(torch.load(weight_path))
    else:
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    
    return model


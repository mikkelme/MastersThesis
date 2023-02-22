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
        Ff_loss = []
        rup_loss = []
        for batch_idx, data in enumerate(dataloader):
            # --- Evaluate --- #
            image, vals = get_inputs(data, device)
            labels = get_labels(data, device)
            outputs = model(image, vals)
            loss, MSE, BCE = criterion(outputs, labels)
       
            # --- Analyse --- #
            losses.append([loss.item(), MSE.item(), BCE.item()])
            # Possibilities to do some more analysis here for accuracy or whatever
            # Check IN5400 mandatory 1
        
        losses = np.array(losses)
        return np.mean(losses, axis = 0)



def train_and_evaluate(model, dataloaders, criterion, optimizer, scheduler, ML_setting, device):
    best_avgprec = 0
    best_epoch = -1

    train_losses = []
    validation_lossed = []
    
    num_epochs = ML_setting['maxnumepochs']

    print('Training model')
    for epoch in range(num_epochs):
        try:
            print('-' * 14)
            # print(f'Epoch: {epoch+1}/{num_epochs}')
            print('Epoch: {}/{}'.format(epoch+1, num_epochs))


            avgloss = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
            train_losses.append(avgloss)

            if scheduler is not None: # TODO: Check this
                scheduler.step()

            avgloss = evaluate_model(model, dataloaders['val'], criterion, device)
            validation_lossed.append(avgloss)
            
            # Do more data analysis here?
        
        
        except KeyboardInterrupt: break
        
    
    print('-' * 14)
    return np.array(train_losses), np.array(validation_lossed)

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

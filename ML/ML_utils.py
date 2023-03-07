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
    
def get_labels(data, keys, device):
    labels = []
    for key in keys:
        labels.append(torch.from_numpy(np.array(data[key], dtype = np.float32)))
    labels = torch.stack(labels, 1).to(device) 
    
    # Ff = torch.from_numpy(np.array(data['Ff_mean'], dtype = np.float32))
    # rupture_stretch = torch.from_numpy(np.array(data['rupture_stretch'], dtype = np.float32)) # When mergening float32 with int32 I believe it stores both as float32 anyway...
    # is_ruptured = torch.from_numpy(np.array(data['is_ruptured'], dtype = np.int32))
    # labels = torch.stack((Ff, rupture_stretch, is_ruptured), 1).to(device) 
    return labels


def save_training_history(name, train_val_hist, ML_setting, precision = 4):
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
            if key == 'epoch':
                outfile.write(f'{data:d} ')
            else:
                outfile.write(f'{data:0.{precision}e} ')
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


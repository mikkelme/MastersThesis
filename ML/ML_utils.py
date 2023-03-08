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
    return labels


def save_training_history(name, history, ML_setting, precision = 4):
    filename = name + '_training_history.txt'
    outfile = open(filename, 'w')
    
    for key in ML_setting:
        outfile.write(f'# {key} = {ML_setting[key]}\n')
    
    epochs = history['epoch']
    
    keys = ''
    for key in history:
        keys += f'{key} '
    keys += '\n' # add linebreak
    outfile.write(keys)
    
    for i, epoch in enumerate(epochs):
        for key in history:
            data = history[key][i]
            if key == 'epoch':
                outfile.write(f'{data:d} ')
            else:
                outfile.write(f'{data:0.{precision}e} ')
        outfile.write('\n')
    outfile.close()


def save_best_model_scores(name, best, history, ML_setting,  precision = 4):
    filename = name + '_best_scores.txt'
    
    best_epoch = best['epoch']
    outfile = open(filename, 'w')
    for key in ML_setting:
        outfile.write(f'# {key} = {ML_setting[key]}\n')
        
    for key in history:
        data = history[key][best_epoch]
        if key == 'epoch':
            outfile.write(f'{key} {data:d}\n')
        else:
            outfile.write(f'{key} {data:0.{precision}e}\n')
        
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


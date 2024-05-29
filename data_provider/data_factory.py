from data_provider import data_loader 
from torch.utils.data import DataLoader

data_dict = {
    'PEMS08': data_loader.Dataset_PEMS,
    'PEMS04': data_loader.Dataset_PEMS,
    'ca':data_loader.Dataset_CA,
    'temperature2016':data_loader.Dataset_npz,
    'humidity2016':data_loader.Dataset_npz,
    'SDWPF': data_loader.Dataset_npz
}

start_time_dict = {
    'humidity2016':'2016-01-01 00:00:00',
    'temperature2016':'2016-01-01 00:00:00',
    'PEMS04': '2018-01-01 00:00:00',
    'PEMS08': '2016-07-01 00:00:00',
    'ca':'2019-01-01 00:00:00',
    'SDWPF':'2018-01-01 00:00:00' # fake year
}


def data_provider(args, flag, forDownstream=False):
    Data = data_dict[args.dataset_name]
    start_time_str = start_time_dict[args.dataset_name]
    
    timeenc = 2

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq

    size = [args.input_length, args.label_len, args.predict_length]
    if forDownstream:
        size =[args.ds_input_length, 0, args.ds_predict_length]
        print('for Downstream:')

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=size,
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        start_time_str=start_time_str
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

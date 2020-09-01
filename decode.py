from utils.decoder import Decoder
import time
import argparse
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model_path')
    parser.add_argument('--save_modelinfo_path')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--inputpath')
    parser.add_argument('--outputpath')
    parser.add_argument('--modeltype', choices=['bilstm-crf', 'bilstm-partial-crf', 'bert-finetune', 'bert-featurebased'])

    args = parser.parse_args()

    if args.gpu >= 0:
        use_cuda = True
        torch.cuda.set_device(args.gpu)
    else:
        use_cuda = False

    decoder = Decoder(
        save_model_path=args.save_model_path,
        batch_size=args.batch_size,
        if_decode_on_gpu=use_cuda,
        data_path=args.save_modelinfo_path,
        model_type=args.modeltype
    )
    start_time = time.time()
    print("Decode  Input:",args.inputpath)
    print("Decode Output:", args.outputpath)
    print("Using     GPU:", args.gpu)
    decoder.decode_file_writeoutput(args.inputpath, args.outputpath)
    end_time = time.time()
    print("Decode complete, time: %.2fs"%(end_time-start_time))

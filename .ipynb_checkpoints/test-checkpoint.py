import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim

from utils.utils import Monitor, get_dataset, get_network, print_args, save_args, load_checkpoint, save_checkpoint, \
    get_dex_ycb_dataset
from utils.epoch import single_epoch
from utils.options import add_opts
import torch.backends.cudnn as cudnn


def main(args):
    # Initialize randoms seeds
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)
    cudnn.benchmark = True

    # create exp result dir
    os.makedirs(args.host_folder, exist_ok=True)
    # Initialize model
    model = get_network(args)

    if args.use_cuda and torch.cuda.is_available():
        print("Using {} GPUs !".format(torch.cuda.device_count()))
        model.cuda()

    start_epoch = 0
    if not args.evaluate:
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(model_params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, gamma=args.lr_decay_gamma)
        if args.use_ho3d:
            train_dat = get_dataset(args, mode="train")
        else:
            train_dat = get_dex_ycb_dataset(args, mode="train")
        print("training dataset size: {}".format(len(train_dat)))
        train_loader = torch.utils.data.DataLoader(train_dat, batch_size=args.train_batch, shuffle=True,
                                                   num_workers=int(args.workers), pin_memory=True, drop_last=False)
        monitor = Monitor(hosting_folder=args.host_folder)

    else:
        assert args.resume is not None, "need trained model for evaluation"
        device = torch.device('cuda') if torch.cuda.is_available() and args.use_cuda else torch.device('cpu')
        load_checkpoint(model, resume_path=args.resume, strict=False, device=device)
        args.epochs = start_epoch + 1

    # print(args.epochs)

    # Initialize validation dataset
    if args.use_ho3d:
        val_dat = get_dataset(args, mode="evaluation")
    else:
        val_dat = get_dex_ycb_dataset(args, mode="evaluation")
    print("evaluation dataset size: {}".format(len(val_dat)))
    val_loader = torch.utils.data.DataLoader(val_dat, batch_size=args.test_batch,
                                             shuffle=False, num_workers=int(args.workers),
                                             pin_memory=True, drop_last=False)

    for epoch in range(start_epoch, args.epochs):
        train_dict = {}
        if not args.evaluate:
            print("Using lr {}".format(optimizer.param_groups[0]["lr"]))
            train_avg_meters = single_epoch(
                loader=train_loader, model=model, optimizer=optimizer,
                epoch=epoch, save_path=args.host_folder, train=True,
                save_results=False, use_cuda=True)

            train_dict = {meter_name: meter.avg
                          for (meter_name, meter) in train_avg_meters.average_meters.items()}
            monitor.log_train(epoch + 1, train_dict)

        # Evaluate on validation set
        if args.evaluate or (epoch + 1) % args.test_freq == 0:
            with torch.no_grad():
                single_epoch(loader=val_loader, model=model, epoch=epoch if not args.evaluate else None,
                             optimizer=None, save_path=args.host_folder,
                             train=False, save_results=args.save_results, use_cuda=args.use_cuda,
                             indices_order=val_dat.jointsMapSimpleToMano if hasattr(val_dat,
                                                                                    "jointsMapSimpleToMano") else None)

        if not args.evaluate:
            if (epoch + 1) % args.snapshot == 0 or (epoch + 1) % 5 == 0:
                print(f"save epoch {epoch + 1} checkpoint to {args.host_folder}")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "network": args.network,
                        "state_dict": model.state_dict(),
                    },
                    checkpoint=args.host_folder, filename=f"checkpoint_{epoch + 1}.pth.tar")

            if args.lr_decay_gamma:
                if args.lr_decay_step is None:
                    scheduler.step(train_dict["mano_joints3d_loss"])
                else:
                    if epoch <= 35 or args.use_ho3d:
                        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand-Object training")
    add_opts(parser)

    args = parser.parse_args()
    # 手动赋值所有命令行参数
    args.HO3D_root = "/root/autodl-tmp/HFL-Net-main-cyt/data/HO3D/data"  # --HO3D_root
    args.host_folder = "host_folder/ho3d"  # --host_folder
    args.dex_ycb_root = "/data1/zhifeng/dex-ycb"  # --dex_ycb_root
    args.epochs = 40  # --epochs
    args.inp_res = 256  # --inp_res
    args.lr = 1e-4  # --lr
    args.train_batch = 64  # --train_batch
    args.mano_lambda_regulshape = 0  # --mano_lambda_regulshape
    args.mano_lambda_regulpose = 0  # --mano_lambda_regulpose
    args.lr_decay_gamma = 0.7  # --lr_decay_gamma
    args.lr_decay_step = 10  # --lr_decay_step
    args.test_batch = 64  # --test_batch
    args.use_ho3d = True  # --use_ho3d（布尔值参数，存在即为True）
    args.evaluate = True
    args.resume = "/root/autodl-tmp/HFL-Net-main/host_folder/ho3d/checkpoint_50.pth.tar"

    if args.use_ho3d:
        args.test_freq = 10
        # args.test_freq = 1
        args.save_results = True
        args.snapshot = 10
        # args.snapshot = 1
    else:
        args.test_freq = 5
        args.save_results = False
        args.snapshot = 2

    print_args(args)
    save_args(args, save_folder=args.host_folder, opt_prefix="option")
    main(args)
    print("All done !")

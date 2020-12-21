from torch.utils.data import DataLoader
from utils.utils import DatasetBuilder, DrivingStateDataset
from utils.functions import *
from models.model import Load_Overall_Models
from models.solver import generator_step, discriminator_step, evaluator_step

def main():

    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument('--input_dim', type=int, default=2, help='')
    parser.add_argument('--onehot_dim', type=int, default=10, help='dim. of driving action space')
    parser.add_argument('--intend_dim', type=int, default=32, help='embedding dim. for onehot vector')
    parser.add_argument('--noise_dim', type=int, default=32, help='')

    # CNN
    parser.add_argument('--resnet_model', type=int, default=50, help='')
    parser.add_argument('--img_size_w', type=int, default=640, help='')
    parser.add_argument('--img_size_h', type=int, default=320, help='')
    parser.add_argument('--conv_flat_size', type=int, default=2048, help='size of ch. dim. of ResNet\' output')
    parser.add_argument('--pre_cnn_exp', type=int, default=12, help='')
    parser.add_argument('--pre_cnn_model', type=int, default=170, help='')

    # Decoder
    parser.add_argument('--emb_dim_dec', type=int, default=128, help='')
    parser.add_argument('--hid_dim_dec', type=int, default=128, help='')
    parser.add_argument('--num_layers_dec', type=int, default=1, help='')


    # Discriminator
    parser.add_argument('--emb_dim_dis', type=int, default=128, help='')
    parser.add_argument('--hid_dim_dis', type=int, default=128, help='')
    parser.add_argument('--emb_dim_dis_drvact', type=int, default=32, help='')
    parser.add_argument('--hid_dim_dis_drvact', type=int, default=32, help='')
    parser.add_argument('--num_layers_dis', type=int, default=1, help='')


    # Dataset
    parser.add_argument('--dataset_path', type=str, default='/home/dooseop/DATASET/', help='driving sequences should be placed in here')
    parser.add_argument('--exp_id', type=int, default=0, help='')
    parser.add_argument('--data_load_step', type=int, default=1, help='')
    parser.add_argument('--obs_seq_len', type=int, default=10, help='the number of previous frames')
    parser.add_argument('--path_len', type=float, default=20, help='a length of a path')
    parser.add_argument('--num_pos', type=float, default=20, help='a path of length \'path_len\' is defined with \'num_pos\' positions')
    parser.add_argument('--data_qual', type=float, default=0.05, help='data frame with GPS localization error less than \'data_qual\' will be used')
    parser.add_argument('--min_car_speed', type=float, default=5, help='data frame with car speed larger than \'min_car_speed\' will be used')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='\'val_ratio\' x 100 % of training data will be used for validation')

    # Training
    parser.add_argument('--model_dir', type=str, default='saved_models/model')
    parser.add_argument('--load_pretrained', type=int, default=0, help='1: pre-trained network will be load, 0: scratch')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--n_critic', type=int, default=2, help='')
    parser.add_argument('--best_k', type=int, default=20, help='the number of paths to be generated during training')

    parser.add_argument('--beta1', type=float, default=0.5, help='')
    parser.add_argument('--grad_clip', type=float, default=1.5, help='')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='')
    parser.add_argument('--learning_rate_cnn', type=float, default=0.00005, help='')
    parser.add_argument('--l2_param', type=float, default=0.0000, help='')
    parser.add_argument('--alpha', type=float, default=0.01, help='weight for bce loss (driving intention)')
    parser.add_argument('--kappa', type=float, default=100.0, help='weight for variety loss')
    parser.add_argument('--theta', type=float, default=0.005, help='weight for bce loss (sequential driving intention)')
    parser.add_argument('--beta', type=float, default=1.0, help='weight for D loss (sequential driving intention)')
    parser.add_argument('--drop_prob_gen', type=float, default=0.0)
    parser.add_argument('--drop_prob_dis', type=float, default=0.0)

    parser.add_argument('--is_avg_bgt', type=int, default=1, help='1: apply image pre-processing, 0: original')
    parser.add_argument('--label_flip_prob', type=float, default=0.05, help='GT becomes generated one w.p. \'label_flip_prob\'')

    parser.add_argument('--max_num_chkpt', type=int, default=5, help='recent \'max_num_chkpt\' pts files will be stored')
    parser.add_argument('--save_every', type=int, default=5, help='validate every \'save_every\' epochs')
    parser.add_argument('--valid_step', type=int, default=2, help='')

    # pc setting
    parser.add_argument('--multi_gpu', type=int, default=1, help='')
    parser.add_argument('--num_cores', type=int, default=4, help='')

    args = parser.parse_args()

    # train starts
    train(args)

def train(args):

    # -------------------------------------------
    # Experimental settings,test
    # -------------------------------------------
    torch.backends.cudnn.benchmark = True

    # type definition
    long_dtype, float_dtype = get_dtypes(useGPU=True)

    # check if there is pre-trained network
    save_directory = args.model_dir + str(args.exp_id)
    if save_directory != '' and not os.path.exists(save_directory):
        os.makedirs(save_directory)
        args.load_pretrained = 0

    # load saved network parameters or save current parameters
    prev_ADE = 100000.0
    start_epoch = args.start_epoch
    if (args.load_pretrained == 1):

        # load config file
        with open(os.path.join(save_directory, 'config.pkl'), 'rb') as f:
            args = pickle.load(f)
            args.load_pretrained = 1

    else:
        with open(os.path.join(save_directory, 'config.pkl'), 'wb') as f:
            pickle.dump(args, f)


    # ------------------------------------------
    # Prepare data and Define DataLoader
    # ------------------------------------------
    builder = DatasetBuilder(args)
    drvsts_dataset = DrivingStateDataset(dataset_path=builder.dataset_path,
                                         dataset_names=builder.dataset_names,
                                         dataset=builder.train_data,
                                         obs_seq_len=builder.obs_seq_len,
                                         pred_seq_len=builder.num_pos,
                                         is_aug=args.is_avg_bgt,
                                         dtype=torch.FloatTensor)

    dataloader = DataLoader(drvsts_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_cores, drop_last=True)



    # ------------------------------------------
    # Define network
    # ------------------------------------------
    PathGen, PathDis, ResNet, opt_g, opt_d, opt_c = Load_Overall_Models(args, float_dtype)

    # For ResNet, pretrained parameters will be used
    if (args.load_pretrained == 0):
        pre_trained_cnn = './pretrained_cnn/saved_cnn_exp%d_model%d.pt' % (args.pre_cnn_exp, args.pre_cnn_model)
        chkpt_resnet = torch.load(pre_trained_cnn)
        ResNet.load_state_dict(chkpt_resnet['resnet_state_dict'])
        print('>> pre-trained cnn {%s} is loaded successfully...' % pre_trained_cnn)

    # load previously trained network
    else:
        ckp_idx = save_read_latest_checkpoint_num(os.path.join(save_directory), 0, isSave=False)
        file_name = save_directory + '/saved_chk_point_%d.pt' % ckp_idx
        checkpoint = torch.load(file_name)
        PathGen.load_state_dict(checkpoint['pathgen_state_dict'])
        PathDis.load_state_dict(checkpoint['pathdis_state_dict'])
        ResNet.load_state_dict(checkpoint['resnet_state_dict'])
        opt_g.load_state_dict(checkpoint['opt_g'])
        opt_d.load_state_dict(checkpoint['opt_d'])
        opt_c.load_state_dict(checkpoint['opt_c'])
        prev_ADE = checkpoint['ADE']
        print('>> trained parameters are loaded from {%s} / {prev_ADE:%.4f}' % (file_name, prev_ADE))


    # ------------------------------------------
    # Training and Evaluation
    # ------------------------------------------
    num_batches = int(builder.num_train_scenes / args.batch_size)
    for e in range(start_epoch, args.num_epochs):

        # ------------------------------------------
        # Training
        # ------------------------------------------

        # turn on train mode
        PathGen.train()
        PathDis.train()
        ResNet.train()

        # performance measure
        g_mse_b, g_fake_b, g_class_b = 0, 0, 0
        d_real_b, d_fake_b, d_class_b = 0, 0, 0

        start = time.time()
        for b, data in enumerate(dataloader):

            start_batch = time.time()

            # ------------------------------------------
            # data load and conversion
            # ------------------------------------------
            '''
            xi : first element of past traj
            xo : displacement between positions in past traj (or speed)
            xp : displacement between positions in future path
            alt : altitudes corresponding to {past traj; future path}
            dp : onehot vector for input driving intention
            dp_fake : a fake version of dp (for examples, an input driving intention is randomly selected)
            pm/pminv : transformation matrix and its inverse
            dp_seq : sequence of onehot vectors for sequential driving intentions
            '''
            xi, xo, xp, alt, dp, dp_fake, imgs, pm, pminv, dp_seq = data

            xo_cuda = xo.permute(1, 0, 2).cuda()
            xp_cuda = xp.permute(1, 0, 2).cuda()
            dp_cuda = torch.squeeze(dp).cuda()
            dp_fake_cuda = torch.squeeze(dp_fake).cuda()
            imgs_cuda = imgs.cuda()
            dp_seq_cuda = dp_seq.permute(1, 0, 2).contiguous().cuda()


            # ------------------------------------------
            # conv operation
            # ------------------------------------------
            conv_features = ResNet(imgs_cuda)


            # ------------------------------------------
            # training generator and discriminator
            # ------------------------------------------
            if (b % args.n_critic == 0):
                g_mse, g_fake, g_class = generator_step(args, PathGen, PathDis, opt_g, opt_c, xo_cuda, xp_cuda, dp_cuda, conv_features, dp_seq_cuda)
                g_mse_b += g_mse
                g_fake_b += g_fake
                g_class_b += g_class

            # train discriminator
            d_real, d_fake, d_class = discriminator_step(args, PathGen, PathDis, opt_d, opt_c, xo_cuda, xp_cuda, dp_cuda, conv_features, dp_seq_cuda)
            d_real_b += d_real
            d_fake_b += d_fake
            d_class_b += d_class

            end_batch = time.time()

            # print ---------------------------------
            print_current_progress(e, b, num_batches, (end_batch-start_batch)/args.batch_size)

        end = time.time()
        time_left = (end - start) * (args.num_epochs - e - 1) / 3600.0
        print('[Epoch %d, %.2f hrs left] G (mse %.4f, score %.4f, class %.4f) / D (score_real %.4f, score_fake %.4f, class %.4f)' %
              (e, time_left, g_mse_b, g_fake_b, g_class_b, d_real_b, d_fake_b, d_class_b))

        # ------------------------------------------
        # Evaluation
        # ------------------------------------------

        if (e % int(args.save_every) == 0):


            # evaluate on test dataset
            ADE, FDE = evaluator_step(PathGen, ResNet, builder, float_dtype, args, e)
            print(">> evaluation results are created .. {ADE:%.4f, FDE:%.4f}" % (ADE, FDE))

            # save trained model
            if (ADE < prev_ADE):
                prev_ADE = ADE
                rt = save_read_latest_checkpoint_num(os.path.join(save_directory), e, isSave=True)
                file_name = save_directory + '/saved_chk_point_%d.pt' % e
                check_point = {
                    'epoch' : e,
                    'pathgen_state_dict': PathGen.state_dict(),
                    'pathdis_state_dict': PathDis.state_dict(),
                    'resnet_state_dict': ResNet.state_dict(),
                    'opt_g': opt_g.state_dict(),
                    'opt_d': opt_d.state_dict(),
                    'opt_c': opt_c.state_dict(),
                    'ADE': ADE,
                    'FDE': FDE}
                torch.save(check_point, file_name)
                print(">> current network is saved ...")
                remove_past_checkpoint(os.path.join('./', save_directory), max_num=args.max_num_chkpt)


if __name__ == '__main__':
    main()

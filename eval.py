from sklearn.neighbors import KernelDensity
from utils.utils import DatasetBuilder
from utils.functions import *
from models.model import PathGenerator, ConvNet
from steer_model.steer_model import Encoder

torch.backends.cudnn.benchmark = True

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=2101)
    parser.add_argument('--gpu_num', type=int, default=0)
    args = parser.parse_args()
    test(args)


def test(args):

    # assign gpu device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

    # path to saved network
    path = 'saved_models/model' + str(args.exp_id)

    # load parameter setting
    with open(os.path.join(path, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    # generator define
    long_dtype, float_dtype = get_dtypes(useGPU=True)
    PathGen = PathGenerator(input_dim=saved_args.input_dim,
                            obs_len=saved_args.obs_seq_len,
                            pred_len=saved_args.num_pos,
                            emb_dim_dec=saved_args.emb_dim_dec,
                            hid_dim_dec=saved_args.hid_dim_dec,
                            num_layers_dec=saved_args.num_layers_dec,
                            onehot_dim=saved_args.onehot_dim,
                            intend_dim=saved_args.intend_dim,
                            conv_flat_size=saved_args.conv_flat_size,
                            noise_dim=saved_args.noise_dim,
                            dropout=saved_args.drop_prob_gen)
    PathGen.type(float_dtype).eval()

    # convnet define
    ResNet = ConvNet(use_pretrained=True,
                     feature_extract=True,
                     resent_model=saved_args.resnet_model)
    if (saved_args.multi_gpu == 1):
        ResNet = nn.DataParallel(ResNet)
    ResNet.type(float_dtype).eval()

    # load pretrained network parameters
    ckp_id = save_read_latest_checkpoint_num(os.path.join('./', path), 0, isSave=False)
    file_path = path + '/saved_chk_point_%d.pt' % ckp_id
    checkpoint = torch.load(file_path)
    PathGen.load_state_dict(checkpoint['pathgen_state_dict'])
    ResNet.load_state_dict(checkpoint['resnet_state_dict'])
    print('>> trained parameters are loaded from {%s}, min ADE {%.4f}' % (file_path, checkpoint['ADE']))

    # steer generator
    SteerGen = Encoder(input_dim=2,
                       embedding_dim=32,
                       h_dim=32,
                       num_layers=1,
                       dropout=0.0)
    SteerGen.type(float_dtype).eval()
    checkpoint = torch.load('./steer_model/saved_chk_point_80.pt')
    SteerGen.load_state_dict(checkpoint['steergen_state_dict'])


    # evaluation setting
    saved_args.dataset_path = '/home/dooseop/DATASET' ##### change path to dataset folder ####
    saved_args.batch_size = 1
    pred_length = saved_args.num_pos
    obs_length = saved_args.obs_seq_len
    saved_args.best_k = 20

    # load test data
    data_loader = DatasetBuilder(saved_args)

    # empty lists
    ADE_best, FDE_best, log_probs, diversity, MSE = [], [], [], [], []

    # for all test samples, do
    for b in range(0, data_loader.num_test_scenes, 1):

        # read data
        _, xo_batch, _, _, dp_batch, img_batch, _, path3d_batch, _, _, steer_batch = \
            data_loader.next_batch_eval([b], mode='test')

        xo = xo_batch[0]
        dp = dp_batch[0]
        img = img_batch[0]
        path3d = path3d_batch[0]
        steer = steer_batch[0]


        # conversion to tensor
        xo_tensor = from_list_to_tensor_to_cuda(xo, float_dtype)
        imgs_tensor = torch.from_numpy(np.array([img])).permute(0, 3, 1, 2).type(float_dtype).cuda()
        dp_onehot, _ = drvint_onehot_batch([dp], float_dtype)


        # path generation
        start = time.time()
        overall_gen_offsets, _, _ = PathGen(xo_tensor, dp_onehot, ResNet(imgs_tensor), saved_args.best_k)
        end = time.time()


        # reconstruct paths
        all_gen_paths = []
        err_values = np.zeros(shape=(saved_args.best_k))
        for z in range(saved_args.best_k):
            gen_offsets = np.squeeze(overall_gen_offsets[z].detach().to('cpu').numpy())
            gen_recon = np.cumsum(gen_offsets, axis=0)
            all_gen_paths.append(gen_recon)
            err_values[z] = np.sum(abs(gen_recon - path3d[obs_length:, 0:2]))


        # cal ADE & FDE
        min_idx = np.argmin(err_values)
        err_vector = path3d[obs_length:, 0:2] - all_gen_paths[min_idx]
        displacement_error = np.sqrt(np.sum(err_vector ** 2, axis=1))
        ADE_best.append(displacement_error[:])
        FDE_best.append(displacement_error[-1])

        # cal MSE-S
        xp_est_tensor = overall_gen_offsets[min_idx]
        est_steer = SteerGen(xo_tensor, xp_est_tensor).detach().to('cpu').numpy()
        MSE.append((steer - est_steer[0])**2)


        # calc diversity
        for z in range(saved_args.best_k):
            cur_gen_path = all_gen_paths[z]
            for zz in range(z, saved_args.best_k):
                if (z != zz):
                    target_gen_path = all_gen_paths[zz]

                    error = (cur_gen_path - target_gen_path)
                    dist = np.sqrt(np.sum(error**2, axis=1))
                    diversity.append(dist)


        # calc marginal log likelihood
        all_gen_paths = np.squeeze(np.array(all_gen_paths))
        for t in range(pred_length):
            gen_data = all_gen_paths[:, t, :]  # (n_samples, n_features)
            kde = KernelDensity()
            kde.fit(gen_data)

            gt_data = path3d[obs_length + t, 0:2].reshape(1, 2)
            log_probs.append(kde.score_samples(gt_data))


        # current status
        print_current_progress(b, data_loader.num_test_scenes, (end-start))


    print('-------------Exp %d / Epoch %d ------------------' % (args.exp_id, ckp_id))
    print('ADE_best : %.4f, FDE_best : %.4f' % (np.mean(ADE_best), np.mean(FDE_best)))
    print('Diversity : %.4f' % (np.mean(diversity)))
    print('Marginal log prob : %.4f' % (np.mean(log_probs)))
    print('MSE : %.4f' % np.mean(MSE))


def print_current_progress(b, num_batchs, time_spent):

    if b == num_batchs-1:
        sys.stdout.write('\r')
    else:
        time_left = ((num_batchs - b) * time_spent) / 60.0 # minutes
        sys.stdout.write('\r %d / %d (%.4f sec/sample, %.2f min. left)' % (b, num_batchs, time_spent, time_left)),

    sys.stdout.flush()


def calculate_error_vector(true_path, gen_path):

    error_path = true_path - gen_path

    return error_path


def from_list_to_tensor_to_cuda(x, dtype):

    '''
    x : a list of (seq_len x input_dim)
    '''

    # batch_size x seq_len x input_dim
    x = np.array(x)
    if (len(x.shape) == 2):
        x = np.expand_dims(x, axis=0)

    y = torch.from_numpy(x).permute(1, 0, 2)

    return y.type(dtype).cuda()

if __name__ == '__main__':
    main()

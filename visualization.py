from utils.utils import DatasetBuilder
from utils.functions import *
from models.model import PathGenerator, ConvNet
from skimage import transform

torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=2101)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--model_num', type=int, default=95)
    parser.add_argument('--best_k', type=int, default=100)
    parser.add_argument('--t_step', type=int, default=10)
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
    print('>> exp id %d setting is loaded..' % args.exp_id)

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

    # load trained weights
    ckp_idx = int(args.model_num)
    file_path = path + '/saved_chk_point_%d.pt' % ckp_idx
    checkpoint = torch.load(file_path)
    PathGen.load_state_dict(checkpoint['pathgen_state_dict'])
    ResNet.load_state_dict(checkpoint['resnet_state_dict'])
    print('>> trained parameters are loaded from {%s}, min ADE {%.4f}' % (file_path, checkpoint['ADE']))


    # evaluation setting
    saved_args.dataset_path = '/home/dooseop/DATASET'  ##### change path to dataset folder ####
    saved_args.batch_size = 1
    obs_length = saved_args.obs_seq_len
    seq_length = obs_length + saved_args.num_pos


    # load test data
    data_loader = DatasetBuilder(saved_args)

    # create folder
    folder_path = './capture'
    if (os.path.exists(folder_path) == False):
        os.mkdir(folder_path)


    # for all test samples, do
    for b in range(0, data_loader.num_test_scenes, args.t_step):

        xi_batch, xo_batch, _, _, dp_batch, img_batch, did_batch, path3d_batch, _, img_ori_batch, _ = \
            data_loader.next_batch_eval([b], mode='test')

        xi = xi_batch[0]
        xo = xo_batch[0]
        dp = dp_batch[0]
        did = did_batch[0]
        img = img_batch[0]
        img_ori = img_ori_batch[0]
        path3d = path3d_batch[0]
        repr_drvint = int(find_representative_drvint(dp))


        # conversion to tensor
        xo_tensor = from_list_to_tensor_to_cuda(xo, float_dtype)
        imgs_tensor = torch.from_numpy(np.array([img])).permute(0, 3, 1, 2).type(float_dtype).cuda()
        dp_onehot, _ = drvint_onehot_batch([dp], float_dtype)


        # conv operation
        conv_img = ResNet(imgs_tensor)


        # path generation with true label
        path3d_recon_k, weight_maps, best_k_idx = path_generation(PathGen, xi, xo, path3d, xo_tensor, dp_onehot, conv_img, args.best_k)


        # ----------------------------------------
        # visualization
        # ----------------------------------------
        fig, ax = plt.subplots()
        ax.imshow(img_ori, extent=[0, 1280, 0, 360])

        K = data_loader.dataset_calis[int(did)][0]
        Rt = data_loader.dataset_calis[int(did)][1]

        weights = path_weight_calc(img_ori, path3d_recon_k, K, Rt, obs_length, seq_length)

        # draw generated paths
        for z in range(args.best_k):
            weight = weights[z]
            color_b = weight * 1 + (1 - weight) * (255 / 255)
            color_g = weight * 0 + (1 - weight) * (0 / 255)
            color_r = weight * 0 + (1 - weight) * (180 / 255)
            ax = draw_path_on_figure(path3d_recon_k[z], ax, K, Rt, obs_length, seq_length, 3.0, (color_b, color_g, color_r))

        # draw GT path
        ax = draw_path_on_figure(path3d, ax, K, Rt, obs_length, seq_length, 3.0, (0, 0, 1), alpha=1)

        # fig to np array
        final_img = fig_to_nparray(fig, ax)
        final_img = cv2.resize(final_img, (640, 320), interpolation=cv2.INTER_CUBIC)

        file_name = 'Exp%d_Scene%04d_Drvint(%s)_K%d.png' % (args.exp_id, b, return_name(repr_drvint-1), args.best_k)
        cv2.imwrite(os.path.join(folder_path, file_name), final_img.astype('uint8'))
        print(" >> {%s} is created .." % file_name)


def path_weight_calc(img, path3d_recon_k, K, Rt, obs_length, seq_length):

    num_path = len(path3d_recon_k)

    scale = 20.0

    x_min = np.min(np.stack(path3d_recon_k)[:, obs_length-1:, 1])
    x_max = np.max(np.stack(path3d_recon_k)[:, obs_length-1:, 1])

    y_min = np.min(np.stack(path3d_recon_k)[:, obs_length-1:, 0])
    y_max = np.max(np.stack(path3d_recon_k)[:, obs_length-1:, 0])

    img_w = int(scale * (x_max - x_min + 1))
    img_h = int(scale * (y_max - y_min + 1))

    zero_img = np.zeros(shape=(img_h, img_w)).astype('uint8')
    zero_imgs = [np.copy(zero_img) for _ in range(num_path)]
    for z in range(num_path):
        cur_path = path3d_recon_k[z][obs_length-1:, 0:2]
        cur_path_y = cur_path[:, 0]
        cur_path_x = cur_path[:, 1]

        cur_pel_h = scale * (y_max - cur_path_y)
        cur_pel_w = scale * (cur_path_x - x_min)

        for j in range(1, 21):

            x_cur = int(cur_pel_w[j])
            x_prev = int(cur_pel_w[j-1])

            y_cur = int(cur_pel_h[j])
            y_prev = int(cur_pel_h[j-1])

            zero_imgs[z] = cv2.line(zero_imgs[z], (x_cur, y_cur), (x_prev, y_prev), 255, 1)

        sum_path_imgs = np.mean(np.stack(zero_imgs), axis=0)
        sum_path_imgs_blur = cv2.blur(sum_path_imgs, (5, 5))

        sum_path_imgs_blur = sum_path_imgs_blur / np.max(sum_path_imgs_blur)

    # cv2.imshow('test', (255*sum_path_imgs_blur).astype('uint8'))
    # cv2.waitKey(0)

    path3d_weight = []
    for z in range(num_path):

        cur_path = path3d_recon_k[z][obs_length-1:, 0:2]
        cur_path_y = cur_path[:, 0]
        cur_path_x = cur_path[:, 1]

        cur_pel_h = scale * (y_max - cur_path_y)
        cur_pel_w = scale * (cur_path_x - x_min)

        cnt = 0
        weight_sum = 0
        for j in range(1, 21):

            x_cur = int(cur_pel_w[j])
            y_cur = int(cur_pel_h[j])

            weight = float(sum_path_imgs_blur[y_cur, x_cur]) * 10
            if (weight < 0):
                continue

            cnt += 1
            weight_sum += weight
        path3d_weight.append(weight_sum / cnt)

    weights_norm = np.array(path3d_weight) - np.min(np.array(path3d_weight))
    weights_norm /= np.max(weights_norm)

    return (weights_norm) ** 1.5

def fig_to_nparray(fig, ax):

    dpi = 80
    fig.set_size_inches(1280 / dpi, 360 / dpi)
    ax.set_axis_off()

    fig.canvas.draw()
    render_fig = np.array(fig.canvas.renderer._renderer)

    final_img = np.zeros_like(render_fig[:, :, :3]) # 450, 1600
    final_img[:, :, 0] = render_fig[:, :, 0]
    final_img[:, :, 1] = render_fig[:, :, 1]
    final_img[:, :, 2] = render_fig[:, :, 2]

    margin_h = 60
    margin_w = 210
    # return final_img[margin_h:450-margin_h, margin_w:1600-margin_w, :]
    return final_img[margin_h:450-margin_h, margin_w:1600-170, :]

def draw_path_on_figure(path3d, ax, K, Rt, obs_length, seq_length, linewidth, color, alpha=0.2):

    xg, yg = ego_to_cam(path3d, K, Rt, obs_length, seq_length)
    ax.plot(xg, 360 - yg, '-', linewidth=linewidth, color=color, alpha=alpha)

    return ax

def path_generation(PathGen, xi, xo, path3d, xo_cuda, dp_cuda, conv_img, best_k):


    overall_gen_offsets, overall_drvsts, weight_maps = PathGen(xo_cuda, dp_cuda, conv_img, best_k)

    xo[0, :] = xi
    path3d_recon_k = []
    errors = []
    for z in range(best_k):
        path3d_recon_copy = np.copy(path3d)
        gen_offsets = np.squeeze(overall_gen_offsets[z].detach().to('cpu').numpy())
        gen_recon = np.concatenate([xo, gen_offsets], axis=0)
        gen_recon = np.cumsum(gen_recon, axis=0)
        path3d_recon_copy[:, 0:2] = gen_recon
        path3d_recon_k.append(path3d_recon_copy)

        err = np.mean((path3d[:, 0:2] - path3d_recon_copy[:, 0:2]) ** 2)
        errors.append(err)

    min_mse_idx = np.argmin(np.array(errors))
    # min_mse_idx = np.argmax(np.array(errors))
    weight_maps_np = torch.squeeze(torch.stack(weight_maps)).detach().to('cpu').numpy()  # best_k x seq_len x 10 x 20

    return path3d_recon_k, weight_maps_np, min_mse_idx

def ego_to_cam(path3d, K, Rt, obs_length, seq_length):

    x = []
    y = []
    for m in range(obs_length - 1, seq_length):
        A = np.matmul(np.linalg.inv(Rt), path3d[m, :].reshape(4, 1))
        B = np.matmul(K, A)

        x_cur = (B[0, 0] / B[2, 0])
        y_cur = (B[1, 0] / B[2, 0])

        if (x_cur < 0 or x_cur > 1280 - 1 or y_cur < -1 or y_cur > 360 - 1):
            continue

        x.append(x_cur)
        y.append(y_cur)

    return np.array(x), np.array(y)

def read_csv(file_dir):
    return np.genfromtxt(file_dir, delimiter=',')

def return_name(max_idx):

    if (max_idx == 0):
        name = 'GO'

    elif (max_idx == 1):
        # orange : turn left
        name = 'TL'

    elif (max_idx == 2):
        # yellow : turn right
        name = 'TR'

    elif (max_idx == 3):
        # green : u-turn
        name = 'UT'

    elif (max_idx == 4):
        # blue : left lc
        name = 'LL'

    elif (max_idx == 5):
        # purple : right lc
        name = 'RL'

    elif (max_idx == 6):
        # cyan : avoidance
        name = 'AV'

    elif (max_idx == 7):
        # pink : left way
        name = 'LW'

    elif (max_idx == 8):
        # white : right way
        name = 'RW'

    return name

if __name__ == '__main__':
    main()

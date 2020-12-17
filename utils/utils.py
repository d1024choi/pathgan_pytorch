from utils.functions import *
from torch.utils.data import Dataset

class DatasetBuilder:

    def __init__(self, args):

        # basic settings
        self.dataset_path = args.dataset_path
        self.data_qual = args.data_qual
        self.val_ratio = args.val_ratio
        self.min_car_speed = args.min_car_speed
        self.path_len = args.path_len
        self.obs_seq_len = args.obs_seq_len
        self.num_pos = args.num_pos
        self.grid_size = self.path_len / self.num_pos
        self.step_t = args.data_load_step
        self.is_avg_bgt = args.is_avg_bgt
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim

        # load data
        filename = './dataset/preprocessed_dataset.cpkl'
        self.load_preprocessed_data(filename)

    def load_preprocessed_data(self, filename):

        if not os.path.exists(filename):
            print('>> [Warning] there is no preprocessed data ..')
            self.make_preprocessed_data()

        f = open(filename, 'rb')
        raw_data = pickle.load(f)
        f.close()
        print('>> {%s} is loaded .. ' % filename)

        self.dataset_paths = raw_data[0]
        self.dataset_names = raw_data[1]
        self.dataset_types = raw_data[2]
        self.dataset_calis = raw_data[3]
        self.train_data = raw_data[4]
        self.valid_data = raw_data[5]
        self.test_data = raw_data[6]

        self.num_train_scenes = len(self.train_data)
        self.num_val_scenes = len(self.valid_data)
        self.num_test_scenes = len(self.test_data)

        print('>> the number of scenes (train %d/ valid %d/ test %d) ' % (self.num_train_scenes, self.num_val_scenes, self.num_test_scenes))

    def make_preprocessed_data(self):

        print('>> Making preprocessed data ..')

        '''
        frm idx (0)             gps_err (13)
        alt (1)                 drvint (14)
        roll (2)                
        pitch (3)               
        yaw (4)                  
        heading (5) 
        east (6) 
        north (7) 
        img_file_name (8) 
        lidar_file_name (9) 
        steer (10) 
        speed (11) 
        turn_signal (12) 
        '''
        # set seed
        np.random.seed(1)

        # folder path where data will be loaded
        dataset_path = []
        dataset_type = []
        dataset_name = []
        dataset_cali = []

        # -----------------------------------------
        # define dataset path
        # -----------------------------------------
        with open('./dataset/train.txt', "r") as myfile:
            folder_name = myfile.readlines()
        for i in range(len(folder_name)):
            dataset_path.append(os.path.join(self.dataset_path, str(folder_name[i][:-1])))
            dataset_type.append('train')
            dataset_name.append(str(folder_name[i][:-1]))

            # read calib info
            file_name = dataset_name[i] + '_cam0_extcalib.csv'
            file_path = os.path.join(os.path.join(dataset_path[i], 'label'), file_name)

            f = open(file_path)
            reader = csv.reader(f, delimiter=',')
            cnt = 0
            for row in reader:
                if (cnt == 0):
                    K = np.array(row).astype('float').reshape(3, 4)
                else:
                    Rt = np.array(row).astype('float').reshape(4, 4)
                cnt+=1
            f.close()
            dataset_cali.append([K, Rt])


        with open('./dataset/test.txt', "r") as myfile:
            folder_name = myfile.readlines()
        for i in range(len(folder_name)):
            dataset_path.append(os.path.join(self.dataset_path, str(folder_name[i][:-1])))
            dataset_type.append('test')
            dataset_name.append(str(folder_name[i][:-1]))

            # read calib info
            file_name = dataset_name[i] + '_cam0_extcalib.csv'
            file_path = os.path.join(os.path.join(dataset_path[i], 'label'), file_name)

            f = open(file_path)
            reader = csv.reader(f, delimiter=',')
            cnt = 0
            for row in reader:
                if (cnt == 0):
                    K = np.array(row).astype('float').reshape(3, 4)
                else:
                    Rt = np.array(row).astype('float').reshape(4, 4)
                cnt+=1
            f.close()
            dataset_cali.append([K, Rt])


        # -----------------------------------------
        # create and save path segments
        # -----------------------------------------
        num_false_path = 0
        pop = np.zeros(shape=(10, 1))
        seq_list_train, seq_list_val, seq_list_test = [], [], []
        for p in range(len(dataset_path)):

            # read current dataset
            file_name = dataset_name[p] + '_drvsts_pw.csv'
            file_path = os.path.join(os.path.join(dataset_path[p], 'label'), file_name)
            data = self.read_csv(file_path)
            print('[%d/%d] {%s} is under preprocessing..' % (p, len(dataset_path), file_name))

            # number of candidate sequences
            num_sequences = int(data.shape[0]-1)

            # for all possible sequences
            for i in range(self.obs_seq_len, num_sequences, self.step_t):

                # Find frm index self.path_len meters away from current position
                target_seq_len = 0
                for j in range(1, 10000):

                    if (i+j > num_sequences - 1):
                        break

                    dist = np.sqrt(np.sum((data[i, 6:8] - data[i+j, 6:8])**2))
                    if (dist > self.path_len + 2):
                        target_seq_len = j
                        break

                # If there is no such position, skip
                if (target_seq_len == 0):
                    # print('[frm %d] cant not interpolate' % (i))
                    continue


                '''NOTE : index "i" is observation'''

                # read current sequence data
                curr_seq_data = data[i:i + target_seq_len]
                past_seq_data = data[i-self.obs_seq_len+1:i]


                # --------------------------------------
                # check quality of current sequence
                # --------------------------------------
                if (True in np.array(curr_seq_data[:, 13] > self.data_qual).tolist()):
                    # print('[frm %d] quality is bad' % (i))
                    continue

                if (True in np.array(curr_seq_data[:, 14] < 1).tolist()):
                    # print('[frm %d] drvsts unavailable' % (i))
                    continue


                # --------------------------------------
                # valancing data distribution
                # --------------------------------------
                reject_prob = 0.95
                if (curr_seq_data[0, 11] < self.min_car_speed and np.random.rand(1) > (1-reject_prob)):
                    # print('[frm %d] car speed is too slow' % (i))
                    continue

                reject_prob = 0.125
                drvact_list_prev = past_seq_data[:, 14].tolist()
                drvact_list_prev.append(curr_seq_data[0, 14])

                drvact_list_curr = curr_seq_data[:, 14].tolist()
                if (True in (np.array(drvact_list_prev) > 1).tolist()):
                    # print('[frm %d] current go signal is right after the other actions.. so accepted' % (i))
                    do_nothing = 0
                elif (True in (np.array(drvact_list_curr) > 1).tolist() and np.random.rand(1) > 0.5):
                    # print('[frm %d] current go signal is before the other actions.. so accepted' % (i))
                    do_nothing = 0
                elif (np.random.rand(1) > reject_prob):
                    # print('[frm %d] current go signal is not accepted % (i))
                    continue



                # --------------------------------------
                # path sampling and interpolation
                # --------------------------------------

                # interpolate current traj data
                curr_seq_data = interpolate_traj_data(curr_seq_data, self.num_pos)
                target_seq_len = curr_seq_data.shape[0]

                # debug : count false trajs
                if (curr_seq_data.shape[0] < self.num_pos):
                    num_false_path += 1

                # pose matrix creation
                cur_obs = curr_seq_data[0, :]
                Pm = self.create_pose_matrix(cur_obs[1], cur_obs[2], cur_obs[3], cur_obs[4], cur_obs[6], cur_obs[7])

                # global to ego transform and interpolation
                past_traj_EnN = np.copy(past_seq_data[:, 6:8])
                past_traj_Alt = np.copy(past_seq_data[:, 1]).reshape(self.obs_seq_len-1, 1)
                past_traj_ENA = np.concatenate([past_traj_EnN, past_traj_Alt], axis=1)
                drvact_past = past_seq_data[:, 14].reshape(self.obs_seq_len -1, 1)

                cur_traj_EnN = np.copy(curr_seq_data[:, 6:8])
                cur_traj_Alt = np.copy(curr_seq_data[:, 1]).reshape(target_seq_len, 1)
                cur_traj_ENA = np.concatenate([cur_traj_EnN, cur_traj_Alt], axis=1)

                # create local path
                local_path = [cur_traj_ENA[0, :].reshape(1, 3)]
                drvact_intp = [curr_seq_data[0, 14]]
                img_num_intp = [curr_seq_data[0, 8]]
                Pm_intp = [Pm]

                past_index = 0
                for j in range(int(self.num_pos)):
                    cur_look_pos = local_path[j]

                    # find the closest point
                    for m in range(past_index, cur_traj_ENA.shape[0]):
                        target_look_pos = cur_traj_ENA[m, :].reshape(1, 3)
                        dist = np.sqrt(np.sum((cur_look_pos[0,:2] - target_look_pos[0,:2])**2))
                        if (dist > self.grid_size):
                            past_index = m
                            break

                    # linear interpolate position
                    look_pos_front = cur_traj_ENA[m, :].reshape(1, 3)
                    look_pos_end = cur_traj_ENA[m-1, :].reshape(1, 3)

                    dist_front = np.sqrt(np.sum((cur_look_pos[0,:2] - look_pos_front[0,:2])**2))
                    dist_end = np.sqrt(np.sum((cur_look_pos[0, :2] - look_pos_end[0, :2]) ** 2))

                    factor = (self.grid_size - dist_front) / (dist_end - dist_front)
                    intp_loos_pos = factor*look_pos_end + (1 - factor)*look_pos_front
                    local_path.append(intp_loos_pos)

                    # assign corresponding driving information
                    drvsts_front = curr_seq_data[m, 14]
                    drvsts_end = curr_seq_data[m-1, 14]

                    # assign corresponding img number
                    img_front = curr_seq_data[m, 8]
                    img_end = curr_seq_data[m-1, 8]

                    # assign corresponding pose matrix
                    front_obs = curr_seq_data[m, :]
                    end_obs = curr_seq_data[m - 1, :]
                    Pm_front = self.create_pose_matrix(front_obs[1], front_obs[2], front_obs[3], front_obs[4], front_obs[6], front_obs[7])
                    Pm_end = self.create_pose_matrix(end_obs[1], end_obs[2], end_obs[3], end_obs[4], end_obs[6], end_obs[7])

                    if (factor > 0.5):
                        drvact_intp.append(drvsts_end)
                        img_num_intp.append(img_end)
                        Pm_intp.append(Pm_end)
                    else:
                        drvact_intp.append(drvsts_front)
                        img_num_intp.append(img_front)
                        Pm_intp.append(Pm_front)

                # conversion to np array
                cur_path_ENA_intp = np.squeeze(np.array(local_path))
                drvact_intp = np.array(drvact_intp).reshape(int(self.num_pos+1), 1)

                # from global to ego centric path
                overall_path_ENA = np.concatenate([past_traj_ENA, cur_path_ENA_intp], axis=0)
                overall_drvact = np.concatenate([drvact_past, drvact_intp], axis=0)

                overall_path_homo = np.concatenate([overall_path_ENA, np.ones(shape=(overall_path_ENA.shape[0], 1))], axis=1)
                overall_path_ego = np.matmul(Pm, overall_path_homo.T).T

                # steering angle
                steer = curr_seq_data[0, 10]

                # dataset number
                did = p

                # debug : dataset characteristic calculation
                repr_drvint = find_representative_drvint(drvact_intp[1:])
                if (dataset_type[p] == 'train'):
                    pop[int(repr_drvint), 0] += 1

                # debug : check error
                point_dists = np.sqrt(np.sum((overall_path_ego[self.obs_seq_len:-1, :2] - overall_path_ego[self.obs_seq_len+1:, :2]) ** 2, axis=1))
                max_dist = np.max(point_dists)
                max_dist_idx = np.argmax(point_dists)
                min_dist = np.min(point_dists)
                if (max_dist > self.grid_size*1.1 or min_dist < self.grid_size*0.9):
                    print('{' + dataset_name[p] + '_frmidx%d_drvsts%d_speed%.1f_numpts%d} max (idx: %d) and min dists : %.2f, %.2f'
                          % (i, curr_seq_data[0,14], curr_seq_data[0,11], curr_seq_data.shape[0], max_dist_idx, max_dist, min_dist))

                # debug : check error
                path_dist = np.sqrt(np.sum((overall_path_ego[self.obs_seq_len, :2] - overall_path_ego[-1, :2]) ** 2))
                if (path_dist < 10):
                    print('{' + dataset_name[p] + '_frmidx%d_drvsts%d} path distance %d meter is smaller than 10 meter' % (i, curr_seq_data[0,14], path_dist))

                # -------------------------------------
                # cam0 view
                # -------------------------------------
                K, Rt = dataset_cali[p]
                img_number = int(curr_seq_data[0, 8])
                img = self.read_image(dataset_name[p], int(img_number))
                for m in range(self.obs_seq_len, overall_path_ego.shape[0]):

                    A = np.matmul(np.linalg.inv(Rt), overall_path_ego[m, :].reshape(4, 1))
                    B = np.matmul(K, A)

                    x = int(B[0, 0] * 0.5 / B[2, 0])
                    y = int(B[1, 0] * 0.88 / B[2, 0])

                    if (x < 0 or x > 640 - 1 or y > 320 - 1):
                        continue
                    img = cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

                text = dataset_name[p] + '_' + str(int(curr_seq_data[0, 0]))
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('test', img)
                cv2.waitKey(0)

                if (dataset_type[p] == 'train'):
                    if (np.random.rand(1) < self.val_ratio):
                        seq_list_val.append([overall_path_ego, overall_path_ENA, overall_drvact, Pm_intp, img_num_intp, did, steer])
                    else:
                        seq_list_train.append([overall_path_ego, overall_path_ENA, overall_drvact, Pm_intp, img_num_intp, did, steer])
                else:
                    seq_list_test.append([overall_path_ego, overall_path_ENA, overall_drvact, Pm_intp, img_num_intp, did, steer])

        raw_data = []
        raw_data.append(dataset_path)
        raw_data.append(dataset_name)
        raw_data.append(dataset_type)
        raw_data.append(dataset_cali)
        raw_data.append(seq_list_train)
        raw_data.append(seq_list_val)
        raw_data.append(seq_list_test)

        # save
        save_dir = './dataset/preprocessed_dataset.cpkl'
        f = open(save_dir, "wb")
        pickle.dump((raw_data), f, protocol=2)
        f.close()
        print('>> {%s} is made .. ' %save_dir)
        print('>> Dataset distribution .. (total: %d)' % np.sum(pop))
        pdf = pop / np.sum(pop[:])
        for p in range(10):
            print('   drvint %d : %d (%.2f)' % (p, pop[p, 0], pdf[p, 0]))

        print('>> number of false path %d' % num_false_path)

    def next_batch_eval(self, index, mode):

        # empty list creation
        xi_batch, xo_batch, xp_batch = [], [], []
        do_batch, dp_batch = [], []
        img_batch, img_ori_batch = [], []
        did_batch, path3d_batch, pm_batch = [], [], []
        depth_batch, steer_batch, drvsts_batch = [], [], []

        # for all target samples
        for i in range(len(index)):

            if (mode == 'valid'):
                overall_path_ego, overall_path_ENA, overall_drvact, Pm, img_num, did, steer = self.valid_data[index[i]]  # (seq_len+1 x 2)
            else:
                overall_path_ego, overall_path_ENA, overall_drvact, Pm, img_num, did, steer = self.test_data[index[i]]  # (seq_len+1 x 2)

            # calculate offset
            offset = np.zeros_like(overall_path_ego[:, 0:2])
            offset[1:] = overall_path_ego[1:, 0:2] - overall_path_ego[:-1, 0:2]

            # path info
            xi = overall_path_ego[0, 0:2].reshape(1, 2)
            xo = offset[:self.obs_seq_len]
            xp = offset[self.obs_seq_len:]

            # drvact info
            do = overall_drvact[:self.obs_seq_len]
            dp = overall_drvact[self.obs_seq_len:]

            # img info
            img = self.read_image(self.dataset_names[did], int(img_num[0]), isOriginal=False)
            img = image_preprocessing(np.copy(img), 0)

            if (mode == 'valid'):
                img_ori = 0
            else:
                img_ori = self.read_image(self.dataset_names[did], int(img_num[0]), isOriginal=True)

            # updated, 200212
            dp_seq = np.zeros(shape=(20, 10))
            for c in range(20):
                dp_seq[c, int(dp[c, 0])] = 1

            # stack into lists
            xi_batch.append(xi)
            xo_batch.append(xo)
            xp_batch.append(xp)
            do_batch.append(do)
            dp_batch.append(dp)
            img_batch.append(img)
            img_ori_batch.append(img_ori)
            did_batch.append(did)
            path3d_batch.append(overall_path_ego)
            pm_batch.append(Pm)
            drvsts_batch.append(dp_seq)
            depth_batch.append(self.read_depth(self.dataset_names[did], int(img_num[0])))
            steer_batch.append(steer / 30.0)

        return xi_batch, xo_batch, xp_batch, do_batch, dp_batch, img_batch, did_batch, path3d_batch, pm_batch, img_ori_batch, steer_batch

    def read_csv(self, file_dir):

        return np.genfromtxt(file_dir, delimiter=',')

    def create_pose_matrix(self, alt, roll, pitch, yaw, east, north):

        # translation vector
        t = np.array([east, north, alt]).reshape((3, 1))

        # rotation matrix
        rx = roll
        ry = pitch
        rz = yaw

        # rotation according to roll
        Rx = [1, 0, 0,
              0, math.cos(rx), -1 * math.sin(rx),
              0, math.sin(rx), math.cos(rx)]
        Rx = np.asarray(Rx).reshape((3, 3))

        # rotation according to pitch
        Ry = [math.cos(ry), 0, math.sin(ry),
              0, 1, 0,
              -1 * math.sin(ry), 0, math.cos(ry)]
        Ry = np.asarray(Ry).reshape((3, 3))

        # rotation according to yaw
        Rz = [math.cos(rz), -1 * math.sin(rz), 0,
              math.sin(rz), math.cos(rz), 0,
              0, 0, 1]
        Rz = np.asarray(Rz).reshape((3, 3))

        # transofrmation matrix
        R = np.matmul(np.matmul(Rz, Ry), Rx)

        return np.linalg.inv(affinematrix(R, t))

    def read_image(self, dataset_name, file_name, isOriginal=False):

        if (isOriginal == False):
            path = os.path.join(os.path.join(self.dataset_path, dataset_name), ('00_640x320/%08d.png' % int(file_name)))
        else:
            path = os.path.join(os.path.join(self.dataset_path, dataset_name), ('00/%08d.png' % int(file_name)))

        return cv2.imread(path)

    def read_depth(self, dataset_name, file_name):
        path = os.path.join(os.path.join(self.dataset_path, dataset_name), ('00_depth/%08d.png' % int(file_name)))
        return cv2.imread(path, -1)

class DrivingStateDataset(Dataset):

    def __init__(self, dataset_path, dataset_names, dataset, obs_seq_len, pred_seq_len, is_aug, dtype):

        self.dataset_path = dataset_path
        self.dataset_names = dataset_names
        self.dataset = dataset
        self.obs_seq_len = obs_seq_len
        self.pred_seq_len = pred_seq_len
        self.is_aug = is_aug
        self.dtype = dtype


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # -------------------------------------------
        # load and preprocessing
        # -------------------------------------------

        # read raw data
        overall_path_ego, overall_path_ENA, overall_drvact, Pmatrix, img_num, did, steer = self.dataset[idx]   # (seq_len+1 x 2)

        # calculate offset
        offset = np.zeros_like(overall_path_ego[:, 0:2])
        offset[1:] = overall_path_ego[1:, 0:2] - overall_path_ego[:-1, 0:2]

        # divide into segments
        xi = overall_path_ego[0, 0:2].reshape(1, 2)
        xo = offset[:self.obs_seq_len]
        xp = offset[self.obs_seq_len:]
        alt = overall_path_ego[:, 2:]
        dp_ = overall_drvact[self.obs_seq_len:]

        # make xi be the same size of xo
        xi_ext = np.zeros_like(xo)
        xi_ext[0, :] = np.copy(xi[:])

        # read image/Pmatrix/Pm_inv
        pminv = np.linalg.inv(Pmatrix[0])
        pm = Pmatrix[0]

        # apply augmentation only for current observation
        img = self.read_image(self.dataset_names[did], int(img_num[0]))
        img = image_preprocessing(np.copy(img), self.is_aug)

        # drvact seq, updated, 200212
        dp_seq = np.zeros(shape=(self.pred_seq_len, 10))
        for i in range(self.pred_seq_len):
            dp_seq[i, int(dp_[i, 0])] = 1

        # -------------------------------------------
        # conversion to tensor
        # -------------------------------------------
        xi_ext = torch.from_numpy(xi_ext).type(self.dtype)
        xo = torch.from_numpy(xo).type(self.dtype)
        xp = torch.from_numpy(xp).type(self.dtype)
        alt = torch.from_numpy(alt).type(self.dtype)
        dp, dp_fake = self.drvint_onehot(dp_, 0, self.dtype)
        img = torch.from_numpy(img).permute(2, 0, 1).type(self.dtype)
        pminv = torch.from_numpy(pminv).type(self.dtype)
        pm = torch.from_numpy(pm).type(self.dtype)
        dp_seq = torch.from_numpy(dp_seq).type(self.dtype)

        return xi_ext, xo, xp, alt, dp, dp_fake, img, pm, pminv, dp_seq

    def read_image(self, dataset_name, file_name):

        path = os.path.join(os.path.join(self.dataset_path, dataset_name), ('00_640x320/%08d.png' % int(file_name)))

        return cv2.imread(path)

    def drvint_onehot(self, drvint, Paction, dtype):

        drvint_onehot = np.zeros(shape=(1, 10))
        drvint_onehot_fake = np.zeros(shape=(1, 10))

        # gt label
        index = self.find_representative_drvint(drvint[:, 0])
        drvint_onehot[0, index] = 1

        # TODO : how to make fake label
        drvint_onehot_fake[0, 0] = 1

        y = torch.from_numpy(drvint_onehot)
        x = torch.from_numpy(drvint_onehot_fake)

        return y.type(dtype), x.type(dtype)

    def find_representative_drvint(self, input):

        '''
        input: seq_len x 1
        '''

        input_5pts = np.copy(input[0:5])
        rnd_idx = random.randint(0, 4)

        return int(input_5pts[rnd_idx])



def affinematrix(R, t):

    affine = np.concatenate((R, t), axis=1)
    affine = np.concatenate((affine, np.asarray([0, 0, 0, 1]).reshape((1, 4))), axis=0)

    return affine

def interpolate_1d(traj_len, y, scale):

    '''
    traj_len: length of input trajectory
    y: input trajectory
    scale: output will be of length int(traj_len * scale)
    '''

    x = np.linspace(0, traj_len-1, num=traj_len, endpoint=True)
    f = interp1d(x, y, kind='cubic')
    xnew = np.linspace(0, traj_len-1, num=int(traj_len*scale), endpoint=True)

    return f(xnew)

def interpolate_traj_data(cur_data, target_num_pos):

    '''
    frm idx (0)             gps_err (13)
    alt (1)                 drvint (14)
    roll (2)                temp (15)
    pitch (3)               go, tl, tr, ut, llc, rlc, lw, rw (16~23)
    yaw (4)
    heading (5)
    east (6)
    north (7)
    img_file_name (8)
    lidar_file_name (9)
    steer (10)
    speed (11)
    turn_signal (12)
    '''

    seq_len = cur_data.shape[0]
    if (seq_len < 2.5 * target_num_pos):
        scale = float(2.5 * target_num_pos / seq_len)

        alt = interpolate_1d(seq_len, np.copy(cur_data[:, 1]), scale)
        east = interpolate_1d(seq_len, np.copy(cur_data[:, 6]), scale)
        north = interpolate_1d(seq_len, np.copy(cur_data[:, 7]), scale)

        # plt.plot(east, north)
        # plt.show()

        intp_data = np.zeros(shape=(len(alt), 24))
        intp_data[:, 1] = alt
        intp_data[:, 6] = east
        intp_data[:, 7] = north

        for i in range(intp_data.shape[0]):
            cur_pos = intp_data[i, 6:8]

            candi_pos_x = (cur_data[:, 6] - cur_pos[0]) ** 2
            candi_pos_y = (cur_data[:, 7] - cur_pos[1]) ** 2

            err = candi_pos_x + candi_pos_y
            best_idx = np.argmin(err)

            intp_data[i, 0] = cur_data[best_idx, 0]
            intp_data[i, 2:6] = cur_data[best_idx, 2:6]
            intp_data[i, 8:23] = cur_data[best_idx, 8:23]

        return intp_data
    else:
        return cur_data


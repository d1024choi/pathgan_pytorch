from utils.functions import *

def generator_step(args, PathGen, PathDis, optimizer_g, optimizer_c, xo, xp, dp_onehot, dp_onehot_fake, conv_out, dp_seq):


    '''
    xo : displacement between positions in past traj (or speed)
    xp : displacement between positions in future path
    dp : onehot vector for input driving intention
    dp_seq : sequence of onehot vectors for sequential driving intentions
    '''

    # overall loss
    loss = torch.zeros(1).to(xo)

    # ---------------------------
    # generate paths
    # ---------------------------

    # traj generation
    overall_xp_gen, overall_seqdrvint_logits, _ = PathGen(xo, dp_onehot, conv_out, args.best_k)
    overall_xp_gen_stack = torch.stack(overall_xp_gen) # best_k x seq_len x batch x 2

    # find path with min err
    overall_xp_gen_np = overall_xp_gen_stack.detach().to('cpu').numpy()
    xp_np = xp.detach().to('cpu').numpy() # seq_len x batch x 2

    min_idx_batch = []
    for b in range(args.batch_size):
        cur_xp = xp_np[:, b, :] # seq_len x 2
        cur_xp_gen = overall_xp_gen_np[:, :, b, :] # best_k x seq_len x 2
        cur_mse_losses = []
        for z in range(args.best_k):
            err = np.mean((cur_xp - cur_xp_gen[z, :, :])**2)
            cur_mse_losses.append(err)
        min_idx_batch.append(np.argmin(np.array(cur_mse_losses)))

    best_xp_gen = []
    for b in range(args.batch_size):
        best_xp_gen.append(overall_xp_gen_stack[:, :, b, :][min_idx_batch[b]])

    # random path selection
    xp_gen = overall_xp_gen_stack[args.best_k - 1]
    seqdrvint_logits = torch.stack(overall_seqdrvint_logits)[args.best_k - 1]


    # ----------------------------------------------
    # path discrimination
    # ----------------------------------------------
    seqdrvint_prob = F.softmax(seqdrvint_logits, dim=2)
    out_scr, out_cls, out_scr_seqdrvint = PathDis(xp_gen, conv_out, seqdrvint_prob)


    # ---------------------------
    # loss calculation
    # ---------------------------

    # g loss
    g_loss_bce = gan_g_loss(out_scr)
    g_loss_bce_seqdrvint = gan_g_loss(out_scr_seqdrvint)

    # classification loss
    g_loss_cls = cross_entropy_loss(out_cls, dp_onehot)
    g_loss_seqdrvint_cls = cross_entropy_loss(seqdrvint_logits.view(-1, 10), dp_seq[1:, :, :].view(-1, 10))

    # ---------------------------
    # final loss
    # ---------------------------

    # mse
    g_loss_mse = l2_loss(xp.permute(1, 0, 2), torch.stack(best_xp_gen))

    # score
    g_loss_bce += args.alpha*g_loss_bce_seqdrvint

    # class
    g_loss_cls += args.theta*g_loss_seqdrvint_cls

    # final
    loss += (g_loss_bce + g_loss_cls + args.kappa*g_loss_mse)


    # ---------------------------
    # backpropagation
    # ---------------------------
    optimizer_g.zero_grad()
    loss.backward(retain_graph=True)
    if args.grad_clip > 0:
        nn.utils.clip_grad_norm_(PathGen.parameters(), args.grad_clip)
    optimizer_g.step()

    losses = [g_loss_mse.item(), g_loss_bce.item(), g_loss_cls.item()]
    return losses



def discriminator_step(args, PathGen, PathDis, optimizer_d, optimizer_c, xo, xp, dp_onehot, dp_onehot_fake, conv_out, dp_seq):

    '''
    xo : displacement between positions in past traj (or speed)
    xp : displacement between positions in future path
    dp : onehot vector for input driving intention
    dp_seq : sequence of onehot vectors for sequential driving intentions
    '''

    # overall loss
    loss = torch.zeros(1).to(xo)

    # ---------------------------
    # operation with real path
    # ---------------------------

    # real future path
    out_scr_real, out_cls_real, out_seqdrvint_real = PathDis(xp, conv_out, dp_seq[1:, :, :])

    # ---------------------------
    # operation with fake path
    # ---------------------------
    xp_fake, seqdrvint_logits, _ = PathGen(xo, dp_onehot, conv_out, 1)
    drvint_prob_fake = F.softmax(seqdrvint_logits[0], dim=2)
    out_scr_fake, _, out_seqdrvint_fake = PathDis(xp_fake[0].detach(), conv_out, drvint_prob_fake.detach())

    # ---------------------------
    # calculate loss
    # ---------------------------

    # reality score
    if (np.random.rand(1) < args.label_flip_prob):
        d_loss_bce_real, d_loss_bce_fake = gan_d_loss(out_scr_fake, out_scr_real)
        d_loss_bce_seqdrvint_real, d_loss_bce_seqdrvint_fake = gan_d_loss(out_seqdrvint_fake, out_seqdrvint_real)
    else:
        d_loss_bce_real, d_loss_bce_fake = gan_d_loss(out_scr_real, out_scr_fake)
        d_loss_bce_seqdrvint_real, d_loss_bce_seqdrvint_fake = gan_d_loss(out_seqdrvint_real, out_seqdrvint_fake)

    # classification loss
    d_loss_cls_real = cross_entropy_loss(out_cls_real, dp_onehot)

    # ---------------------------
    # # Final loss
    # ---------------------------
    loss += (d_loss_bce_fake + d_loss_bce_real + d_loss_cls_real) + args.beta * (d_loss_bce_seqdrvint_real + d_loss_bce_seqdrvint_fake)


    # ---------------------------
    # backpropagation
    # ---------------------------
    optimizer_d.zero_grad()
    optimizer_c.zero_grad()
    loss.backward()
    if args.grad_clip > 0:
        nn.utils.clip_grad_norm_(PathDis.parameters(), args.grad_clip)
    optimizer_d.step()
    optimizer_c.step()

    losses = [d_loss_bce_real.item(), d_loss_bce_fake.item(), d_loss_cls_real.item()]
    return losses

def evaluator_step(PathGen, ResNet, data_loader, dtype, args, e):

    # set to evaluation mode
    PathGen.eval()
    ResNet.eval()

    ADE, FDE = [], []
    for b in range(0, len(data_loader.valid_data), args.valid_step):

        # data preparation
        _, xo_batch, _, _, dp_batch, img_batch, _, path3d_batch, _, _, _ = \
            data_loader.next_batch_eval([b], mode='valid')

        xo = xo_batch[0]
        dp = dp_batch[0]
        img = img_batch[0]
        path3d = path3d_batch[0]


        # conversion to tensor
        xo_tensor = from_list_to_tensor_to_cuda(np.copy(xo), dtype)
        imgs_tensor = torch.from_numpy(np.array([img])).permute(0, 3, 1, 2).type(dtype).cuda()


        # offset prediction
        dp_onehot, dp_onehot_fake = drvint_onehot_batch([dp], dtype)
        overall_gen_offsets, _, _ = PathGen(xo_tensor, dp_onehot, ResNet(imgs_tensor), args.best_k)


        # find the best prediction sample
        err_values = np.zeros(shape=(args.best_k))
        offsets = []
        for z in range(args.best_k):
            gen_offsets = np.squeeze(overall_gen_offsets[z].detach().to('cpu').numpy())
            gen_recon = np.cumsum(gen_offsets, axis=0)
            err_values[z] = np.sum(abs(gen_recon - path3d[args.obs_seq_len:, 0:2]))
            offsets.append(gen_offsets)

        # reconstruct predicted traj
        min_idx = np.argmin(err_values)
        gen_recon = np.cumsum(offsets[min_idx], axis=0)

        # calculate error vector
        err_vector = path3d[args.obs_seq_len:, 0:2] - gen_recon
        displacement_error = np.sqrt(err_vector[:, 0] ** 2 + err_vector[:, 1] ** 2)
        ADE.append(displacement_error[:])
        FDE.append(displacement_error[-1])

    return np.mean(ADE), np.mean(FDE)

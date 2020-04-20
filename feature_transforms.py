import torch

def wct(alpha, cf, sf):
    #content image whitening
    cf = cf.double()
    c_channels, c_width, c_height = cf.size(0), cf.size(1), cf.size(2)
    cfv = cf.view(c_channels, -1)

    c_mean = torch.mean(cfv, 1)
    c_mean = c_mean.unsqueeze(1).expand_as(cfv)
    cfv = cfv - c_mean

    c_covm = torch.mm(cfv, cfv.t()).div((c_width * c_height) - 1)#covariance matrix
    c_u, c_e, c_v = torch.svd(c_covm, some=False)

    k_c = c_channels
    for i in range(c_channels):
        if c_e[i] < 0.00001:
            k_c = i
            break
    
    c_d = (c_e[0:k_c]).pow(-0.5)

    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
    whiten_cf = torch.mm(step2, cfv)

    #style image coloring
    sf = sf.double()
    s_channels, s_width, s_height = sf.size(0), sf.size(1), sf.size(2)
    sfv = sf.view(s_channels, -1)

    s_mean = torch.mean(sfv, 1)
    s_mean = s_mean.unsqueeze(1).expand_as(sfv)
    sfv = sfv - s_mean

    s_covm = torch.mm(sfv, sfv.t()).div((s_width * s_height) - 1)
    s_u, s_e, s_v = torch.svd(s_covm, some=False)

    s_k = c_channels
    for i in range(s_channels):
        if s_e[i] < 0.00001:
            s_k = i
            break

    s_d = (s_e[0:s_k]).pow(0.5)

    step1 = torch.mm(s_v[:, 0:s_k], torch.diag(s_d))
    step2 = torch.mm(step1, s_v[:, 0:s_k].t())
    colored = torch.mm(step2, whiten_cf)

    feature = colored + s_mean.resize_as_(colored)
    feature = feature.view_as(cf)

    target_feature = alpha * feature + (1.0 - alpha) * cf
    return target_feature.float().unsqueeze(0)

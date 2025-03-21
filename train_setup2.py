import numpy as np
import os
import torch
from utils.config import args
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
cudnn.benchmark = True
import nets as models
from utils.bar_show import progress_bar
from src.noisydataset import cross_modal_dataset
import src.utils as utils
import scipy
import scipy.spatial
from src.creat_data import creat_my_data
from src.new import data
from pre_data import office_load

best_acc = 0  # best test accuracy
start_epoch = 0
best_a_to_b = 0
best_b_to_a = 0
best_epoch = 0
args.log_dir = os.path.join(args.root_dir, 'logs', args.log_name)
args.ckpt_dir = os.path.join(args.root_dir, 'ckpt', args.ckpt_dir)

os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.ckpt_dir, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES']="0"
def load_dict(model, path):
    chp = torch.load(path)
    state_dict = model.state_dict()
    for key in state_dict:
        if key in chp['model_state_dict']:
            state_dict[key] = chp['model_state_dict'][key]
    model.load_state_dict(state_dict)

def main():
    print('===> Preparing data ..')
    print(args.choose_model)
    if args.choose_model == 'olddata':
        train_dataset = cross_modal_dataset(args.data_name, args.noisy_ratio, 'train')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            # sampler=sampler,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False
        )

        valid_dataset = cross_modal_dataset(args.data_name, args.noisy_ratio, 'valid')
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )

        test_dataset = cross_modal_dataset(args.data_name, args.noisy_ratio, 'test')
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )
    elif args.choose_model == 'mydata':
        train_dataset = data(args.data_name,args.noisy_ratio, 'train')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            # sampler=sampler,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False
        )

        valid_dataset = data(args.data_name,args.noisy_ratio,  'valid')
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )

        test_dataset = data(args.data_name,args.noisy_ratio,  'test')
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )
    elif args.choose_model == 'predata':
        
        train_dataset = office_load(args,'train')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=False
        )
        valid_dataset = office_load(args, 'valid')
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )

        test_dataset = office_load(args,  'test')
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )

    print('===> Building Models..')
    multi_models = []
    n_view = len(train_dataset.train_data)
    #print(n_view)
    #print(train_dataset.train_data[0].shape[0])
    for v in range(n_view):
        if v == args.views.index('Img'): # Images
            
            multi_models.append(models.__dict__['ImageNet'](input_dim=512, output_dim=args.output_dim).cuda())
        elif v == args.views.index('Img'):
            #multi_models.append(models.__dict__['Model'](args,1024,1024).cuda())
            multi_models.append(models.__dict__['ImageNet1'](input_dim=512, output_dim=args.output_dim).cuda())
        else: # Default to use ImageNet
            multi_models.append(models.__dict__['ImageNet'](input_dim=512, output_dim=args.output_dim).cuda())
    #print(multi_models)
    C = torch.Tensor(args.output_dim, args.output_dim)
    C = torch.nn.init.orthogonal(C, gain=1)[:, 0: train_dataset.class_num].cuda()
    C.requires_grad = True

    embedding = torch.eye(train_dataset.class_num).cuda()
    embedding.requires_grad = False

    parameters = [C]
    for v in range(n_view):
        parameters += list(multi_models[v].parameters())
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(parameters, lr=args.lr, betas=[0.5, 0.999], weight_decay=args.wd)
    if args.ls == 'cos':
        lr_schedu = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=0, last_epoch=-1)
    else:
        lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [200, 400], gamma=0.1)

    if args.loss == 'CE':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    elif args.loss == 'MCE':
        criterion = utils.MeanClusteringError(train_dataset.class_num, tau=args.tau).cuda()
    elif args.loss == 'MAE':
        criterion = utils.MAELoss(num_classes=train_dataset.class_num).cuda()
    elif args.loss == 'GCE':
        criterion = utils.GCELoss(q = 1).cuda()
    elif args.loss == 'GECE':
        criterion = utils.GECELoss(q = 0.8).cuda()
    elif args.loss == 'FL':
        criterion = utils.FocalLoss().cuda()
    elif args.loss == 'RCE':
        criterion = utils.RCELoss(num_classes=train_dataset.class_num).cuda()
    elif args.loss == 'NLNL':
        criterion = utils.NLNL(num_classes=train_dataset.class_num,train_loader=train_loader).cuda()
    elif args.loss == 'SCE':
        criterion = utils.SCELoss(num_classes=train_dataset.class_num).cuda()
        #print('criterion')
    elif args.loss == 'NCE':
        criterion = utils.NCELoss(num_classes=train_dataset.class_num).cuda()
    elif args.loss == 'NMAE':
        criterion = utils.NMAE(num_classes=train_dataset.class_num).cuda()
    elif args.loss == 'NRCE':
        criterion = utils.NRCELoss(num_classes=train_dataset.class_num).cuda()
    elif args.loss == 'NFL':
        criterion = utils.NormalizedFocalLoss(num_classes=train_dataset.class_num).cuda()
    elif args.loss == 'NGCE':
        criterion = utils.NGCELoss(num_classes=train_dataset.class_num).cuda()

    else:
        raise Exception('No such loss function.')

    summary_writer = SummaryWriter(args.log_dir)

    if args.resume:
        ckpt = torch.load(os.path.join(args.ckpt_dir, args.resume))
        for v in range(n_view):
            multi_models[v].load_state_dict(ckpt['model_state_dict_%d' % v])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        print('===> Load last checkpoint data')
    else:
        start_epoch = 0
        print('===> Start from scratch')
    #print(multi_models)
    def set_train():
        for v in range(n_view):
            multi_models[v].train()

    def set_eval():
        for v in range(n_view):
            multi_models[v].eval()

    def cross_modal_contrastive_ctriterion(fea, tau=1.):   #MCL的损失Lc
        batch_size = fea[0].shape[0]
        all_fea = torch.cat(fea)
        sim = all_fea.mm(all_fea.t())

        sim = (sim / tau).exp()
        sim = sim - sim.diag().diag()
        sim_sum1 = sum([sim[:, v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag1 = torch.cat([sim_sum1[v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        loss1 = -(diag1 / sim.sum(1)).log().mean()

        sim_sum2 = sum([sim[v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag2 = torch.cat([sim_sum2[:, v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        loss2 = -(diag2 / sim.sum(1)).log().mean()
        return loss1 + loss2

    def train(epoch):
        print('\nEpoch: %d / %d' % (epoch, args.max_epochs))
        set_train()
        train_loss, loss_list, correct_list, total_list = 0., [0.] * n_view, [0.] * n_view, [0.] * n_view
        
        for batch_idx, (batches, targets, index) in enumerate(train_loader):

            batches, targets = [batches[v].cuda() for v in range(n_view)], [targets[v].cuda() for v in range(n_view)]     
            norm = C.norm(dim=0, keepdim=True)
            C.data = (C / norm).detach()
            for v in range(n_view):
                for name, param in multi_models[v].named_parameters():
                    if v == 0:
                        if "image1_model" in name:
                            param.requires_grad = False
                    elif v == 1:
                        if "image2_model" in name:
                            param.requires_grad = False

            for v in range(n_view):
                multi_models[v].zero_grad()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            outputs0,feature0 = multi_models[0](batches[0])
            outputs1,feature1 = multi_models[1](batches[1])


            outputs=[outputs0,outputs1]
            feature=[feature0,feature1]
            preds = [outputs[v].mm(C) for v in range(n_view)]

            loss_cca1 = 0
            loss_cca2 = 0
            loss_cca = 0
       
            outputs_src_cls_tgt = multi_models[0].fc1(feature0)
            outputs_tgt_cls_tgt = multi_models[1].fc1(feature0)
            outputs_tgt_cls_src = multi_models[1].fc1(feature1)
            outputs_src_cls_src = multi_models[0].fc1(feature1)
            loss_cca1 += args.lambda_cross_domain * (
                    (outputs_src_cls_tgt - outputs_tgt_cls_tgt).abs().sum(1).mean())
            loss_cca2 += args.lambda_cross_domain * (
                    (outputs_src_cls_src - outputs_tgt_cls_src).abs().sum(1).mean())
            loss_cca = loss_cca1 + loss_cca2
         
            #print(preds[0])
            #print(preds[0].shape,targets[0].shape)
            losses = [criterion(preds[v], targets[v]) for v in range(n_view)]#loss直接是交叉熵函数，哪里用了RC
            #print(targets[v].long())
            loss = sum(losses)
            print('in_domain:',loss,'loss_cca:',loss_cca)
            #loss = args.beta * loss + (1. - args.beta) * cross_modal_contrastive_ctriterion(outputs, tau=args.tau)
            loss = args.beta * loss + (1. - args.beta) * loss_cca
            if epoch >= 0:
                loss.backward()
                optimizer.step()
            train_loss += loss.item()

            for v in range(n_view):
                loss_list[v] += losses[v]
                _, predicted = preds[v].max(1)
                total_list[v] += targets[v].size(0)
                acc = predicted.eq(targets[v]).sum().item()
                correct_list[v] += acc
            #progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | LR: %g'
                        # % (train_loss / (batch_idx + 1), optimizer.param_groups[0]['lr']))

        train_dict = {('view_%d_loss' % v): loss_list[v] / len(train_loader) for v in range(n_view)}
        train_dict['sum_loss'] = train_loss / len(train_loader)
        summary_writer.add_scalars('Loss/train', train_dict, epoch)
        summary_writer.add_scalars('Accuracy/train', {'view_%d_acc': correct_list[v] / total_list[v] for v in range(n_view)}, epoch)

    def eval(data_loader, epoch, mode='test'):
        fea, lab = [[] for _ in range(n_view)], [[] for _ in range(n_view)]
        test_loss, loss_list, correct_list, total_list = 0., [0.] * n_view, [0.] * n_view, [0.] * n_view
        k = True
        with torch.no_grad():
            if 1 == 1:
                # for v in range (n_view):
                #     iter_test = iter(data_loader.dataset.train_data[v])
                # #   print(len(loader))
                
                #     for i in range(len(data_loader.dataset.train_data[v])):
                #         data = next(iter_test)
                #         inputs = data
                #         gt_labels = data_loader.dataset.noise_label[v][i]
                #         print(gt_labels)
                #         inputs = inputs.cuda()
                #         #print(data[1])
                #         outputs = multi_models[v](inputs)
                #         # outputs = F.normalize(outputs)
                #         fea[v].append(outputs)
                #         lab[v].append(gt_labels)
                #         pred.append(outputs.mm(C))
                for batch_idx, (batches, targets, index) in enumerate(data_loader):
                    batches, targets = [batches[v].cuda() for v in range(n_view)], [targets[v].cuda() for v in range(n_view)]


                    outputs0,feature0 = multi_models[0](batches[0])
                    outputs1,feature1 = multi_models[1](batches[1])

                    outputs=[outputs0,outputs1]
                    feature=[feature0,feature1]
                    #print(outputs)
                    pred, losses = [], []
                    for v in range(n_view):
                        fea[v].append(outputs[v])
                        lab[v].append(targets[v])
                        pred.append(outputs[v].mm(C))
                        losses.append(criterion(pred[v], targets[v].long()))
                        loss_list[v] += losses[v]
                        _, predicted = pred[v].max(1)
                        total_list[v] += targets[v].size(0)
                        acc = predicted.eq(targets[v]).sum().item()
                        correct_list[v] += acc
                    loss = sum(losses)
                    test_loss += loss.item()
            else:
                pred, losses = [], []
                for v in range(n_view):
                    if mode == 'train':
                        count = int(np.ceil(2982) / data_loader.batch_size)
                    if mode == 'test' or 'valid':
                        count = int(np.ceil(780) / data_loader.batch_size)
                    for ct in range(count):
                        #print((data_loader.dataset.train_data[v][ct * data_loader.batch_size: (ct + 1) * data_loader.batch_size]))
                        targets = torch.Tensor(data_loader.dataset.noise_label[v][ct * data_loader.batch_size: (ct + 1) * data_loader.batch_size]).long().cuda()
                        batch = []
                        for i in range(ct * data_loader.batch_size,(ct + 1) * data_loader.batch_size):
                            batches = torch.Tensor(data_loader.dataset.train_data[v][i])
                            batch.append(batches)
                        print(batch)
                        #batch = torch.Tensor(batch).cuda()
                        
                        outputs = multi_models[v](batch)
                       
                        fea[v].append(outputs)
                        lab[v].append(targets)
                        pred.append(outputs.mm(C))
                        losses.append(criterion(pred[v], targets))
                        loss_list[v] += losses[v]
                        _, predicted = pred[v].max(1)
                        total_list[v] += targets.size(0)
                        acc = predicted.eq(targets).sum().item()
                        correct_list[v] += acc
                    loss = sum(losses)
                    test_loss += loss.item()

            fea = [torch.cat(fea[v]).cpu().detach().numpy() for v in range(n_view)]
            lab = [torch.cat(lab[v]).cpu().detach().numpy() for v in range(n_view)]
        test_dict = {('view_%d_loss' % v): loss_list[v] / len(data_loader) for v in range(n_view)}
        test_dict['sum_loss'] = test_loss / len(data_loader)
        summary_writer.add_scalars('Loss/' + mode, test_dict, epoch)

        summary_writer.add_scalars('Accuracy/' + mode, {('view_%d_acc' % v): correct_list[v] / total_list[v] for v in range(n_view)}, epoch)
        return fea, lab

    def multiview_test(fea, lab):
        MAPs = np.zeros([n_view, n_view])
        val_dict = {}
        print_str = ''
        for i in range(n_view):
            for j in range(n_view):
                if i == j:
                    continue       #fx_calc_map_label(train, train_labels, test, test_label, k=0, metric='cosine'):
                MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
                key = '%s2%s' % (args.views[i], args.views[j])
                val_dict[key] = MAPs[i, j]
                print_str = print_str + key + ': %.3f\t' % val_dict[key]
        return val_dict, print_str


    def test(epoch):
            global best_acc
            set_eval()
            # switch to evaluate mode
            fea, lab = eval(train_loader, epoch, 'train')
            #print(n_view)
            MAPs = np.zeros([n_view, n_view])
            train_dict = {}
            for i in range(n_view):
                for j in range(n_view):
                    MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
                    #print(lab[j])
                    train_dict['%s2%s' % (args.views[i], args.views[j])] = MAPs[i, j]

            train_avg = MAPs.sum() / n_view / (n_view - 1.)
            train_dict['avg'] = train_avg
            summary_writer.add_scalars('Retrieval/train', train_dict, epoch)#加入tensorboard中
            #print(train_dict)
            fea, lab = eval(valid_loader, epoch, 'valid')
            MAPs = np.zeros([n_view, n_view])
            val_dict = {}
            print_val_str = 'Validation: '

            for i in range(n_view):
                for j in range(n_view):
                    if i == j:
                        continue
                    MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
                    key = '%s2%s' % (args.views[i], args.views[j])
                    val_dict[key] = MAPs[i, j]
                    print_val_str = print_val_str + key +': %g\t' % val_dict[key]


            val_avg = MAPs.sum() / n_view / (n_view - 1.)
            val_dict['avg'] = val_avg
            print_val_str = print_val_str + 'Avg: %g' % val_avg
            summary_writer.add_scalars('Retrieval/valid', val_dict, epoch)

            fea, lab = eval(test_loader, epoch, 'test')
            MAPs = np.zeros([n_view, n_view])
            test_dict = {}
            print_test_str = 'Test: '
            for i in range(n_view):
                for j in range(n_view):
                    if i == j:#跳过相同模态的检测
                        continue
                    MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
                    key = '%s2%s' % (args.views[i], args.views[j])
                    test_dict[key] = MAPs[i, j]
                    print_test_str = print_test_str + key + ': %g\t' % test_dict[key]

            test_avg = MAPs.sum() / n_view / (n_view - 1.)
            print_test_str = print_test_str + 'Avg: %g' % test_avg
            test_dict['avg'] = test_avg
            summary_writer.add_scalars('Retrieval/test', test_dict, epoch)
            with open('./mydata/office_information.txt','a') as f:
                k=str(args.loss)
                b=str(args.dset)
                c=str(epoch)
                d=str(args.beta)
                line = 'dset:'+b+' ' + 'epoch:'+c +' ' +'loss:'+ k +' '+'beta:'+ d +' '+ print_test_str +'\n'
                f.write(line)
            print(print_val_str)
            print(print_test_str)
            global best_a_to_b
            global best_b_to_a
            global best_epoch
            if val_avg > best_acc:
                best_acc = val_avg
                
                print('Saving..')
                state = {}
                for v in range(n_view):
                    # models[v].load_state_dict(ckpt['model_state_dict_%d' % v])
                    state['model_state_dict_%d' % v] = multi_models[v].state_dict()
                for key in test_dict:
                    state[key] = test_dict[key]
                state['epoch'] = epoch
                state['optimizer_state_dict'] = optimizer.state_dict()
                state['C'] = C
                torch.save(state, os.path.join(args.ckpt_dir, '%s_%s_%d_best_checkpoint.t7' % ('MRL', args.data_name, args.output_dim)))
            if MAPs[0,1] > best_a_to_b:
                best_a_to_b = MAPs[0,1]
                best_epoch = epoch
                torch.save(multi_models[0], './dest_pth/a2r.pth')
            if MAPs[1,0] > best_b_to_a:
                best_b_to_a = MAPs[1,0]
                torch.save(multi_models[1], './dest_pth/r2a.pth')
            print(args.dset,'%.3f'%best_a_to_b,  '%.3f'%best_b_to_a,best_epoch)
            return val_dict

    # test(1)
    best_prec1 = 0.
    lr_schedu.step(start_epoch)
    #train(-1)
    #results = test(-1)
    for epoch in range(start_epoch, args.max_epochs):
        train(epoch)
        lr_schedu.step(epoch)
        test_dict = test(epoch + 1)
        if test_dict['avg'] == best_acc:
            multi_model_state_dict = [{key: value.clone() for (key, value) in m.state_dict().items()} for m in multi_models]
            W_best = C.clone()

    print('Evaluation on Last Epoch:')
    fea, lab = eval(test_loader, epoch, 'test')
    test_dict, print_str = multiview_test(fea, lab)
    print(print_str)

    print('Evaluation on Best Validation:')
    [multi_models[v].load_state_dict(multi_model_state_dict[v]) for v in range(n_view)]
    fea, lab = eval(test_loader, epoch, 'test')
    test_dict, print_str = multiview_test(fea, lab)
    print(print_str)
    import scipy.io as sio
    save_dict = dict(**{args.views[v]: fea[v] for v in range(n_view)}, **{args.views[v] + '_lab': lab[v] for v in range(n_view)})
    save_dict['C'] = W_best.detach().cpu().numpy()
    sio.savemat('features/%s_%g.mat' % (args.data_name, args.noisy_ratio), save_dict)

def fx_calc_map_multilabel_k(train, train_labels, test, test_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(test, train, metric)
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i].reshape(-1)

        tmp_label = (np.dot(train_labels[order], test_label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res)

def fx_calc_map_label(train, train_labels, test, test_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(test, train, metric)

    ord = dist.argsort(1)
    #print(ord)
    numcases = train_labels.shape[0]
    #print(numcases)
    #print(test_label)
    #print(train_labels)
    if k == 0:
        k = numcases
    if k == -1:
        ks = [50, numcases]
    else:
        ks = [k]

    def calMAP(_k):
        _res = []
        for i in range(len(test_label)):
            order = ord[i]
            p = 0.0
            r = 0.0
            for j in range(_k):
                if test_label[i] == train_labels[order[j]]:
                    r += 1
                    p += (r / (j + 1))
            if r > 0:
                _res += [p / r]
            else:
                _res += [0]
        return np.mean(_res)

    res = []
    for k in ks:
        res.append(calMAP(k))
    return res

if __name__ == '__main__':
    main()


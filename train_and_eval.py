import numpy as np
import copy
import torch
import dgl
from utils import set_seed
import torch.nn.functional as F

"""
1. Train and eval
"""


def train(model, data, feats, labels, criterion, optimizer, idx_train, lamb=1):
    """
    GNN full-batch training. Input the entire graph `g` as data.
    lamb: weight parameter lambda
    """
    model.train()

    # Compute loss and prediction
    if "GCN" in model.model_name or "GAT" in model.model_name or "APPNP" in model.model_name:
        _, logits = model(data, feats)
    else:
        logits = model(data, feats)
    out = logits.log_softmax(dim=1)
    loss = criterion(out[idx_train], labels[idx_train])
    loss_val = loss.item()

    loss *= lamb
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_val


def train_sage(model, dataloader, feats, labels, criterion, optimizer, lamb=1):
    """
    Train for GraphSAGE. Process the graph in mini-batches using `dataloader` instead the entire graph `g`.
    lamb: weight parameter lambda
    """
    device = feats.device
    model.train()
    total_loss = 0
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [blk.int().to(device) for blk in blocks]
        batch_feats = feats[input_nodes]
        batch_labels = labels[output_nodes]
        # Compute loss and prediction
        logits = model(blocks, batch_feats)
        out = logits.log_softmax(dim=1)
        loss = criterion(out, batch_labels)
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


def train_mini_batch(model, feats, labels, batch_size, criterion, optimizer, lamb=1):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    lamb: weight parameter lambda
    """
    model.train()
    num_batches = max(1, feats.shape[0] // batch_size)
    idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]

    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    for i in range(num_batches):
        _, logits = model(None, feats[idx_batch[i]])
        out = logits.log_softmax(dim=1)

        loss = criterion(out, labels[idx_batch[i]])
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / num_batches


def get_PGD_inputs(model, data, feats, labels, criterion, args, idx_train=None):
    iters = 5
    eps = args.adv_eps
    alpha = eps / 4

    # init
    delta = torch.rand(feats.shape) * eps * 2 - eps
    delta = delta.to(feats.device)
    delta = torch.nn.Parameter(delta)

    for i in range(iters):
        p_feats = feats + delta

        if "GCN" in model.model_name or "GAT" in model.model_name or "APPNP" in model.model_name:
            _, logits = model(data, p_feats)
            out = logits.log_softmax(dim=1)[idx_train]
        elif "MLP" in model.model_name:
            _, logits = model(None, p_feats)
            out = logits.log_softmax(dim=1)
        else:
            logits = model(data, p_feats)
            out = logits.log_softmax(dim=1)

        loss = criterion(out, labels)
        loss.backward()

        # delta update
        delta.data = delta.data + alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)

    output = delta.detach()
    return output


def train_both_distillation_batch_adv_mlp(model, feats, labels, teacher_emb, args, batch_size, criterion, optimizer,
                                      lamb=1):
    model.train()
    num_batches = max(1, feats.shape[0] // batch_size)
    idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]

    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    for i in range(num_batches):
        batch_mlp_emb, logits = model(None, feats[idx_batch[i]])
        out = logits.log_softmax(dim=1)

        # adversarial learning
        if args.adv:
            adv_deltas = get_PGD_inputs(model, None, feats[idx_batch[i]], labels[idx_batch[i]], criterion, args)
            adv_feats = torch.add(feats[idx_batch[i]], adv_deltas)
            _, adv_logits = model(None, adv_feats)
            adv_out = adv_logits.log_softmax(dim=1)
            loss_adv = criterion(adv_out, labels[idx_batch[i]])

        # feature distillation
        if args.feat_distill:
            feat_distill_loss = torch.tensor(0.0)
            for j, _ in enumerate(args.teacher_emb_layers):
                teacher_hidden_layer_num = args.teacher_emb_layers[j]
                mlp_hidden_layer_num = args.student_emb_layers[j]
                layer_loss = args.feat_distill_weights[j]
                
                batch_mlp_emb_layer = batch_mlp_emb[mlp_hidden_layer_num]
                batch_mlp_emb_layer = model.encode_model4kd(batch_mlp_emb_layer)[j]

                batch_teacher_emb_layer = teacher_emb[j][idx_batch[i]]

                mlp_sim_matrix = torch.mm(batch_mlp_emb_layer, batch_mlp_emb_layer.T)
                teacher_sim_matrix = torch.mm(batch_teacher_emb_layer, batch_teacher_emb_layer.T)
                loss_feature = torch.mean((teacher_sim_matrix - mlp_sim_matrix) ** 2)
                feat_distill_loss += float(layer_loss) * loss_feature

        loss_label = criterion(out, labels[idx_batch[i]])
        loss = loss_label
        if args.adv:
            loss += 0.1 * loss_adv
        if args.feat_distill:
            # loss += float(args.feat_distill_weight) * loss_feature
            loss += feat_distill_loss
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / num_batches

def train_both_distillation_batch_adv_gnn(model, data, feats, labels, teacher_emb, args, criterion, optimizer, idx_train,
                                      lamb=1):
    model.train()
    # Compute loss and prediction
    if "GCN" in model.model_name or "GAT" in model.model_name or "APPNP" in model.model_name:
        student_emb, logits = model(data, feats)
    else:
        logits = model(data, feats)
    out = logits.log_softmax(dim=1)

    # adversarial learning
    if args.adv:
        adv_deltas = get_PGD_inputs(model, data, feats, labels[idx_train], criterion, args)
        adv_feats = torch.add(feats[idx_train], adv_deltas)
        _, adv_logits = model(data, adv_feats)
        adv_out = adv_logits.log_softmax(dim=1)
        loss_adv = criterion(adv_out, labels[idx_train])

    # feature distillation
    if args.feat_distill:
        feat_distill_loss = torch.tensor(0.0)
        for j, _ in enumerate(args.teacher_emb_layers):
            student_hidden_layer_num = args.student_emb_layers[j]
            layer_loss = args.feat_distill_weights[j]

            # print(len(student_emb))
            # print(idx_train.shape)
            
            student_emb_layer = student_emb[student_hidden_layer_num][idx_train]
            # print(student_emb_layer.shape)
            student_emb_layer = model.encode_model4kd(student_emb_layer)[j]
            # print(student_emb_layer.shape)    # 140 by 64

            teacher_emb_layer = teacher_emb[j]
            # print(len(teacher_emb))
            # print(teacher_emb_layer.shape)

            student_sim_matrix = torch.mm(student_emb_layer, student_emb_layer.T)
            teacher_sim_matrix = torch.mm(teacher_emb_layer, teacher_emb_layer.T)
            loss_feature = torch.mean((teacher_sim_matrix - student_sim_matrix) ** 2)
            feat_distill_loss += float(layer_loss) * loss_feature

    loss_label = criterion(out[idx_train], labels[idx_train])
    loss = loss_label
    if args.adv:
        loss += 0.1 * loss_adv
    if args.feat_distill:
        # loss += float(args.feat_distill_weight) * loss_feature
        loss += feat_distill_loss

    loss *= lamb
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def train_both_distillation_batch_adv_sage(model, data, feats, labels, teacher_emb, args, criterion, optimizer,
                                      lamb=1):
    device = feats.device
    model.train()
    total_loss = 0
    for step, (input_nodes, output_nodes, blocks) in enumerate(data):
        blocks = [blk.int().to(device) for blk in blocks]
        batch_feats = feats[input_nodes]
        batch_labels = labels[output_nodes]
        block_teacher_emb = [teacher_emb[i][output_nodes] for i in range(len(teacher_emb))]

        # Compute loss and prediction
        block_student_emb, logits = model(blocks, batch_feats, return_sage_emb=True)
        out = logits.log_softmax(dim=1)

        for i in range(len(block_student_emb)):
            block_student_emb[i] = block_student_emb[i][ : blocks[-1].num_dst_nodes()]

        # adversarial learning
        if args.adv:
            adv_deltas = get_PGD_inputs(model, blocks, batch_feats, batch_labels, criterion, args)
            adv_feats = torch.add(batch_feats, adv_deltas)
            adv_logits = model(blocks, adv_feats)
            adv_out = adv_logits.log_softmax(dim=1)
            loss_adv = criterion(adv_out, batch_labels)

        # feature distillation
        if args.feat_distill:
            feat_distill_loss = torch.tensor(0.0)
            for j, _ in enumerate(args.teacher_emb_layers):
                student_hidden_layer_num = args.student_emb_layers[j]
                layer_loss = args.feat_distill_weights[j]

                block_student_emb_layer = block_student_emb[student_hidden_layer_num]
                block_student_emb_layer = model.encode_model4kd(block_student_emb_layer)[j]

                block_teacher_emb_layer = block_teacher_emb[j]

                student_sim_matrix = torch.mm(block_student_emb_layer, block_student_emb_layer.T)
                teacher_sim_matrix = torch.mm(block_teacher_emb_layer, block_teacher_emb_layer.T)
                loss_feature = torch.mean((teacher_sim_matrix - student_sim_matrix) ** 2)
                feat_distill_loss += float(layer_loss) * loss_feature
        
    
        loss_label = criterion(out, batch_labels)
        loss = loss_label
        if args.adv:
            loss += 0.1 * loss_adv
        if args.feat_distill:
            loss += feat_distill_loss
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(data)

def compute_ece(probs, labels, n_bins=15):
    confidences, predictions = probs.max(1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            acc = accuracies[mask].float().mean()
            conf = confidences[mask].mean()
            ece += (mask.float().mean()) * torch.abs(acc - conf)
    return ece.item()


def compute_adaptive_ece(probs, labels, num_bins=15):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    # Sort by confidence
    sorted_confidences, sorted_indices = confidences.sort()
    sorted_accuracies = accuracies[sorted_indices]

    n = len(confidences)
    bin_size = n // num_bins
    ace = 0.0

    for i in range(num_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < num_bins - 1 else n

        bin_conf = sorted_confidences[start:end]
        bin_acc = sorted_accuracies[start:end]

        if len(bin_conf) == 0:
            continue

        avg_conf = bin_conf.mean()
        avg_acc = bin_acc.float().mean()

        ace += (avg_conf - avg_acc).abs() * len(bin_conf)

    return ace / n

def compute_brier_score(probs, labels, num_classes):
    num_classes = probs.size(1)
    true_one_hot = F.one_hot(labels, num_classes=num_classes).float()
    brier = (probs - true_one_hot).pow(2).sum(dim=1).mean() # MSE
    return brier

def evaluate(model, data, feats, labels, num_classes, criterion, evaluator, idx_eval=None, temperatures=None, calculate_calibration=False, return_logits=False):
    """
    Returns:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.

    Note that the temperature parameter is used to calculate only ece, ace and brier, not logits.
    """
    model.eval()
    with torch.no_grad():
        if "GCN" in model.model_name or "GAT" in model.model_name or "APPNP" in model.model_name:
            emb_list, logits = model.inference(data, feats)
        else:
            logits, emb_list = model.inference(data, feats)
        out = logits.log_softmax(dim=1)
        eces = []
        aces = []
        briers = []
        if idx_eval is None:
            loss = criterion(out, labels)
            score = evaluator(out, labels)
            if calculate_calibration:
                if temperatures is None:
                    probs = (logits).softmax(dim=-1)
                    eces = compute_ece(probs, labels)
                    aces = compute_adaptive_ece(probs, labels)
                    briers = compute_brier_score(probs, labels, num_classes)
                else:
                    for temperature in temperatures:
                        probs = (logits/temperature).softmax(dim=-1)
                        ece = compute_ece(probs, labels)
                        ace = compute_adaptive_ece(probs, labels)
                        brier = compute_brier_score(probs, labels, num_classes)
                        eces.append(ece)
                        aces.append(ace)
                        briers.append(brier)
        else:
            loss = criterion(out[idx_eval], labels[idx_eval])
            score = evaluator(out[idx_eval], labels[idx_eval])
            if calculate_calibration:
                if temperatures is None:
                    probs = logits.softmax(dim=-1)
                    eces = compute_ece(probs[idx_eval], labels[idx_eval])
                    aces = compute_adaptive_ece(probs[idx_eval], labels[idx_eval])
                    briers = compute_brier_score(probs[idx_eval], labels[idx_eval], num_classes)
                else:
                    for temperature in temperatures:
                        probs = (logits/temperature).softmax(dim=-1)
                        ece = compute_ece(probs[idx_eval], labels[idx_eval])
                        ace = compute_adaptive_ece(probs[idx_eval], labels[idx_eval])
                        brier = compute_brier_score(probs[idx_eval], labels[idx_eval], num_classes)
                        eces.append(ece)
                        aces.append(ace)
                        briers.append(brier)

    if return_logits:
        if calculate_calibration:
            return out, loss.item(), score, emb_list, eces, aces, briers, logits
        else:
            return out, loss.item(), score, emb_list, logits
    else:
        if calculate_calibration:
            return out, loss.item(), score, emb_list, eces, aces, briers
        else:
            return out, loss.item(), score, emb_list


def evaluate_mini_batch(
        model, feats, labels, num_classes, criterion, batch_size, evaluator, idx_eval=None, temperatures=None, calculate_calibration=False, return_logits=False
):
    """
    Evaluate MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    Return:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.

    Note that the temperature parameter is used to calculate only ece, ace and brier, not logits.
    """

    model.eval()
    with torch.no_grad():
        num_batches = int(np.ceil(len(feats) / batch_size))
        out_list = []
        logits_list = []
        for i in range(num_batches):
            _, logits = model.inference(None, feats[batch_size * i: batch_size * (i + 1)])
            out = logits.log_softmax(dim=1)
            logits_list += [logits.detach()]
            out_list += [out.detach()]

        logits_all = torch.cat(logits_list)
        out_all = torch.cat(out_list)

        eces = []
        aces = []
        briers = []
        
        if idx_eval is None:
            loss = criterion(out_all, labels)
            score = evaluator(out_all, labels)
            if calculate_calibration:
                if temperatures is None:
                    probs = logits.softmax(dim=-1)
                    eces = compute_ece(probs, labels)
                    aces = compute_adaptive_ece(probs, labels)
                    briers = compute_brier_score(probs, labels, num_classes)
                else:
                    for temperature in temperatures:
                        probs = (logits_all/temperature).softmax(dim=-1)
                        ece = compute_ece(probs, labels)
                        ace = compute_adaptive_ece(probs, labels)
                        brier = compute_brier_score(probs, labels, num_classes)
                        eces.append(ece)
                        aces.append(ace)
                        briers.append(brier)
        else:
            loss = criterion(out_all[idx_eval], labels[idx_eval])
            score = evaluator(out_all[idx_eval], labels[idx_eval])
            if calculate_calibration:
                if temperatures is None:
                    probs = logits_all.softmax(dim=-1)
                    eces = compute_ece(probs[idx_eval], labels[idx_eval])
                    aces = compute_adaptive_ece(probs[idx_eval], labels[idx_eval])
                    briers = compute_brier_score(probs[idx_eval], labels[idx_eval], num_classes)
                else:
                    for temperature in temperatures:
                        probs = (logits_all/temperature).softmax(dim=-1)
                        ece = compute_ece(probs[idx_eval], labels[idx_eval])
                        ace = compute_adaptive_ece(probs[idx_eval], labels[idx_eval])
                        brier = compute_brier_score(probs[idx_eval], labels[idx_eval], num_classes)
                        eces.append(ece)
                        aces.append(ace)
                        briers.append(brier)

    if return_logits:
        if calculate_calibration:
            return out_all, loss.item(), score, eces, aces, briers, logits_all
        else:
            return out_all, loss.item(), score, logits_all
    else:
        if calculate_calibration:
            return out_all, loss.item(), score, eces, aces, briers
        else:
            return out_all, loss.item(), score


"""
2. Run teacher
"""


def run_transductive(
        conf,
        model,
        g,
        feats,
        labels,
        indices,
        criterion,
        evaluator,
        optimizer,
        logger,
        loss_and_score,
        args
):
    """
    Train and eval under the transductive setting.
    The train/valid/test split is specified by `indices`.
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]

    idx_train, idx_val, idx_test = indices

    feats = feats.to(device)
    labels = labels.to(device)

    temperatures = args.temperatures
    num_classes = args.label_dim

    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves memory and CPU.
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        dataloader = dgl.dataloading.DataLoader(
            g,
            idx_train,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"],
        )
        

        # SAGE inference is implemented as layer by layer, so the full-neighbor sampler only collects one-hop neighors
        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader_eval = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        data = dataloader
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:
        feats_train, labels_train = feats[idx_train], labels[idx_train]
        feats_val, labels_val = feats[idx_val], labels[idx_val]
        feats_test, labels_test = feats[idx_test], labels[idx_test]
    else:
        g = g.to(device)
        data = g
        data_eval = g

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name:
            loss = train_sage(model, data, feats, labels, criterion, optimizer)
        elif "MLP" in model.model_name:
            loss = train_mini_batch(
                model, feats_train, labels_train, batch_size, criterion, optimizer
            )
        else:
            loss = train(model, data, feats, labels, criterion, optimizer, idx_train)

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                _, loss_train, score_train = evaluate_mini_batch(
                    model, feats_train, labels_train, num_classes, criterion, batch_size, evaluator
                )
                _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, labels_val, num_classes, criterion, batch_size, evaluator
                )
                _, loss_test, score_test = evaluate_mini_batch(
                    model, feats_test, labels_test, num_classes, criterion, batch_size, evaluator
                )
            else:
                out, loss_train, score_train, emb_list = evaluate(
                    model, data_eval, feats, labels, num_classes, criterion, evaluator, idx_train
                )
                _, loss_val, score_val, _ = evaluate(
                    model, data_eval, feats, labels, num_classes, criterion, evaluator, idx_val
                )
                _, loss_test, score_test, _ = evaluate(
                    model, data_eval, feats, labels, num_classes, criterion, evaluator, idx_test
                )

            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_train,
                    loss_val,
                    loss_test,
                    score_train,
                    score_val,
                    score_test,
                ]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    if "MLP" in model.model_name:
        out, _, score_val, logits = evaluate_mini_batch(
            model, feats, labels, num_classes, criterion, batch_size, evaluator, idx_val, return_logits=True
        )
        _, _, _, ece_test, ace_test, brier_test = evaluate_mini_batch(
            model, feats, labels, num_classes, criterion, batch_size, evaluator, idx_test, temperatures=temperatures, calculate_calibration=True
        )
        emb_list = None
    else:
        out, _, score_val, emb_list, logits = evaluate(
            model, data_eval, feats, labels, num_classes, criterion, evaluator, idx_val, return_logits=True
        )
        _, _, _, _, ece_test, ace_test, brier_test = evaluate(
            model, data_eval, feats, labels, num_classes, criterion, evaluator, idx_test, temperatures=temperatures, calculate_calibration=True
        )

    score_test = evaluator(out[idx_test], labels[idx_test])
    return out, score_test, emb_list, ece_test, ace_test, brier_test, logits


def run_inductive(
        conf,
        model,
        g,
        feats,
        labels,
        indices,
        criterion,
        evaluator,
        optimizer,
        logger,
        loss_and_score,
        args
):
    """
    Train and eval under the inductive setting.
    The train/valid/test split is specified by `indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    loss_and_score: Stores losses and scores.
    """

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices

    feats = feats.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_g = g.subgraph(idx_obs)

    temperatures = args.temperatures
    num_classes = args.label_dim

    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves momory and CPU.
        obs_g.create_formats_()
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        obs_dataloader = dgl.dataloading.DataLoader(
            obs_g,
            obs_idx_train,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        obs_dataloader_eval = dgl.dataloading.DataLoader(
            obs_g,
            torch.arange(obs_g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )
        dataloader_eval = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        obs_data = obs_dataloader
        obs_data_eval = obs_dataloader_eval
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:
        feats_train, labels_train = obs_feats[obs_idx_train], obs_labels[obs_idx_train]
        feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
        feats_test_tran, labels_test_tran = (
            obs_feats[obs_idx_test],
            obs_labels[obs_idx_test],
        )
        feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    else:
        obs_g = obs_g.to(device)
        g = g.to(device)

        obs_data = obs_g
        obs_data_eval = obs_g
        data_eval = g

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name:
            loss = train_sage(
                model, obs_data, obs_feats, obs_labels, criterion, optimizer
            )
        elif "MLP" in model.model_name:
            loss = train_mini_batch(
                model, feats_train, labels_train, batch_size, criterion, optimizer
            )
        else:
            loss = train(
                model,
                obs_data,
                obs_feats,
                obs_labels,
                criterion,
                optimizer,
                obs_idx_train,
            )

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                _, loss_train, score_train = evaluate_mini_batch(
                    model, feats_train, labels_train, num_classes, criterion, batch_size, evaluator
                )
                _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, labels_val, num_classes, criterion, batch_size, evaluator
                )
                _, loss_test_tran, score_test_tran = evaluate_mini_batch(
                    model,
                    feats_test_tran,
                    labels_test_tran,
                    num_classes,
                    criterion,
                    batch_size,
                    evaluator,
                )
                _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                    model,
                    feats_test_ind,
                    labels_test_ind,
                    num_classes,
                    criterion,
                    batch_size,
                    evaluator,
                )
            else:
                obs_out, loss_train, score_train, emb_list = evaluate(
                    model,
                    obs_data_eval,
                    obs_feats,
                    obs_labels,
                    num_classes,
                    criterion,
                    evaluator,
                    obs_idx_train,
                )
                _, loss_val, score_val, _ = evaluate(
                    model,
                    obs_data_eval,
                    obs_feats,
                    obs_labels,
                    num_classes,
                    criterion,
                    evaluator,
                    obs_idx_val,
                )
                _, loss_test_tran, score_test_tran, _ = evaluate(
                    model,
                    obs_data_eval,
                    obs_feats,
                    obs_labels,
                    num_classes,
                    criterion,
                    evaluator,
                    obs_idx_test,
                )

                # Evaluate the inductive part with the full graph
                out, loss_test_ind, score_test_ind, emb_list  = evaluate(
                    model, data_eval, feats, labels, num_classes, criterion, evaluator, idx_test_ind
                )
            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_train,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_train,
                    score_val,
                    score_test_tran,
                    score_test_ind,
                ]
            ]
            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    if "MLP" in model.model_name:
        obs_out, _, score_val, obs_logits = evaluate_mini_batch(
            model, obs_feats, obs_labels, num_classes, criterion, batch_size, evaluator, obs_idx_val, return_logits=True
        )
        # _, _, _, ece_test_tran, ace_test_tran, brier_test_tran = evaluate_mini_batch(
        #     model, obs_feats, obs_labels, num_classes, criterion, batch_size, evaluator, obs_idx_test, temperatures=temperatures, calculate_calibration=True
        # )
        out, _, score_test_ind, ece_test_ind, ace_test_ind, brier_test_ind, logits = evaluate_mini_batch(
            model, feats, labels, num_classes, criterion, batch_size, evaluator, idx_test_ind, temperatures=temperatures, calculate_calibration=True, return_logits=True
        )

    else:
        obs_out, _, score_val, emb_list, obs_logits = evaluate(
            model,
            obs_data_eval,
            obs_feats,
            obs_labels,
            num_classes, 
            criterion,
            evaluator,
            obs_idx_val,
            return_logits=True
        )
        # _, _, _, _, ece_test_tran, ace_test_tran, brier_test_tran = evaluate(
        #     model, obs_data_eval, obs_feats, obs_labels, num_classes, criterion, evaluator, obs_idx_test, temperatures=temperatures, calculate_calibration=True
        # )
        out, _, score_test_ind, emb_list, ece_test_ind, ace_test_ind, brier_test_ind, logits = evaluate(
            model, data_eval, feats, labels, num_classes, criterion, evaluator, idx_test_ind, temperatures=temperatures, calculate_calibration=True, return_logits=True
        )

    out[idx_obs] = obs_out
    logits[idx_obs] = obs_logits
    logger.info(
        f"Best valid model at epoch: {best_epoch :3d}, score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )

    if "MLP" in model.model_name:  # used in train_teacher with MLP as teacher model
        return out, score_test_ind, None, ece_test_ind, ace_test_ind, brier_test_ind, logits
    return out, score_test_ind, emb_list, ece_test_ind, ace_test_ind, brier_test_ind, logits


"""
3. Distill
"""


def distill_run_transductive(
        conf,
        model,
        feats,
        labels,
        logits_temp_t_all,
        out_emb_t_all,
        distill_indices,
        criterion_l,
        criterion_t,
        evaluator,
        optimizer,
        logger,
        loss_and_score,
        graph,
        args
):
    """
    Distill training and eval under the transductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively
    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    idx_l, idx_t, idx_val, idx_test = distill_indices

    feats = feats.to(device)
    labels = labels.to(device)

    if "SAGE" in model.model_name:
        graph.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        dataloader_l = dgl.dataloading.DataLoader(
            graph,
            idx_l,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"]
        )

        dataloader_t = dgl.dataloading.DataLoader(
            graph,
            idx_t,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"]
        )

        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader_eval = dgl.dataloading.DataLoader(
            graph,
            torch.arange(graph.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        data_t = dataloader_t
        data_l = dataloader_l
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:
        feats_l, labels_l = feats[idx_l], labels[idx_l]
        feats_t, logits_temp_t = feats[idx_t], logits_temp_t_all[idx_t]
        feats_val, labels_val = feats[idx_val], labels[idx_val]
        feats_test, labels_test = feats[idx_test], labels[idx_test]
        out_t_soft = logits_temp_t.log_softmax(dim=1)
    else:
        graph = graph.to(device)
        data = graph
        data_eval = graph
    
    out_emb_t = [t[idx_t] for t in out_emb_t_all]
    out_emb_l = [t[idx_l] for t in out_emb_t_all]
    out_t_soft_all = logits_temp_t_all.log_softmax(dim=1)

    num_classes = args.label_dim

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name:
            loss_l = train_both_distillation_batch_adv_sage(
                model, data_l, feats, labels, out_emb_t_all, args, criterion_l, optimizer, 1 - lamb
            )
            loss_t = train_both_distillation_batch_adv_sage(
                model, data_t, feats, out_t_soft_all, out_emb_t_all, args, criterion_t, optimizer, lamb
            )
        elif "MLP" in model.model_name:
            loss_l = train_both_distillation_batch_adv_mlp(
                model, feats_l, labels_l, out_emb_l, args, batch_size, criterion_l, optimizer, 1 - lamb
            )
            loss_t = train_both_distillation_batch_adv_mlp(
                model, feats_t, out_t_soft, out_emb_t, args, batch_size, criterion_t, optimizer, lamb
            )
        else:
            loss_l = train_both_distillation_batch_adv_gnn(
                model, data, feats, labels, out_emb_l, args, criterion_l, optimizer, idx_l, 1 - lamb
            )
            loss_t = train_both_distillation_batch_adv_gnn(
                model, data, feats, out_t_soft_all, out_emb_t, args, criterion_t, optimizer, idx_t, lamb
            )

        loss = loss_l + loss_t

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                _, loss_l, score_l = evaluate_mini_batch(
                    model, feats_l, labels_l, num_classes, criterion_l, batch_size, evaluator
                )
                _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, labels_val, num_classes, criterion_l, batch_size, evaluator
                )
                _, loss_test, score_test = evaluate_mini_batch(
                    model, feats_test, labels_test, num_classes, criterion_l, batch_size, evaluator
                )
            else:
                out, loss_l, score_l, _ = evaluate(
                    model, data_eval, feats, labels, num_classes, criterion_l, evaluator, idx_l
                )
                _, loss_val, score_val, _ = evaluate(
                    model, data_eval, feats, labels, num_classes, criterion_l, evaluator, idx_val
                )
                _, loss_test, score_test, _ = evaluate(
                    model, data_eval, feats, labels, num_classes, criterion_l, evaluator, idx_test
                )
            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
            )
            loss_and_score += [
                [epoch, loss_l, loss_val, loss_test, score_l, score_val, score_test]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    if "MLP" in model.model_name:
        out, _, score_val = evaluate_mini_batch(
            model, feats, labels, num_classes, criterion_l, batch_size, evaluator, idx_val
        )
        _, _, _, ece_test, ace_test, brier_test = evaluate_mini_batch(
            model, feats, labels, num_classes, criterion_l, batch_size, evaluator, idx_test, calculate_calibration=True
        )
    else:
        out, _, score_val, _ = evaluate(
            model, data_eval, feats, labels, num_classes, criterion_l, evaluator, idx_val
        )
        _, _, _, _, ece_test, ace_test, brier_test = evaluate(
            model, data_eval, feats, labels, num_classes, criterion_l, evaluator, idx_test, calculate_calibration=True
        )
    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test = evaluator(out[idx_test], labels[idx_test])

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    
    return out, score_test, ece_test, ace_test, brier_test

def distill_run_inductive(
        conf,
        model,
        feats,
        labels,
        logits_temp_t_all,
        out_emb_t_all,
        distill_indices,
        criterion_l,
        criterion_t,
        evaluator,
        optimizer,
        logger,
        loss_and_score,
        graph,
        args
):
    """
    Distill training and eval under the inductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively.
    loss_and_score: Stores losses and scores.
    """

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    (
        obs_idx_l,
        obs_idx_t,
        obs_idx_val,
        obs_idx_test,
        idx_obs,
        idx_test_ind,
    ) = distill_indices

    feats = feats.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_logits_temp_t = logits_temp_t_all[idx_obs]
    obs_g = graph.subgraph(idx_obs)

    if "SAGE" in model.model_name:
        obs_g.create_formats_()
        graph.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        obs_dataloader_l = dgl.dataloading.DataLoader(
            obs_g,
            obs_idx_l,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"],
        )
        obs_dataloader_t = dgl.dataloading.DataLoader(
            obs_g,
            obs_idx_t,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"]
        )
        
        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        obs_dataloader_eval = dgl.dataloading.DataLoader(
            obs_g,
            torch.arange(obs_g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"]
        )
        dataloader_eval = dgl.dataloading.DataLoader(
            graph,
            torch.arange(graph.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"]
        )

        obs_data_t = obs_dataloader_t
        obs_data_l = obs_dataloader_l
        obs_data_eval = obs_dataloader_eval
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:
        feats_l, labels_l = obs_feats[obs_idx_l], obs_labels[obs_idx_l]
        feats_t, logits_temp_t = obs_feats[obs_idx_t], obs_logits_temp_t[obs_idx_t]
        feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
        feats_test_tran, labels_test_tran = (
            obs_feats[obs_idx_test],
            obs_labels[obs_idx_test],
        )
        feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

        out_t_soft = logits_temp_t.log_softmax(dim=1)
    else:
        obs_g = obs_g.to(device)
        graph = graph.to(device)

        obs_data = obs_g
        obs_data_eval = obs_g
        data_eval = graph
    
    out_t_soft_all = logits_temp_t_all.log_softmax(dim=1)
    obs_out_t_soft = obs_logits_temp_t.log_softmax(dim=1)

    out_emb_t = [t[obs_idx_t] for t in out_emb_t_all]
    out_emb_l = [t[obs_idx_l] for t in out_emb_t_all]
    num_classes = args.label_dim
    
    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name:
            loss_l = train_both_distillation_batch_adv_sage(
                model, obs_data_l, obs_feats, obs_labels, out_emb_t_all, args, criterion_l, optimizer, 1 - lamb
            )
            loss_t = train_both_distillation_batch_adv_sage(
                model, obs_data_t, obs_feats, obs_out_t_soft, out_emb_t_all, args, criterion_t, optimizer, lamb
            )
        elif "MLP" in model.model_name:
            loss_l = train_both_distillation_batch_adv_mlp(
                model, feats_l, labels_l, out_emb_l, args, batch_size, criterion_l, optimizer, 1 - lamb
            )
            loss_t = train_both_distillation_batch_adv_mlp(
                model, feats_t, out_t_soft, out_emb_t, args, batch_size, criterion_t, optimizer, lamb
            )
        else:
            loss_l = train_both_distillation_batch_adv_gnn(
                model, obs_data, obs_feats, obs_labels, out_emb_l, args, criterion_l, optimizer, obs_idx_l, 1 - lamb
            )
            loss_t = train_both_distillation_batch_adv_gnn(
                model, obs_data, obs_feats, obs_out_t_soft, out_emb_t, args, criterion_t, optimizer, obs_idx_t, lamb
            )
        loss = loss_l + loss_t

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                _, loss_l, score_l = evaluate_mini_batch(
                    model, feats_l, labels_l, num_classes, criterion_l, batch_size, evaluator
                )
                _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, labels_val, num_classes, criterion_l, batch_size, evaluator
                )
                _, loss_test_tran, score_test_tran = evaluate_mini_batch(
                    model,
                    feats_test_tran,
                    labels_test_tran,
                    num_classes,
                    criterion_l,
                    batch_size,
                    evaluator,
                )
                _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                    model,
                    feats_test_ind,
                    labels_test_ind,
                    num_classes,
                    criterion_l,
                    batch_size,
                    evaluator,
                )
            else:
                _, loss_l, score_l, _ = evaluate(
                    model,
                    obs_data_eval,
                    obs_feats,
                    obs_labels,
                    num_classes,
                    criterion_l,
                    evaluator,
                    obs_idx_l,
                )
                _, loss_val, score_val, _ = evaluate(
                    model,
                    obs_data_eval,
                    obs_feats,
                    obs_labels,
                    num_classes,
                    criterion_l,
                    evaluator,
                    obs_idx_val
                )
                _, loss_test_tran, score_test_tran, _ = evaluate(
                    model,
                    obs_data_eval,
                    obs_feats,
                    obs_labels,
                    num_classes,
                    criterion_l,
                    evaluator,
                    obs_idx_test,
                )
                _, loss_test_ind, score_test_ind, _  = evaluate(
                    model,
                    data_eval,
                    feats,
                    labels,
                    num_classes,
                    criterion_l,
                    evaluator,
                    idx_test_ind
                )

            logger.debug(
                f"Ep {epoch:3d} | l: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_l,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_l,
                    score_val,
                    score_test_tran,
                    score_test_ind,
                ]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    if "MLP" in model.model_name:
        obs_out, _, score_val = evaluate_mini_batch(
            model, obs_feats, obs_labels, num_classes, criterion_l, batch_size, evaluator, obs_idx_val
        )
        # _, _, _, ece_test_tran, ace_test_tran, brier_test_tran = evaluate_mini_batch(
        #     model, obs_feats, obs_labels, num_classes, criterion_l, batch_size, evaluator, obs_idx_test, calculate_calibration=True
        # )
        out, _, score_test_ind, ece_test_ind, ace_test_ind, brier_test_ind = evaluate_mini_batch(
            model, feats, labels, num_classes, criterion_l, batch_size, evaluator, idx_test_ind, calculate_calibration=True
        )
    else:
        obs_out, _, score_val, _ = evaluate(
            model,
            obs_data_eval,
            obs_feats,
            obs_labels,
            num_classes, 
            criterion_l,
            evaluator,
            obs_idx_val,
        )
        # _, _, _, _, ece_test_tran, ace_test_tran, brier_test_tran = evaluate(
        #     model, obs_data_eval, obs_feats, obs_labels, num_classes, criterion_l, evaluator, obs_idx_test, calculate_calibration=True
        # )
        out, _, score_test_ind, _, ece_test_ind, ace_test_ind, brier_test_ind = evaluate(
            model, data_eval, feats, labels, num_classes, criterion_l, evaluator, idx_test_ind, calculate_calibration=True
        )

    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
    out[idx_obs] = obs_out

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d} score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return out, score_test_ind, ece_test_ind, ace_test_ind, brier_test_ind

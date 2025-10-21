import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, SAGEConv, APPNPConv, GATConv


class MLP(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            position_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            norm_type="none",
            graph=None,
            byte_idx_train=None,
            labels_one_hot=None,
            teacher_hidden_dim=None,
            num_distill_layers=None
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim + position_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim + position_dim, hidden_dim))

            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))
            self.mlp_feature_encoders = []
            if teacher_hidden_dim and num_distill_layers:
                for layer in range(num_distill_layers):
                    self.mlp_feature_encoders.append(nn.Linear(hidden_dim, teacher_hidden_dim))

    def forward(self, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)

        return h_list, h

    def encode_model4kd(self, mlp_feat):
        mlp_feature_encoders = []
        for encoder in self.mlp_feature_encoders:
            mlp_feature_encoders.append(self.dropout(F.relu(encoder(mlp_feat))))
        return mlp_feature_encoders


"""
Adapted from the SAGE implementation from the official DGL example
https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/ogbn-products/graphsage/main.py
"""


class SAGE(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            activation,
            norm_type="none",
            teacher_hidden_dim=None,
            num_distill_layers=None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(SAGEConv(input_dim, output_dim, "gcn"))
        else:
            self.layers.append(SAGEConv(input_dim, hidden_dim, "gcn"))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, "gcn"))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(SAGEConv(hidden_dim, output_dim, "gcn"))
            self.sage_feature_encoders = []
            if teacher_hidden_dim and num_distill_layers:
                for layer in range(num_distill_layers):
                    self.sage_feature_encoders.append(nn.Linear(hidden_dim, teacher_hidden_dim))

    def forward(self, blocks, feats):
        h = feats
        h_list = []
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[: block.num_dst_nodes()]
            h = layer(block, (h, h_dst))
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h_list, h

    def inference(self, dataloader, feats):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        dataloader : The entire graph loaded in blocks with full neighbors for each node.
        feats : The input feats of entire node set.
        """
        device = feats.device
        emb_list = []
        for l, layer in enumerate(self.layers):
            # add-on
            hidden_emb = torch.zeros(
                feats.shape[0],
                self.hidden_dim if l != self.num_layers - 1 else self.output_dim,
            ).to(device)

            y = torch.zeros(
                feats.shape[0],
                self.hidden_dim if l != self.num_layers - 1 else self.output_dim,
            ).to(device)
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0].int().to(device)

                h = feats[input_nodes]
                h_dst = h[: block.num_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != self.num_layers - 1:
                    hidden_emb[output_nodes] = h
                    if self.norm_type != "none":
                        h = self.norms[l](h)
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h

            feats = y
            if l != self.num_layers - 1:
                emb_list.append(hidden_emb)
        # return y
        return y, emb_list
    
    def encode_model4kd(self, sage_feat):
        sage_feature_encoders = []
        for encoder in self.sage_feature_encoders:
            sage_feature_encoders.append(self.dropout(F.relu(encoder(sage_feat))))
        return sage_feature_encoders


class GCN(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            activation,
            norm_type="none",
            teacher_hidden_dim=None,
            num_distill_layers=None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(GraphConv(input_dim, output_dim, activation=activation))
        else:
            self.layers.append(GraphConv(input_dim, hidden_dim, activation=activation))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(
                    GraphConv(hidden_dim, hidden_dim, activation=activation)
                )
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(GraphConv(hidden_dim, output_dim))
            self.gcn_feature_encoders = []
            if teacher_hidden_dim and num_distill_layers:
                for layer in range(num_distill_layers):
                    self.gcn_feature_encoders.append(nn.Linear(hidden_dim, teacher_hidden_dim))

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.dropout(h)
        return h_list, h
    
    def encode_model4kd(self, gcn_feat):
        gcn_feature_encoders = []
        for encoder in self.gcn_feature_encoders:
            gcn_feature_encoders.append(self.dropout(F.relu(encoder(gcn_feat))))
        return gcn_feature_encoders


class GAT(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            activation,
            num_heads=8,
            attn_drop=0.3,
            negative_slope=0.2,
            residual=False,
            teacher_hidden_dim=None,
            num_distill_layers=None
    ):
        super(GAT, self).__init__()
        # For GAT, the number of layers is required to be > 1
        assert num_layers > 1

        hidden_dim //= num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)

        heads = ([num_heads] * num_layers) + [1]
        # input (no residual)
        self.layers.append(
            GATConv(
                input_dim,
                hidden_dim,
                heads[0],
                dropout_ratio,
                attn_drop,
                negative_slope,
                False,
                self.activation,
            )
        )

        for l in range(1, num_layers - 1):
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.layers.append(
                GATConv(
                    hidden_dim * heads[l - 1],
                    hidden_dim,
                    heads[l],
                    dropout_ratio,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )

        self.layers.append(
            GATConv(
                hidden_dim * heads[-2],
                output_dim,
                heads[-1],
                dropout_ratio,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

        self.gat_feature_encoders = []
        if teacher_hidden_dim and num_distill_layers:
            for layer in range(num_distill_layers):
                self.gat_feature_encoders.append(nn.Linear(hidden_dim*num_heads, teacher_hidden_dim))

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            # [num_head, node_num, nclass] -> [num_head, node_num*nclass]
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = h.flatten(1)
                h_list.append(h)
            else:
                h = h.mean(1)
        return h_list, h
    
    def encode_model4kd(self, gat_feat):
        gat_feature_encoders = []
        for encoder in self.gat_feature_encoders:
            gat_feature_encoders.append(self.dropout(F.relu(encoder(gat_feat))))
        return gat_feature_encoders


class APPNP(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            activation,
            norm_type="none",
            edge_drop=0.5,
            alpha=0.1,
            k=10,
            teacher_hidden_dim=None,
            num_distill_layers=None
    ):

        super(APPNP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))
            self.appnp_feature_encoders = []
            if teacher_hidden_dim and num_distill_layers:
                for layer in range(num_distill_layers):
                    self.appnp_feature_encoders.append(nn.Linear(hidden_dim, teacher_hidden_dim))

        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)

            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)

        h = self.propagate(g, h)
        return h_list, h
    
    def encode_model4kd(self, appnp_feat):
        appnp_feature_encoders = []
        for encoder in self.appnp_feature_encoders:
            appnp_feature_encoders.append(self.dropout(F.relu(encoder(appnp_feat))))
        return appnp_feature_encoders


class Model(nn.Module):
    """
    Wrapper of different models
    """

    def __init__(self, conf, args, position_feature_dim=0, graph=None, byte_idx_train=None, labels_one_hot=None):
        super(Model, self).__init__()
        self.model_name = conf["model_name"]
        if "MLP" in conf["model_name"]:
            # origin
            self.encoder = MLP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                position_dim=position_feature_dim,
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                norm_type=conf["norm_type"],
                graph=graph,
                byte_idx_train=byte_idx_train,
                labels_one_hot=labels_one_hot,
                teacher_hidden_dim=conf["teacher_hidden_dim"],
                num_distill_layers=conf["num_distill_layers"]
            ).to(conf["device"])

        elif "SAGE" in conf["model_name"]:
            self.encoder = SAGE(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
                teacher_hidden_dim=conf["teacher_hidden_dim"],
                num_distill_layers=conf["num_distill_layers"]
            ).to(conf["device"])
        elif "GCN" in conf["model_name"]:
            self.encoder = GCN(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
                teacher_hidden_dim=conf["teacher_hidden_dim"],
                num_distill_layers=conf["num_distill_layers"]
            ).to(conf["device"])
        elif "GAT" in conf["model_name"]:
            self.encoder = GAT(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                attn_drop=conf["attn_dropout_ratio"],
                teacher_hidden_dim=conf["teacher_hidden_dim"],
                num_distill_layers=conf["num_distill_layers"]
            ).to(conf["device"])
        elif "APPNP" in conf["model_name"]:
            self.encoder = APPNP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
                teacher_hidden_dim=conf["teacher_hidden_dim"],
                num_distill_layers=conf["num_distill_layers"]
            ).to(conf["device"])

    def forward(self, data, feats, return_sage_emb=False):
        """
        data: a graph `g` or a `dataloader` of blocks
        """
        if "MLP" in self.model_name:
            return self.encoder(feats)
        elif "GCN" in self.model_name or "GAT" in self.model_name or "APPNP" in self.model_name:
            return self.encoder(data, feats)
        else:
            if return_sage_emb:
                return self.encoder(data, feats)
            else:
                return self.encoder(data, feats)[1]

    def forward_fitnet(self, data, feats):
        """
        Return a tuple (h_list, h)
        h_list: intermediate hidden representation
        h: final output
        """
        if "MLP" in self.model_name:
            return self.encoder(feats)
        else:
            return self.encoder(data, feats)

    def inference(self, data, feats):
        if "SAGE" in self.model_name:
            return self.encoder.inference(data, feats)
        else:
            return self.forward(data, feats)

    def encode_model4kd(self, feat):
        return self.encoder.encode_model4kd(feat)

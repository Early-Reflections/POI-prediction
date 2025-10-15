import torch
from torch import nn
from layer import (
    BertCheckinEmbedding,
    TrajFeatureEmbedding,
    CheckinEmbedding,
    EdgeEmbedding,
    HypergraphTransformer,
    TimeEncoder,
    DistanceEncoderHSTLSTM,
    DistanceEncoderSTAN,
    DistanceEncoderSimple
)


class STHGCN(nn.Module):
    def __init__(self, cfg):
        super(STHGCN, self).__init__()

        self.device = cfg.run_args.device
        self.batch_size = cfg.run_args.batch_size
        self.eval_batch_size = cfg.run_args.eval_batch_size
        self.do_traj2traj = cfg.model_args.do_traj2traj  # Whether to perform trajectory-to-trajectory convolution
        self.distance_encoder_type = cfg.model_args.distance_encoder_type  # Distance encoder type
        self.dropout_rate = cfg.model_args.dropout_rate
        self.generate_edge_attr = cfg.model_args.generate_edge_attr  # Whether to generate edge attribute embeddings
        self.num_conv_layers = len(cfg.model_args.sizes)  # Number of convolution layers
        self.num_poi = cfg.dataset_args.num_poi  # Number of POIs
        self.embed_fusion_type = cfg.model_args.embed_fusion_type  # Embedding fusion type

        # Initialize checkin embedding layer
        self.checkin_embedding_layer = BertCheckinEmbedding(
            embed_size=cfg.model_args.embed_size,
            fusion_type=self.embed_fusion_type,
            dataset_args=cfg.dataset_args
        )
        self.checkin_embed_size = self.checkin_embedding_layer.output_embed_size  # Embedding output dimension

        # # % add the traj feature embedding layer %
        # self.traj_feature_embedding_layer = TrajFeatureEmbedding(
        #     embed_size=cfg.model_args.embed_size,
        #     fusion_type=self.embed_fusion_type,
        #     dataset_args=cfg.dataset_args
        # )

        # Initialize edge type embedding layer
        # if cfg.model_args.edge_fusion_mode == 'add':
        self.edge_type_embedding_layer = EdgeEmbedding(
            embed_size=self.checkin_embed_size,
            fusion_type=self.embed_fusion_type,
            num_edge_type=cfg.model_args.num_edge_type
        )

        # Activation function selection
        if cfg.model_args.activation == 'elu':
            self.act = nn.ELU()
        elif cfg.model_args.activation == 'relu':
            self.act = nn.RReLU()
        elif cfg.model_args.activation == 'leaky_relu':
            self.act = nn.LeakyReLU()
        else:
            self.act = torch.tanh

        # Set continuous time encoding dimension based on whether additive fusion is performed
        if cfg.conv_args.time_fusion_mode == 'add':
            continuous_encoder_dim = self.checkin_embed_size
        else:
            continuous_encoder_dim = cfg.model_args.st_embed_size

        # Initialize edge attribute layer based on whether to generate edge attribute embeddings
        if self.generate_edge_attr:
            self.edge_attr_embedding_layer = EdgeEmbedding(
                embed_size=self.checkin_embed_size,
                fusion_type=self.embed_fusion_type,
                num_edge_type=cfg.model_args.num_edge_type
            )
        else:
            # If not generating edge attribute embeddings, directly transform raw edge features with linear layer
            if cfg.conv_args.edge_fusion_mode == 'add':
                self.edge_attr_embedding_layer = nn.Linear(3, self.checkin_embed_size)
            else:
                self.edge_attr_embedding_layer = None

        self.feature_attention_dropout = cfg.conv_args.feature_attention_dropout
        self.feature_residual_alpha = cfg.conv_args.feature_residual_alpha

        self.conv_list = nn.ModuleList()

        # conv for ci2traj within which some ci2traj relations have been removed by time to prevent data leakage


        # Initialize hypergraph Transformer layer for time filtering
        self.conv_for_time_filter = HypergraphTransformer(
            in_channels=self.checkin_embed_size,
            out_channels=self.checkin_embed_size,
            attn_heads=cfg.conv_args.num_attention_heads,
            residual_beta=cfg.conv_args.residual_beta,
            learn_beta=cfg.conv_args.learn_beta,
            dropout=cfg.conv_args.conv_dropout_rate,
            trans_method=cfg.conv_args.trans_method,
            edge_fusion_mode=cfg.conv_args.edge_fusion_mode,
            time_fusion_mode=cfg.conv_args.time_fusion_mode,
            head_fusion_mode=cfg.conv_args.head_fusion_mode,
            residual_fusion_mode=None,
            edge_dim=None,
            rel_embed_dim=self.checkin_embed_size,
            time_embed_dim=continuous_encoder_dim,
            dist_embed_dim=continuous_encoder_dim,
            negative_slope=cfg.conv_args.negative_slope,
            have_query_feature=False,
            feature_attention_dropout=self.feature_attention_dropout,
            feature_residual_alpha=self.feature_residual_alpha
        )

        # Initialize batch normalization and dropout layers for time filtering
        self.norms_for_time_filter = nn.BatchNorm1d(self.checkin_embed_size)
        self.dropout_for_time_filter = nn.Dropout(self.dropout_rate)

        # If trajectory-to-trajectory convolution is enabled, build convolution layers based on each layer's settings
        if self.do_traj2traj:
            self.conv_list = nn.ModuleList()
            for i in range(self.num_conv_layers):
                if i == 0:
                    # First layer convolution doesn't use query features and residual fusion, traj2traj layer has no query (target features)
                    have_query_feature = False
                    residual_fusion_mode = None
                    edge_size = None
                else:
                    # Subsequent layer convolutions use query features and residual fusion
                    have_query_feature = True
                    residual_fusion_mode = cfg.conv_args.residual_fusion_mode
                    if self.edge_attr_embedding_layer is None:
                        edge_size = 3
                    else:
                        edge_size = self.checkin_embed_size

                # Add hypergraph Transformer layer to convolution list
                self.conv_list.append(
                    HypergraphTransformer(
                        in_channels=self.checkin_embed_size,
                        out_channels=self.checkin_embed_size,
                        attn_heads=cfg.conv_args.num_attention_heads,
                        residual_beta=cfg.conv_args.residual_beta,
                        learn_beta=cfg.conv_args.learn_beta,
                        dropout=cfg.conv_args.conv_dropout_rate,
                        trans_method=cfg.conv_args.trans_method,
                        edge_fusion_mode=cfg.conv_args.edge_fusion_mode,
                        time_fusion_mode=cfg.conv_args.time_fusion_mode,
                        head_fusion_mode=cfg.conv_args.head_fusion_mode,
                        residual_fusion_mode=residual_fusion_mode,
                        edge_dim=edge_size,
                        rel_embed_dim=self.checkin_embed_size,
                        time_embed_dim=continuous_encoder_dim,
                        dist_embed_dim=continuous_encoder_dim,
                        negative_slope=cfg.conv_args.negative_slope,
                        have_query_feature=have_query_feature,
                        feature_attention_dropout=self.feature_attention_dropout,
                        feature_residual_alpha=self.feature_residual_alpha
                    )
                )

            # Batch normalization and dropout layers for post-convolution processing of each layer
            self.norms_list = nn.ModuleList([nn.BatchNorm1d(self.checkin_embed_size) for _ in range(self.num_conv_layers)])
            self.dropout_list = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(self.num_conv_layers)])

        # Initialize time and distance encoders
        self.continuous_time_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim)
        if self.distance_encoder_type == 'stan':
            self.continuous_distance_encoder = DistanceEncoderSTAN(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
        elif self.distance_encoder_type == 'time':
            self.continuous_distance_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim)
        elif self.distance_encoder_type == 'hstlstm':
            self.continuous_distance_encoder = DistanceEncoderHSTLSTM(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
        elif self.distance_encoder_type == 'simple':
            self.continuous_distance_encoder = DistanceEncoderSimple(cfg.model_args, continuous_encoder_dim, cfg.dataset_args.spatial_slots)
        else:
            raise ValueError(f"Get wrong distance_encoder_type argument: {cfg.model_args.distance_encoder_type}!")

        # Output layer and loss function
        self.linear = nn.Linear(self.checkin_embed_size, self.num_poi)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, data, label=None, mode='train'):
        input_x = data['x']  # Get input features with shape [?, 8]
        split_idx = data['split_index']  # Split index

        # Extract checkin features and trajectory features
        check_in_x = input_x[split_idx + 1:]
        checkin_feature = self.checkin_embedding_layer(check_in_x)
        trajectory_feature = torch.zeros(
            split_idx+1,
            self.checkin_embed_size,
            device=checkin_feature.device
        )
        # Try to use original features
        # trajectory_feature = input_x[:split_idx+1]
        # trajectory_feature = self.traj_feature_embedding_layer(input_x[:split_idx+1])
        x = torch.cat([trajectory_feature, checkin_feature], dim=0)

        # Calculate time embedding
        edge_time_embed = self.continuous_time_encoder(data['delta_ts'][0] / (60 * 60))
        
        # Calculate distance embedding
        if self.distance_encoder_type == 'stan':
            edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0], dist_type='ch2tj')
        else:
            edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0])

        # Get edge attribute and edge type embeddings
        edge_attr_embed, edge_type_embed = None, None
        if data['edge_type'][0] is not None:
            if self.generate_edge_attr:
                edge_attr_embed = self.edge_attr_embedding_layer(data['edge_type'][0])
            edge_type_embed = self.edge_type_embedding_layer(data['edge_type'][0])

        # Update features using time-filtered hypergraph Transformer
        x_for_time_filter = self.conv_for_time_filter(
            x,
            edge_index=data['edge_index'][0],   # [filtered_ci2traj,ci2traj,traj2traj]
            edge_attr_embed=edge_attr_embed,
            edge_time_embed=edge_time_embed,
            edge_dist_embed=edge_distance_embed,
            edge_type_embed=edge_type_embed
        )
        x_for_time_filter = self.norms_for_time_filter(x_for_time_filter)
        x_for_time_filter = self.act(x_for_time_filter)
        x_for_time_filter = self.dropout_for_time_filter(x_for_time_filter)

        # Perform trajectory-to-trajectory convolution ci2traj, traj2traj
        if data['edge_index'][-1] is not None and self.do_traj2traj:
            # all conv
            for idx, (edge_index, edge_attr, delta_ts, delta_dis, edge_type) in enumerate(
                    zip(data["edge_index"][1:], data["edge_attr"][1:], data["delta_ts"][1:], data["delta_ss"][1:],
                        data["edge_type"][1:])
            ):
                # Calculate edge time and distance embeddings
                edge_time_embed = self.continuous_time_encoder(delta_ts / (60 * 60))
                if self.distance_encoder_type == 'stan':
                    edge_distance_embed = self.continuous_distance_encoder(delta_dis, dist_type='tj2tj')
                else:
                    edge_distance_embed = self.continuous_distance_encoder(delta_dis)

                # Edge attribute and edge type embeddings
                edge_attr_embed, edge_type_embed = None, None
                if edge_type is not None:
                    edge_type_embed = self.edge_type_embedding_layer(edge_type)
                    if self.generate_edge_attr:
                        edge_attr_embed = self.edge_attr_embedding_layer(edge_type)
                    else:
                        if self.edge_attr_embedding_layer:
                            edge_attr_embed = self.edge_attr_embedding_layer(edge_attr.to(torch.float32))
                        else:
                            edge_attr_embed = edge_attr.to(torch.float32)

                # Select target features based on current mode and layer number
                if idx == len(data['edge_index']) - 2:  # If idx=1, i.e., traj2traj layer, target features are checkin features
                    batch_size = self.eval_batch_size if mode in ('test', 'validate') else self.batch_size
                    x_target = x_for_time_filter[:batch_size]
                else:   # If it's other ci2traj layers, target features are trajectory features, which are all zeros
                    x_target = x[:edge_index.sparse_sizes()[0]]  # traj2traj.sparse_sizes()[0] = traj2traj.shape[0]
                    # Check if this is an all-zero matrix
                    # if torch.all(x_target == 0):
                    #     print(f"idx: {idx}, x_target is all zero matrix")

                # print(f"start of idx: {idx}")
                # Update features through trajectory convolution layer
                x = self.conv_list[idx](
                    (x, x_target),
                    edge_index=edge_index,
                    edge_attr_embed=edge_attr_embed,
                    edge_time_embed=edge_time_embed,
                    edge_dist_embed=edge_distance_embed,
                    edge_type_embed=edge_type_embed
                )
                x = self.norms_list[idx](x)
                x = self.act(x)
                x = self.dropout_list[idx](x)
                # print(f"end of idx: {idx}")
        else:
            x = x_for_time_filter  # If not performing trajectory convolution, directly use time-filtered features

        # Output layer calculates classification logits and loss
        logits = self.linear(x)
        loss = self.loss_func(logits, label.long())
        return logits, loss

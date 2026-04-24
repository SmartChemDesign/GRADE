"""DOSpredict model with s,p,d,f separate decoders."""

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d

from torch_geometric.nn import GATv2Conv, TransformerConv

from .kan import KANLinear
from .gc_block import GC_block


def model_summary(model):
    """Print model summary."""
    model_params_list = list(model.named_parameters())
    print("-" * 74)
    line_new = "{:>30}  {:>20} {:>20}".format(
        "Layer.Parameter", "Param Tensor Shape", "Param #"
    )
    print(line_new)
    print("-" * 74)
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>30}  {:>20} {:>20}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("-" * 74)
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)


class DOSpredict(torch.nn.Module):
    """DOSpredict model with s,p,d,f separate decoders."""
    
    def __init__(
        self,
        data,
        dim1=64,
        dim2=64,
        pre_fc_count=1,
        gc_count=3,
        graph_conv_type="MessagePassing",
        batch_norm="True",
        batch_track_stats="True",
        basis_expansion="False",
        num_func=20,
        perceptron_type="Regular",
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        # Coefficients KAN params
        grid_size_c=5,
        spline_order_c=3,
        scale_noise_c=0.1,
        scale_base_c=1.0,
        scale_spline_c=1.0,
        enable_standalone_scale_spline_c=True,
        base_activation_c=torch.nn.SiLU,
        grid_eps_c=0.02,
        grid_range_c=[-1, 1],
        # Mean KAN params
        grid_size_m=5,
        spline_order_m=3,
        scale_noise_m=0.1,
        scale_base_m=1.0,
        scale_spline_m=1.0,
        enable_standalone_scale_spline_m=True,
        base_activation_m=torch.nn.SiLU,
        grid_eps_m=0.02,
        grid_range_m=[-1, 1],
        # Dispersion KAN params
        grid_size_d=5,
        spline_order_d=3,
        scale_noise_d=0.1,
        scale_base_d=1.0,
        scale_spline_d=1.0,
        enable_standalone_scale_spline_d=True,
        base_activation_d=torch.nn.SiLU,
        grid_eps_d=0.02,
        grid_range_d=[-1, 1],
        # Scaling KAN params
        grid_size_w=5,
        spline_order_w=3,
        scale_noise_w=0.1,
        scale_base_w=1.0,
        scale_spline_w=1.0,
        enable_standalone_scale_spline_w=True,
        base_activation_w=torch.nn.SiLU,
        grid_eps_w=0.02,
        grid_range_w=[-1, 1],
        dropout_rate=0.0,
        **kwargs
    ):
        super(DOSpredict, self).__init__()

        self.batch_track_stats = batch_track_stats != "False"
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.graph_conv_type = graph_conv_type
        
        print("Type of graph convolution:", graph_conv_type)
        print("Number of features:", data.num_features)

        # Determine gc dimension
        assert gc_count > 0, "Need at least 1 GC layer"
        if pre_fc_count == 0:
            self.gc_dim = data.num_features
        else:
            self.gc_dim = dim1
        
        # Determine post_fc dimension
        post_fc_dim = data.num_features if pre_fc_count == 0 else dim1
        output_dim = 400
        
        print("Expansion size:", num_func)

        if basis_expansion == "True":
            print("Basis expansion will be predicted by neural network")
            self.basis_expansion = True
            self.num_func = num_func
            self.output_dim_basis = self.num_func
        else:
            print("Grid values will be predicted by neural network")
            self.basis_expansion = False

        # Perceptron type setup
        if perceptron_type == "Regular":
            self.p_type = "Regular"
            print("Regular perceptron used to construct neural network")
        if perceptron_type == "KAN":
            self.p_type = "KAN"
            print("KAN perceptron used to construct neural network")
            self.grid_size = grid_size
            self.spline_order = spline_order
            self.scale_noise = scale_noise
            self.scale_base = scale_base
            self.scale_spline = scale_spline
            self.enable_standalone_scale_spline = enable_standalone_scale_spline
            self.base_activation = base_activation
            self.grid_eps = grid_eps
            self.grid_range = grid_range
            print("KAN Grid size:", self.grid_size)

        if perceptron_type == "KAN" and basis_expansion == "True":
            # Store all KAN params for basis expansion
            self._store_kan_params(
                grid_size_w, spline_order_w, scale_noise_w, scale_base_w,
                scale_spline_w, enable_standalone_scale_spline_w,
                base_activation_w, grid_eps_w, grid_range_w,
                grid_size_c, spline_order_c, scale_noise_c, scale_base_c,
                scale_spline_c, enable_standalone_scale_spline_c,
                base_activation_c, grid_eps_c, grid_range_c,
                grid_size_m, spline_order_m, scale_noise_m, scale_base_m,
                scale_spline_m, enable_standalone_scale_spline_m,
                base_activation_m, grid_eps_m, grid_range_m,
                grid_size_d, spline_order_d, scale_noise_d, scale_base_d,
                scale_spline_d, enable_standalone_scale_spline_d,
                base_activation_d, grid_eps_d, grid_range_d
            )

        # Build pre-GNN dense linear layers
        if pre_fc_count > 0:
            self.pre_lin_list = torch.nn.ModuleList()
            for i in range(pre_fc_count):
                if i == 0:
                    lin = Sequential(
                        torch.nn.Linear(data.num_features, dim1), 
                        torch.nn.PReLU()
                    )
                else:
                    lin = Sequential(
                        torch.nn.Linear(dim1, dim1), 
                        torch.nn.PReLU()
                    )
                self.pre_lin_list.append(lin)
        else:
            self.pre_lin_list = torch.nn.ModuleList()

        # Build GNN layers
        self.conv_list = torch.nn.ModuleList()
        self.conv_list1 = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        self.bn_list1 = torch.nn.ModuleList()
        
        for j in range(gc_count):
            conv = GC_block(self.gc_dim, data.num_edge_features, aggr="mean")
            self.conv_list1.append(conv)
        
        for i in range(gc_count):
            if self.graph_conv_type == "MessagePassing":
                conv = GC_block(self.gc_dim, data.num_edge_features, aggr="mean")
            elif self.graph_conv_type == "GATV2":
                conv = GATv2Conv(
                    in_channels=self.gc_dim,
                    out_channels=self.gc_dim, 
                    heads=1, 
                    add_self_loops=True,
                    edge_dim=data.num_edge_features, 
                    residual=True
                )
            elif self.graph_conv_type == "Transformer":
                conv = TransformerConv(
                    in_channels=self.gc_dim,
                    out_channels=self.gc_dim, 
                    heads=1,
                    edge_dim=data.num_edge_features
                )
            self.conv_list.append(conv)
            
            if self.batch_norm == "True":
                bn = BatchNorm1d(
                    self.gc_dim, 
                    track_running_stats=self.batch_track_stats, 
                    affine=True
                )
                bn1 = BatchNorm1d(
                    self.gc_dim, 
                    track_running_stats=self.batch_track_stats, 
                    affine=True
                )
                self.bn_list.append(bn)
                self.bn_list1.append(bn1)

        # Build decoder layers
        self._build_decoders(post_fc_dim, dim2, output_dim)

    def _store_kan_params(self, *args):
        """Store KAN parameters for basis expansion."""
        (self.grid_size_w, self.spline_order_w, self.scale_noise_w, 
         self.scale_base_w, self.scale_spline_w, self.enable_standalone_scale_spline_w,
         self.base_activation_w, self.grid_eps_w, self.grid_range_w,
         self.grid_size_c, self.spline_order_c, self.scale_noise_c, 
         self.scale_base_c, self.scale_spline_c, self.enable_standalone_scale_spline_c,
         self.base_activation_c, self.grid_eps_c, self.grid_range_c,
         self.grid_size_m, self.spline_order_m, self.scale_noise_m, 
         self.scale_base_m, self.scale_spline_m, self.enable_standalone_scale_spline_m,
         self.base_activation_m, self.grid_eps_m, self.grid_range_m,
         self.grid_size_d, self.spline_order_d, self.scale_noise_d, 
         self.scale_base_d, self.scale_spline_d, self.enable_standalone_scale_spline_d,
         self.base_activation_d, self.grid_eps_d, self.grid_range_d) = args

    def _build_decoders(self, post_fc_dim, dim2, output_dim):
        """Build decoder layers for s,p,d,f components."""
        if not self.basis_expansion:
            self._build_grid_decoders(post_fc_dim, dim2, output_dim)
        else:
            self._build_basis_decoders(post_fc_dim, dim2)

    def _build_grid_decoders(self, post_fc_dim, dim2, output_dim):
        """Build decoders for grid-based prediction."""
        if self.p_type == "Regular":
            for comp in ['s', 'p', 'd', 'f']:
                setattr(self, f'dos_mlp_{comp}', Sequential(
                    Linear(post_fc_dim, dim2),
                    torch.nn.ReLU(),
                    Linear(dim2, output_dim),
                    torch.nn.ReLU(),
                ))
                setattr(self, f'scaling_mlp_{comp}', Sequential(
                    Linear(post_fc_dim, dim2),
                    torch.nn.ReLU(),
                    Linear(dim2, 1),
                ))
        else:  # KAN
            for comp in ['s', 'p', 'd', 'f']:
                lin1 = KANLinear(
                    post_fc_dim, output_dim,
                    grid_size=self.grid_size, spline_order=self.spline_order,
                    scale_noise=self.scale_noise, scale_base=self.scale_base,
                    scale_spline=self.scale_spline, base_activation=self.base_activation,
                    grid_eps=self.grid_eps, grid_range=self.grid_range,
                )
                lin3 = KANLinear(
                    post_fc_dim, 1,
                    grid_size=self.grid_size, spline_order=self.spline_order,
                    scale_noise=self.scale_noise, scale_base=self.scale_base,
                    scale_spline=self.scale_spline, base_activation=self.base_activation,
                    grid_eps=self.grid_eps, grid_range=self.grid_range,
                )
                setattr(self, f'dos_mlp_{comp}', Sequential(lin1))
                setattr(self, f'scaling_mlp_{comp}', Sequential(lin3))
            print("KAN layers used in output head")

    def _build_basis_decoders(self, post_fc_dim, dim2):
        """Build decoders for basis expansion prediction."""
        if self.p_type == "Regular":
            for comp in ['s', 'p', 'd', 'f']:
                setattr(self, f'dos_mlp_basis_coef_{comp}', Sequential(
                    Linear(post_fc_dim, dim2),
                    torch.nn.PReLU(),
                    Linear(dim2, self.output_dim_basis),
                ))
                setattr(self, f'dos_mlp_basis_mean_{comp}', Sequential(
                    Linear(post_fc_dim, dim2),
                    torch.nn.PReLU(),
                    Linear(dim2, self.output_dim_basis),
                ))
                setattr(self, f'dos_mlp_basis_sigma_{comp}', Sequential(
                    Linear(post_fc_dim, dim2),
                    torch.nn.PReLU(),
                    Linear(dim2, self.output_dim_basis),
                    torch.nn.ReLU(),
                ))
                setattr(self, f'scaling_mlp_{comp}', Sequential(
                    Linear(post_fc_dim, dim2),
                    torch.nn.ReLU(),
                    Linear(dim2, 1),
                ))
        else:  # KAN
            for comp in ['s', 'p', 'd', 'f']:
                lin_coef = KANLinear(
                    post_fc_dim, self.output_dim_basis,
                    grid_size=self.grid_size_c, spline_order=self.spline_order_c,
                    scale_noise=self.scale_noise_c, scale_base=self.scale_base_c,
                    scale_spline=self.scale_spline_c, base_activation=self.base_activation_c,
                    grid_eps=self.grid_eps_c, grid_range=self.grid_range_c,
                )
                lin_mean = KANLinear(
                    post_fc_dim, self.output_dim_basis,
                    grid_size=self.grid_size_m, spline_order=self.spline_order_m,
                    scale_noise=self.scale_noise_m, scale_base=self.scale_base_m,
                    scale_spline=self.scale_spline_m, base_activation=self.base_activation_m,
                    grid_eps=self.grid_eps_m, grid_range=self.grid_range_m,
                )
                lin_sigma = KANLinear(
                    post_fc_dim, self.output_dim_basis,
                    grid_size=self.grid_size_d, spline_order=self.spline_order_d,
                    scale_noise=self.scale_noise_d, scale_base=self.scale_base_d,
                    scale_spline=self.scale_spline_d, base_activation=self.base_activation_d,
                    grid_eps=self.grid_eps_d, grid_range=self.grid_range_d,
                )
                lin_scaling = KANLinear(
                    post_fc_dim, 1,
                    grid_size=self.grid_size_w, spline_order=self.spline_order_w,
                    scale_noise=self.scale_noise_w, scale_base=self.scale_base_w,
                    scale_spline=self.scale_spline_w, base_activation=self.base_activation_w,
                    grid_eps=self.grid_eps_w, grid_range=self.grid_range_w,
                )
                setattr(self, f'dos_mlp_basis_coef_{comp}', Sequential(lin_coef))
                setattr(self, f'dos_mlp_basis_mean_{comp}', Sequential(lin_mean))
                setattr(self, f'dos_mlp_basis_sigma_{comp}', Sequential(lin_sigma))
                setattr(self, f'scaling_mlp_{comp}', Sequential(lin_scaling))
            print("KAN layers used in output heads")

    def forward(self, data):
        """Forward pass of the model."""
        # Pre-GNN dense layers
        for i in range(len(self.pre_lin_list)):
            if i == 0:
                out = self.pre_lin_list[i](data.x)
            else:
                out = self.pre_lin_list[i](out)
        
        # GNN layers
        for i in range(len(self.conv_list)):
            if len(self.pre_lin_list) == 0 and i == 0:
                if self.batch_norm == "True":
                    if self.graph_conv_type != "Sage":
                        out = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
                        out = self.bn_list[i](out)
                    else:
                        out = self.conv_list[i](data.x, data.edge_index)
                        out = self.bn_list[i](out)
                else:
                    if self.graph_conv_type != "Sage":
                        out = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
                    else:
                        out = self.conv_list[i](data.x, data.edge_index)
            else:
                if self.batch_norm == "True":
                    if self.graph_conv_type != "Sage":
                        out = self.conv_list[i](out, data.edge_index, data.edge_attr)
                        out = self.bn_list[i](out)
                    else:
                        out = self.conv_list[i](out, data.edge_index)
                        out = self.bn_list[i](out)
                else:
                    if self.graph_conv_type != "Sage":
                        out = self.conv_list[i](out, data.edge_index, data.edge_attr)
                    else:
                        out = self.conv_list[i](out, data.edge_index)

        # Extra convolution layers
        for i in range(len(self.conv_list1)):
            if self.batch_norm == "True":
                out = self.conv_list1[i](out, data.edge_index, data.edge_attr)
                out = self.bn_list1[i](out)
            else:
                out = self.conv_list1[i](out, data.edge_index, data.edge_attr)

        out = F.dropout(out, p=self.dropout_rate, training=self.training)

        # Decoders
        if not self.basis_expansion:
            return self._forward_grid(out)
        else:
            return self._forward_basis(out)

    def _forward_grid(self, out):
        """Forward pass for grid-based prediction."""
        results = []
        for comp in ['s', 'p', 'd', 'f']:
            dos_out = getattr(self, f'dos_mlp_{comp}')(out)
            scaling = getattr(self, f'scaling_mlp_{comp}')(out)
            results.extend([dos_out, scaling.view(-1)])
        
        if results[0].shape[1] == 1:
            return tuple(r.view(-1) if i % 2 == 0 else r for i, r in enumerate(results))
        return tuple(results)

    def _forward_basis(self, out):
        """Forward pass for basis expansion prediction."""
        results = []
        for comp in ['s', 'p', 'd', 'f']:
            coef = getattr(self, f'dos_mlp_basis_coef_{comp}')(out)
            mean = getattr(self, f'dos_mlp_basis_mean_{comp}')(out)
            sigma = getattr(self, f'dos_mlp_basis_sigma_{comp}')(out)
            scaling = getattr(self, f'scaling_mlp_{comp}')(out)
            results.extend([coef, mean, sigma, scaling.view(-1)])
        
        if results[0].shape[1] == 1:
            return tuple(r.view(-1) if i % 4 != 3 else r for i, r in enumerate(results))
        return tuple(results)


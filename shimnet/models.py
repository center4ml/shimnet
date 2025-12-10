import torch

# class ConvEncoder(torch.nn.Module):
#     def __init__(self, hidden_dim=64, output_dim=None, dropout=0, kernel_size=7):
#         super().__init__()
#         if output_dim is None:
#             output_dim = hidden_dim
#         self.conv4 = torch.nn.Conv1d(1, hidden_dim, kernel_size)
#         self.conv3 = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size)
#         self.conv2 = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size)
#         self.conv1 = torch.nn.Conv1d(hidden_dim, output_dim, kernel_size)
#         self.dropout = torch.nn.Dropout(dropout)

#     def forward(self, feature):                                        #(samples, 1, 2048)
#         feature = self.dropout(self.conv4(feature))                    #(samples, 64, 2042)
#         feature = feature.relu()
#         feature = self.dropout(self.conv3(feature))                    #(samples, 64, 2036)
#         feature = feature.relu()
#         feature = self.dropout(self.conv2(feature))                    #(samples, 64, 2030)
#         feature = feature.relu()
#         feature = self.dropout(self.conv1(feature))                    #(samples, 64, 2024)
#         return feature

# class ConvDecoder(torch.nn.Module):
#     def __init__(self, input_dim=None, hidden_dim=64, output_dim=None, dropout=0, kernel_size=7):
#         super().__init__()
#         if output_dim is None:
#             output_dim = hidden_dim
#         self.convTranspose1 = torch.nn.ConvTranspose1d(input_dim, hidden_dim, kernel_size)
#         self.convTranspose2 = torch.nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size)
#         self.convTranspose3 = torch.nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size)
#         self.convTranspose4 = torch.nn.ConvTranspose1d(hidden_dim, 1, kernel_size)
    
#     def forward(self, feature):                                        #(samples, 1, 2048)
#         feature = self.convTranspose1(feature)                         #(samples,  64, 2030)
#         feature = feature.relu()
#         feature = self.convTranspose2(feature)                         #(samples,  64, 2036)
#         feature = feature.relu()
#         feature = self.convTranspose3(feature)                         #(samples,  64, 2042)
#         feature = feature.relu()
#         feature = self.convTranspose4(feature) 
#         return feature
def get_activation(activation_name: str) -> torch.nn.Module:
    if activation_name == "relu":
        return torch.nn.ReLU()
    elif activation_name == "gelu":
        return torch.nn.GELU()
    elif activation_name == "leaky_relu":
        return torch.nn.LeakyReLU()
    elif activation_name == "tanh":
        return torch.nn.Tanh()
    elif activation_name == "sigmoid":
        return torch.nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")


class ConvEncoder(torch.nn.Module):
    def __init__(self, hidden_dim=64, output_dim=None, input_dim=1, dropout=0, kernel_size=7, activation="relu"):
        super().__init__()
        if output_dim is None:
            output_dim = hidden_dim
        layers = [
            torch.nn.Conv1d(input_dim, hidden_dim, kernel_size),
            get_activation(activation),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size),
            get_activation(activation),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size),
            get_activation(activation),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(hidden_dim, output_dim, kernel_size),
            get_activation(activation),
            torch.nn.Dropout(dropout),
        ]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, feature):
        return self.net(feature)

class ConvDecoder(torch.nn.Module):
    def __init__(self, input_dim=None, hidden_dim=64, output_dim=1, dropout=0, kernel_size=7, activation="relu", last_bias=True, last_activation=True):
        super().__init__()
        if input_dim is None:
            input_dim = hidden_dim
        layers = [
            torch.nn.ConvTranspose1d(input_dim, hidden_dim, kernel_size),
            get_activation(activation),
            torch.nn.Dropout(dropout),
            torch.nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size),
            get_activation(activation),
            torch.nn.Dropout(dropout),
            torch.nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size),
            get_activation(activation),
            torch.nn.Dropout(dropout),
            torch.nn.ConvTranspose1d(hidden_dim, output_dim, kernel_size, bias=last_bias),
        ]
        if last_activation:
            layers.append(get_activation(activation))
            layers.append(torch.nn.Dropout(dropout))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, feature):                                              
        return self.net(feature)   

class ConvMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64], activation="relu"):
        super().__init__()
        mlp_dims = [input_dim] + hidden_dims + [output_dim]
        mlp_layers = [torch.nn.Conv1d(mlp_dims[0], mlp_dims[1], kernel_size=1)]
        for dims_in, dims_out in zip(mlp_dims[1:-1], mlp_dims[2:]):
            mlp_layers.extend([
                get_activation(activation),
                torch.nn.Conv1d(dims_in, dims_out, kernel_size=1)
            ])
        self.mlp = torch.nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.mlp(x)

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64], activation="relu"):
        super().__init__()
        mlp_dims = [input_dim] + hidden_dims + [output_dim]
        mlp_layers = [torch.nn.Linear(mlp_dims[0], mlp_dims[1])]
        for dims_in, dims_out in zip(mlp_dims[1:-1], mlp_dims[2:]):
            mlp_layers.extend([
                get_activation(activation),
                torch.nn.Linear(dims_in, dims_out)
            ])
        self.mlp = torch.nn.Sequential(*mlp_layers)
    
    def forward(self, x):
        return self.mlp(x)

class ResponseHead(torch.nn.Module):
    def __init__(self, input_dim, output_length, hidden_dims=[128]):
        super().__init__()
        response_head_dims = [input_dim]+hidden_dims + [output_length]
        response_head_layers = [torch.nn.Linear(response_head_dims[0], response_head_dims[1])]
        for dims_in, dims_out in zip(response_head_dims[1:-1], response_head_dims[2:]):
            response_head_layers.extend([
                torch.nn.GELU(),
                torch.nn.Linear(dims_in, dims_out)
            ])
        self.response_head = torch.nn.Sequential(*response_head_layers)

    def forward(self, feature):
        return self.response_head(feature)

class ShimNetWithSCRF(torch.nn.Module):
    def __init__(self,
        encoder_hidden_dims=64,
        encoder_dropout=0,
        bottleneck_dim=64,
        rensponse_length=61,
        resnponse_head_dims=[128],
        decoder_hidden_dims=64
        ):
        super().__init__()
        self.encoder = ConvEncoder(hidden_dim=encoder_hidden_dims, output_dim=bottleneck_dim, dropout=encoder_dropout)
        self.query = torch.nn.Parameter(torch.empty(1, 1, bottleneck_dim))
        torch.nn.init.xavier_normal_(self.query)

        self.decoder = ConvDecoder(input_dim=2*bottleneck_dim, hidden_dim=decoder_hidden_dims)
        
        self.rensponse_length = rensponse_length
        self.response_head = ResponseHead(bottleneck_dim, rensponse_length, resnponse_head_dims)
        
    def forward(self, feature):                                        #(samples,   1, 2048)
        feature = self.encoder(feature)                                #(samples,  64, 2042)
        energy = self.query @ feature                                  #(samples,   1, 2024)
        weight = torch.nn.functional.softmax(energy, 2)                #(samples,   1, 2024)
        global_features = feature @ weight.transpose(1, 2)                    #(samples,  64,    1)
        
        response = self.response_head(global_features.squeeze(-1))
        
        feature, global_features = torch.broadcast_tensors(feature, global_features) #(samples,  64, 2048)
        feature = torch.cat([feature, global_features], 1)                    #(samples, 128, 2024)
        denoised_spectrum = self.decoder(feature)                            #(samples, 1, 2048)
        
        return {
            'denoised': denoised_spectrum,
            'response': response,
            'attention': weight.squeeze(1)
        }

class KVAttention(torch.nn.Module):
    """attention with learnable query"""
    def __init__(self,
        kv_dim =64,
        num_heads=4,
        k_processor = None,
        v_processor = None,
    ):
        super().__init__()
        if k_processor is None:
            k_processor = torch.nn.Identity()
        if v_processor is None:
            v_processor = torch.nn.Identity()
        self.k_processor = k_processor
        self.v_processor = v_processor

        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.query = torch.nn.Parameter(torch.empty(1, num_heads, kv_dim))
        torch.nn.init.xavier_normal_(self.query)
    
    def forward(self, feature): # (samples, input_dim, seq_len)
        batch_size = feature.shape[0]
        seq_len = feature.shape[-1]
        keys = self.k_processor(feature)
        values = feature

        # Reshape for multi-head attention
        keys = keys.view(batch_size, self.num_heads, self.kv_dim, seq_len)     #(samples, num_heads, kv_dim, seq_len)
        
        # Multi-head attention computation
        queries = self.query.expand(batch_size, -1, -1)              #(samples, num_heads, kv_dim)
        energy = torch.einsum('bhd,bhdl->bhl', queries, keys)          #(samples, num_heads, seq_len)
        weight = torch.nn.functional.softmax(energy, dim=2)            #(samples, num_heads, seq_len)
        
        # Apply attention weights
        global_features = torch.einsum('bhl,bhdl->bhd', weight, feature.view(batch_size, self.num_heads, -1, seq_len))  #(samples, (num_heads* head_dim))
        global_features = global_features.reshape(batch_size, -1)  #(samples, (num_heads* head_dim))
        
        # process values if needed
        global_features = self.v_processor(global_features) # (samples, input_dim)
        # global_features = global_features.reshape(batch_size, -1, 1)  
        
        return global_features, weight
        

class ShimnetModular(torch.nn.Module):
    def __init__(self,
        encoder,
        decoder,
        response_head,
        attention_module,
        local_feature_processor,
        global_feature_processor
        ):
        super().__init__()
        self.encoder = encoder
        self.attention_module = attention_module
        self.decoder = decoder
        self.response_head = response_head
        self.local_feature_processor = local_feature_processor
        self.global_feature_processor = global_feature_processor
        
    def forward(self, feature):                                        #(samples,   1, seq_len_in)
        feature = self.encoder(feature)                                #(samples,  encoder_features_dim, seq_len) # seq_len != seq_len_in
        local_features = self.local_feature_processor(feature)         #(samples,  local_features_dim, seq_len)
        
        global_features, weight = self.attention_module(feature)      #(samples,  global_features_hidden_dim,    1), (samples, num_heads, seq_len)
        
        response = self.response_head(global_features.squeeze(-1))  # (samples, response_length)

        global_features_for_decoding = self.global_feature_processor(global_features).unsqueeze(-1)  #(samples, global_features_dim, 1)
        
        local_features, global_features_for_decoding = torch.broadcast_tensors(local_features, global_features_for_decoding)  #(samples,  local_features_dim, seq_len), (samples,  global_features_dim, seq_len)
        feature = torch.cat([local_features, global_features_for_decoding], 1)    #(samples, local_features_dim + global_features_dim, seq_len)
        denoised_spectrum = self.decoder(feature)                    #(samples, 1, seq_len_in)
        
        return {
            'denoised': denoised_spectrum,
            'response': response,
            'attention': weight.sum(1)  # (samples, seq_len)
        }

class Predictor:
    def __init__(self, model=None, weights_file=None):
        self.model = model
        if weights_file is not None:
            self.model.load_state_dict(torch.load(weights_file, map_location='cpu', weights_only=True))

    def __call__(self, nsf_frq):
        with torch.no_grad():
            msf_frq = self.model(nsf_frq[None, None])["denoised"]
        return msf_frq[0, 0]

if __name__ == "__main__":
    encoder_hidden_dims = 64
    encoder_dropout = 0
    encoder_features_dim = 128

    local_features_dim = 64

    attention_kv_dim = 32
    attention_num_heads = 8
    global_features_hidden_dim = 256

    global_features_dim = 64
    response_length = 81


    encoder = ConvEncoder(hidden_dim=encoder_hidden_dims, output_dim=encoder_features_dim, dropout=encoder_dropout)
    local_feature_processor = ConvMLP(encoder_features_dim, local_features_dim, hidden_dims=[256, 128])
    attention = KVAttention(
        kv_dim=attention_kv_dim, num_heads=attention_num_heads,
        k_processor = ConvMLP(encoder_features_dim, attention_kv_dim*attention_num_heads, hidden_dims=[512, 256]),
        v_processor = MLP(encoder_features_dim, global_features_hidden_dim, hidden_dims=[512, 256]),
    )
    global_feature_processor = MLP(global_features_hidden_dim, global_features_dim, hidden_dims=[512, 256])
    response_head = MLP(global_features_hidden_dim, response_length, hidden_dims=[512, 256])

    decoder = ConvDecoder(input_dim=local_features_dim + global_features_dim, hidden_dim=64)


    ### step by step
    inputs = torch.randn(2, 1, 2048)

    feature = encoder(inputs)                                #(samples,  encoder_features_dim, seq_len) # seq_len != seq_len_in
    print(f"Encoder output shape: {feature.shape}")

    local_features = local_feature_processor(feature)         #(samples,  local_features_dim, seq_len)
    print(f"Local features shape: {local_features.shape}")

    global_features, weight = attention(feature)      #(samples,  global_features_hidden_dim,    1), (samples, num_heads, seq_len)
    print(f"Global features shape: {global_features.shape}")
    print(f"Attention weights shape: {weight.shape}")

    response = response_head(global_features)  # (samples, response_length)
    print(f"Response shape: {response.shape}")

    global_features_for_decoding = global_feature_processor(global_features).unsqueeze(-1)  #(samples, global_features_dim, 1)

    local_features, global_features_for_decoding = torch.broadcast_tensors(local_features, global_features_for_decoding)  #(samples,  local_features_dim, seq_len), (samples,  global_features_dim, seq_len)
    feature = torch.cat([local_features, global_features_for_decoding], 1)    #(samples, local_features_dim + global_features_dim, seq_len)
    denoised_spectrum = decoder(feature)       

    print("="*80)
    ### assemble model

    model = ShimnetModular(
        encoder=encoder,
        decoder=decoder,
        response_head=response_head,
        attention_module=attention,
        local_feature_processor=local_feature_processor,
        global_feature_processor=global_feature_processor
    )

    for k, v in model(inputs).items():
        print(f"{k}: {v.shape}")
import torch

class ConvEncoder(torch.nn.Module):
    def __init__(self, hidden_dim=64, output_dim=None, dropout=0, kernel_size=7):
        super().__init__()
        if output_dim is None:
            output_dim = hidden_dim
        self.conv4 = torch.nn.Conv1d(1, hidden_dim, kernel_size)
        self.conv3 = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size)
        self.conv2 = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size)
        self.conv1 = torch.nn.Conv1d(hidden_dim, output_dim, kernel_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, feature):                                        #(samples, 1, 2048)
        feature = self.dropout(self.conv4(feature))                    #(samples, 64, 2042)
        feature = feature.relu()
        feature = self.dropout(self.conv3(feature))                    #(samples, 64, 2036)
        feature = feature.relu()
        feature = self.dropout(self.conv2(feature))                    #(samples, 64, 2030)
        feature = feature.relu()
        feature = self.dropout(self.conv1(feature))                    #(samples, 64, 2024)
        return feature

class ConvDecoder(torch.nn.Module):
    def __init__(self, input_dim=None, hidden_dim=64, output_dim=None, dropout=0, kernel_size=7):
        super().__init__()
        if output_dim is None:
            output_dim = hidden_dim
        self.convTranspose1 = torch.nn.ConvTranspose1d(input_dim, hidden_dim, kernel_size)
        self.convTranspose2 = torch.nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size)
        self.convTranspose3 = torch.nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size)
        self.convTranspose4 = torch.nn.ConvTranspose1d(hidden_dim, 1, kernel_size)
    
    def forward(self, feature):                                        #(samples, 1, 2048)
        feature = self.convTranspose1(feature)                         #(samples,  64, 2030)
        feature = feature.relu()
        feature = self.convTranspose2(feature)                         #(samples,  64, 2036)
        feature = feature.relu()
        feature = self.convTranspose3(feature)                         #(samples,  64, 2042)
        feature = feature.relu()
        feature = self.convTranspose4(feature) 
        return feature

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

class Predictor:
    def __init__(self, model=None, weights_file=None):
        self.model = model
        if weights_file is not None:
            self.model.load_state_dict(torch.load(weights_file, map_location='cpu', weights_only=True))

    def __call__(self, nsf_frq):
        with torch.no_grad():
            msf_frq = self.model(nsf_frq[None, None])["denoised"]
        return msf_frq[0, 0]

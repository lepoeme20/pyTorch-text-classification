import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        data_size = args.data_size
        embedding_dim = args.embed_dim
        embedding_num = args.embed_num
        class_number = args.class_num
        in_channel = 1 #args.in_channel
        out_channel = args.kernel_num
        kernel_sizes = args.kernel_sizes

        self.embed = nn.Embedding(embedding_num+1, embedding_dim)
        self.conv = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, embedding_dim)) for K in kernel_sizes])

        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(kernel_sizes) * out_channel, class_number)


    def forward(self, input_x):
        """
        :param input_x: a list size having the number of batch_size elements with the same length
        :return: batch_size X num_aspects tensor
        """
        # Embedding
        x = self.embed(input_x)  # dim: (batch_size, max_seq_len, embedding_size)

        if self.args.static:
            x = F.Variable(input_x)

        # Conv & max pool
        x = x.unsqueeze(1)  # dim: (batch_size, 1, max_seq_len, embedding_size)

        # turns to be a list: [ti : i \in kernel_sizes] where ti: tensor of dim([batch, num_kernels, max_seq_len-i+1])
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]

        # dim: [(batch_size, num_kernels), ...]*len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        # Dropout & output
        x = self.dropout(x)  # (batch_size,len(kernel_sizes)*num_kernels)
        logit = F.softmax(self.fc(x))  # (batch_size, num_aspects)

        return logit

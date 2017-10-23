import sys
import os
sys.path.append(os.getcwd()+'/Yoon')
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import argparse
import datetime
import data_helpers
from text_cnn import TextCNN


parser = argparse.ArgumentParser(description='CNN text classificer')

# Model Hyperparameters
parser.add_argument('-lr', type=float, default=1e-5, help='setting learning rate')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')

# Training parameters
parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 64]')
parser.add_argument('-num-epochs', type=int, default=50, help='number of epochs for train [default: 200]')
parser.add_argument('-dev-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
parser.add_argument('-list4ES', type=list,  default=[], help='Empty list for appending dev-acc')

# Data Set
parser.add_argument('-json-path', type=str, default="./data/amazon/Video_Games_5.json", help='Data source')
parser.add_argument('-vocab-size', type=int, default=0 , help='Vocab size')
parser.add_argument('-max-len', type=int, default=0 , help='max length among all of sentences')
parser.add_argument('-data-size', type=int, default=0, help='Data size')
parser.add_argument('-class-num', type=int, default=2, help='Number of classes')
parser.add_argument('-trn-sample-percentage', type=float, default=.5, help='Percentage of the data to use for training')
parser.add_argument('-dev-sample-percentage', type=float, default=.2, help='Percentage of the data to use for validation')
parser.add_argument('-test-sample-percentage', type=float, default=.3, help='Percentage of the data to use for testing')
parser.add_argument('-target-num', type=int, default=2, help='Number of classification target')

# saver
parser.add_argument('-iter', type=int, default=0, help='For checking iteration')
parser.add_argument('-save-dir', type=str, default='../RUNS/', help='Data size')
parser.add_argument('-saved-model', type=str, default=None, help='Saved model [default: None]')
args, unknown = parser.parse_known_args()


print("Loading data...")
x_text, y = data_helpers.load_json(args.json_path)
max_len = data_helpers.max_len(x_text)
x, vocab_dic = data_helpers.word2idx(x_text, max_len)
x = data_helpers.fill_zeros(x, max_len)
print("list to array...")
x = np.array(x)
y = np.array(y)


# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]


# Split train/test set
# TODO: This is very crude, should use cross-validation
trn_sample_index = -1 * int(args.trn_sample_percentage * float(len(y)))
test_sample_index = -1 * int(args.test_sample_percentage * float(len(y)))
x_train, x_dev, x_test = x_shuffled[:trn_sample_index], x_shuffled[trn_sample_index:test_sample_index], x_shuffled[test_sample_index:]
y_train, y_dev, y_test = y_shuffled[:trn_sample_index], y_shuffled[trn_sample_index:test_sample_index], y_shuffled[test_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_dic)))
print("Train/Dev split: {:d}/{:d}/{:d}".format(len(y_train), len(y_dev), len(y_test)))



# update args and print
args.embed_num = len(vocab_dic)
args.max_len = max_len
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
if args.saved_model is None:
    cnn = TextCNN(args)
else:
    print('\nLoading model from [%s]...' % args.snapshot)
    try:
        cnn = torch.load(args.saved_model)
    except:
        print("Sorry, This snapshot doesn't exist.")
        exit()


print("make train_step def")
def train_step(x_batch, y_batch, x_dev, y_dev,  model, args):
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()

    batches = data_helpers.batch_iter(
        list(zip(x_train, y_train)), args.batch_size, args.num_epochs)

    for batch in batches:
        x_batch, y_batch = zip(*batch)
        x_batch_Tensor, y_batch_Tensor = data_helpers.tensor4batch(x_batch, y_batch, args)

        x_batch_Variable = Variable(x_batch_Tensor).cuda()
        y_batch_Variable = Variable(y_batch_Tensor).cuda()

        optimizer.zero_grad()
        logit = model(x_batch_Variable)

        loss = F.cross_entropy(logit, torch.max(y_batch_Variable, 1)[1])
        loss.backward()
        optimizer.step()

        args.iter += 1

        if args.iter % args.log_interval == 0:
            corrects = (torch.max(logit, 1)[1] == torch.max(y_batch_Variable, 1)[1]).sum()
            corrects = corrects.data.cpu().numpy()[0]
            accuracy = 100.0 * corrects/args.batch_size
            sys.stdout.write(
                '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(args.iter,
                                                                         loss.data[0],
                                                                         accuracy,
                                                                         corrects,
                                                                         args.batch_size))

        if args.iter % args.dev_interval == 0:
            dev_step(x_dev, y_dev, model, args)

        if args.iter % args.save_interval == 0:
            if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
            save_prefix = os.path.join(args.save_dir, 'snapshot')
            save_path = '{}_steps{}.pt'.format(save_prefix, args.iter)
            torch.save(model, save_path)



print("make dev_step def")
def dev_step(x_dev, y_dev, model, args):
    model.cuda()
    model.eval()
    corrects_dev, avg_loss = 0, 0

    batches_dev = data_helpers.batch_iter(
        list(zip(x_dev, y_dev)), args.batch_size, 1)

    for batch in batches_dev:
        x_dev_batch, y_dev_batch = zip(*batch)
        x_dev_Tensor, y_dev_Tensor = data_helpers.tensor4batch(x_dev_batch, y_dev_batch, args)


        x_dev_Variable = Variable(x_dev_Tensor).cuda()
        y_dev_Variable = Variable(y_dev_Tensor).cuda()

        logit = model(x_dev_Variable)

        loss = F.cross_entropy(logit,  torch.max(y_dev_Variable, 1)[1], size_average=False)
        avg_loss += loss.data[0]
        corrects_dev += (torch.max(logit, 1)[1] == torch.max(y_dev_Variable, 1)[1]).data.sum()


    size = len(y_dev)
    avg_loss = avg_loss/size
    accuracy = 100.0 * corrects_dev/size

    args.list4ES.append(accuracy)

    if len(args.list4ES) > 4:
        if args.list4ES[len(args.list4ES) - 1] <= args.list4ES[len(args.list4ES) - 5]:
            if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
            save_prefix = os.path.join(args.save_dir, 'snapshot')
            save_path = '{}_steps{}.pt'.format(save_prefix, args.iter)
            torch.save(model, save_path)
            sys.exit("training is meaningless")

    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects_dev,
                                                                       size))


train_step(x_train, y_train, x_dev, y_dev, cnn, args)
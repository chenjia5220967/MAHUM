import os
import scipy.io as sio
import numpy as np
import time, datetime
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss,Module, Sequential, Conv3d, ReLU, LeakyReLU, Linear, Sigmoid, Conv2d, Softmax, BatchNorm3d
from torch.optim import Adam
from torch.autograd import Variable
import sys

class ProgressBar():
    def __init__(self, epoch_count, one_batch_count, pattern):
        self.total_count = one_batch_count
        self.current_index = 0
        self.current_epoch = 1
        self.epoch_count = epoch_count
        self.train_timer = Timer()
        self.pattern = pattern

    def show(self,currentEpoch, *args):
        self.current_index += 1
        if self.current_index == 1 :
            self.train_timer.tic()
        self.current_epoch = currentEpoch
        perCount = int(self.total_count / 100) # 7
        perCount = 1 if perCount == 0 else perCount
        percent = int(self.current_index / perCount)

        if self.total_count % perCount == 0:
            dotcount = int(self.total_count / perCount)
        else:
            dotcount = int(self.total_count / perCount)

        s1 = "\rEpoch:%d / %d [%s%s] %d / %d "%(
            self.current_epoch,
            self.epoch_count,
            "*"*(int(percent)),
            " "*(dotcount-int(percent)),
            self.current_index,
            self.total_count
        )

        s2 = self.pattern % tuple([float("{:.5f}".format(x)) for x in args])

        s3 = "%s,%s,remain=%s" % (
            s1, s2, self.train_timer.remain(self.current_index, self.total_count))
        sys.stdout.write(s3)
        sys.stdout.flush()
        if self.current_index == self.total_count :
            self.train_timer.toc()
            s3 = "%s,%s,total=%s" % (
                s1, s2, self.train_timer.averageTostr())
            sys.stdout.write(s3)
            sys.stdout.flush()
            self.current_index = 0
            print("\r")
            
class Timer(object):
    def __init__(self):
        self.init_time = time.time()
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0
        self.average_time = 0
        self.remain_time = 0

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def remain(self, iters, max_iters):
        if iters == 0:
            self.remain_time = 0
        else:
            self.remain_time = (time.time() - self.init_time) * (max_iters - iters) / iters

        return str(datetime.timedelta(seconds=int(self.remain_time)))

    def average(self):
        return str("%.3f" % self.average_time)

    def averageTostr(self):
        return str(datetime.timedelta(seconds=int(self.average_time)))
    
class autoencoder_model(Module):
    def __init__(self):
        super(autoencoder_model, self).__init__()


#
        self.encoder_cnn = Sequential(
            Conv3d(
                in_channels=1, out_channels=128, kernel_size=(3, 3, 6), stride=(1, 1, 2), padding=(1, 1, 0), bias=False
            ),
            ReLU(),
            Conv3d(
                in_channels=128, out_channels=64, kernel_size=(3, 3, 4), stride=(1, 1, 2), padding=(1, 1, 0), bias=False
            ),
            ReLU(),
            Conv3d(
                in_channels=64, out_channels=32, kernel_size=(3, 3, 5), stride=(1, 1, 2), padding=(0, 0, 0), bias=False
            ),
            ReLU(),
            Conv3d(
                in_channels=32, out_channels=16, kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 0), bias=False
            ),
            ReLU(),
            Conv3d(
                in_channels=16, out_channels=8, kernel_size=(1, 1, 4), stride=(1, 1, 2), padding=(0, 0, 0), bias=False
            ),
            ReLU(),
            Conv3d(
                in_channels=8, out_channels=3, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 0), bias=False
            )
        )

        self.decoder_linear = Sequential(
            Linear(3, 156*3, bias=False),
            #   ReLU(True)
        )
        self.decoder_nonlinear = Sequential(
            Linear(156*3, 156, bias=True),
            Sigmoid(),
            Linear(156, 156, bias=True),
            Sigmoid(),
            Linear(156, 156, bias=True)
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, 1, 3, 3, 156))
        out_encoder = self.encoder_cnn(x)
        out_encoder = torch.reshape(out_encoder, (-1, 3))
        out_encoder = out_encoder.abs()
        out_encoder = out_encoder.t() / out_encoder.sum(1)
        out_encoder = out_encoder.t()
        out_linear = self.decoder_linear(out_encoder) 
        out_nonlinear = self.decoder_nonlinear(out_linear)

        return out_linear,out_nonlinear, out_encoder

    def get_endmember(self, x):
        endmember = self.decoder_linear(x)
        return endmember

    # def get_abundance(self, x):
    #     x = self.encoder_cnn(x)
    #     x = torch.reshape(x, (-1, 4))
    #     weights = self.encoder_cnn(x)
    #     weights = torch.reshape(weights, (-1, 156))
    #     weights = weights.abs()
    #     weights = weights.t() / weights.sum(1)
    #     weights = weights.t()
    #     return weights



model_name = 'autoencoder'
workspace = 'D:\\坚果云\\我的坚果云\\我的坚果云\\代码\\解混代码\\3DCNN-var-main'
torch.cuda.set_device(0)
GPU_NUMS = 1
EPOCH = 200
BATCH_SIZE = 400
learning_rate = 1e-3
num_endmember = 3
num_band = 156
la = 0.05
ga = 0.008
dataset='samson_patch'

## --------------------- Load the data ---------------------------------------
N = 95*95;
hsi_name = 'data_samson_patch_3'
file_path = os.path.join(workspace, "data", "%s.mat" % hsi_name)
datamat = sio.loadmat(file_path)
hsi = datamat[hsi_name]
hsi = torch.from_numpy(hsi)
hsi = hsi[0:N, :, :]

endmember_name = 'weight_samson_long'
file_path = os.path.join(workspace, "data", "%s.mat" % endmember_name)
datamat = sio.loadmat(file_path)
W_init = datamat[endmember_name]
W_init = torch.from_numpy(W_init)

model = autoencoder_model()
model.decoder_linear[0].weight.data = W_init


model = model.cuda() if GPU_NUMS > 0 else model

print(model)

criterion = MSELoss()

## ----------------------------------------------------------------
if model_name == 'autoencoder':
    ignored_params = list(map(id, model.decoder_linear[0].parameters()))  # 需要微调的参数
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())  # 需要调整的参数
    optimizer = Adam([
        {'params': base_params},
        {'params': model.decoder_linear[0].parameters(), 'lr': 1e-4}
    ], lr=learning_rate, weight_decay=1e-5)

vector_all = []
linear_all = []
code_onehot = torch.eye(num_endmember)
code_onehot = Variable(code_onehot).cuda()

W_init = Variable(W_init).cuda()
data_loader = DataLoader(hsi, batch_size=BATCH_SIZE, shuffle=False)

proBar = ProgressBar(EPOCH, len(data_loader), "Loss:%.5f")

for epoch in range(1, EPOCH):
    l_item = 0
    for data in data_loader:
        pixel = data
        pixel_1 = Variable(pixel).cuda() if GPU_NUMS > 0 else Variable(pixel)
        # ===================forward=====================
        pixel = torch.reshape(pixel_1, (-1, 1, 3, 3, num_band))
        output_linear, out_nonlinear, vector = model(pixel)
        pixel = pixel_1[:,4,:]
        pixel = torch.reshape(pixel, (-1, num_band))

        output_linear_get = output_linear[:, 0:num_band] + output_linear[:, num_band:num_band*2] + output_linear[:, num_band*2:num_band*3]

        loss_reconstruction = criterion(output_linear_get+out_nonlinear, pixel)


        #l2
        l2_temp1 = model.decoder_nonlinear[0].weight
        l2_temp1 = l2_temp1.reshape(num_band, num_band * 3)
        l2_temp2 = model.decoder_nonlinear[2].weight
        l2_temp3 = model.decoder_nonlinear[4].weight
        l2_regularization = torch.cat((l2_temp1, l2_temp2, l2_temp3), 1)
        l2_regularization = torch.norm(l2_regularization)

        #smooth
        weight_temp = model.get_endmember(code_onehot)
        loss_diff_temp = weight_temp[:, 2:-1] - weight_temp[:, 1:-2]
        loss_diff_temp = loss_diff_temp.abs()
        loss_difference = loss_diff_temp.mean()

        loss = loss_reconstruction + la*l2_regularization + ga*loss_difference

                # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        proBar.show(epoch, loss.item())
        # ===================log========================
        if epoch == EPOCH-1:
            vector_temp = vector.cpu().data
            vector_temp = vector_temp.numpy()
            vector_all = np.append(vector_all,vector_temp)
            vector_all = vector_all.reshape(-1, num_endmember)
            name_vector_all = 'vector_all_samson_pre.mat'
            sio.savemat(name_vector_all, {'vector_all': vector_all})
        
        
 # save_path = str(dataset) +"_"+"_"+model_my+ str(beta)+'_cycunet_result.mat'
# sio.savemat(save_path,{'Y':Y,'abu_est':abu_est1, 'A':A, 'M':M_true[:,index]})


torch.save(model.encoder_cnn.state_dict(), 'samson_model.pth')
endmember = model.get_endmember(code_onehot)
endmember = endmember.cpu().data
endmember = endmember.numpy()
sio.savemat('endmember_pre_samson.mat', {'endmember': endmember})
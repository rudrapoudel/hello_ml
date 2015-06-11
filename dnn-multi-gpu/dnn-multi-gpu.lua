----------------------------------------------------------------------
-- A complete and minimum example of multi-GPUs DNN in Torch.
--
-- Rudra Poudel
--
-- Dependency: cudnn, fbcunn
----------------------------------------------------------------------
-- Include modules/libraries
require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'optim'
require 'fbcunn'
require 'fbnn'

-- Command line arguments
cmd = torch.CmdLine()
cmd:text()
cmd:text('DNN Example')
cmd:text()
cmd:text('Options:')
cmd:option('-seed',            1,           'fixed input seed for repeatable experiments')
cmd:option('-threads',         4,           'number of threads')
cmd:option('-gpuid',           1,           'gpu id')
cmd:option('-num_gpu',         1,           'num gpus')
cmd:option('-train_criterion', 'NLL',       'train_criterion: MSE | NLL')
cmd:option('-learning_rate',   5e-2,        'learning rate at t=0')
cmd:option('-momentum',        0.6,         'momentum')
cmd:option('-weight_decay',    1e-5,        'weight decay')
cmd:option('-batch_size',      256,           'mini-batch size (1 = pure stochastic)')
cmd:option('-epoch_size',      10,          'number of batches per epoch')
cmd:option('-dropout',         false,       'do dropout with 0.5 probability')
cmd:option('-print_layers_op', false,       'Output the values from each layers')
cmd:text()
opt = cmd:parse(arg or {})

-- Set system options
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.gpuid)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

-- Set DNN parameters
num_classes = 6
batch_size = opt.batch_size
sample_size = {3, 32, 32} -- input: channel/depth , height, width 
feature_size = {3, 32, 64, 96} -- input channel/depth, feature maps ...
filter_size = {5, 5}
pool_size = {2, 2}
pool_step = {2, 2}
classifer_hidden_units = {512}
features_out = feature_size[3] * 5 * 5 -- WARNING: change this if you change feature/filter/pool size/step
-- Dropout
dropout_p = 0.5
if opt.dropout then
  print("... using dropout")
end

-- Configuring optimizer
optim_state = {
  learningRate = opt.learning_rate,
  learningRateDecay = 0,
  momentum = opt.momentum,
  dampening = 0.0,
  weightDecay = opt.weight_decay,
}

-- Define DNN model
print 'Defining DNN model'
model = nn.Sequential()
-- Stage 1
model:add(cudnn.SpatialConvolution(feature_size[1], feature_size[2], filter_size[1], filter_size[2]), 1, 1) -- 64 - 5 + 1 = 60
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(pool_size[1],pool_size[2],pool_step[1],pool_step[2])) -- 30
-- Stage 2
model:add(cudnn.SpatialConvolution(feature_size[2], feature_size[3], filter_size[1], filter_size[2]), 1, 1) -- 30 - 5 + 1 = 26
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(pool_size[1],pool_size[2],pool_step[1],pool_step[2])) -- 13
-- Stage 3
--model:add(cudnn.SpatialConvolution(feature_size[3], feature_size[4], filter_size[1], filter_size[2]), 1, 1) -- 13 - 5 + 1 = 9
--model:add(cudnn.ReLU(true))
--model:add(cudnn.SpatialMaxPooling(pool_size[1],pool_size[2],pool_step[1],pool_step[2])) -- 4

-- Get feature vectors i.e. flat feature maps
model:add(nn.Reshape(features_out, true)) 
if opt.dropout then
  model:add(nn.Dropout(dropout_p))
end
-- Fully connected layers
model:add(nn.Linear(features_out, classifer_hidden_units[1]))
model:add(nn.ReLU())
if opt.dropout then
  model:add(nn.Dropout(dropout_p))
end
model:add(nn.Linear(classifer_hidden_units[1], num_classes))

-- Output model
if opt.train_criterion == 'MSE' then
  model:add(nn.SoftMax())
  criterion = nn.MSECriterion()
elseif opt.train_criterion == 'NLL' then
  model:add(nn.LogSoftMax())
  criterion = nn.ClassNLLCriterion() 
end

-- Multi-GPUs
if opt.num_gpu > 1 then
  print('Using data parallel')
  local gpu_net = nn.DataParallel(1):cuda()
  for i = 1, opt.num_gpu do
    local cur_gpu = math.fmod(opt.gpuid + (i-1)-1, cutorch.getDeviceCount())+1
    cutorch.setDevice(cur_gpu)
    gpu_net:add(model:clone():cuda(), cur_gpu)
  end
  cutorch.setDevice(opt.gpuid)

  model = gpu_net
end

-- Apply cuda
model = model:cuda()
criterion = criterion:cuda()

-- Create artificial data for testing
-- Note: Spatial*MM use BDHW and  Spatial*CUDA use DHWB
-- bmode = 'DHWB' -- depth/channels x height x width x batch
bmode = 'BDHW' -- batch x depth/channels x height x width
inputs = torch.rand(batch_size, sample_size[1], sample_size[2], sample_size[3]):float()
targets = torch.floor( (torch.rand(batch_size) * num_classes) + 1):float()
targets_matrix = torch.Tensor(batch_size, num_classes):zero():float()
for i = 1, batch_size do
  targets_matrix[{i,targets[i]}] = 1
end
--print (targets)
--print (targets_matrix)
if opt.train_criterion == 'MSE' then
  targets = targets_matrix
end
inputs = inputs:cuda()
targets = targets:cuda()
  
-- Optimizer
optimator = nn.Optim(model, optim_state)

-- The the tensor variables for model params and gradient params 
if opt.num_gpu>1 then
  params, grad_params = model:get(1):getParameters()
  optimator:setParameters(optim_state)
  cutorch.synchronize()
  -- set the dropouts to training mode
  model:training()
  model:cuda()  -- get it back on the right GPUs
else
  params, grad_params = model:getParameters()
  -- set the dropouts to training mode
  model:training()
end


-- Define the function for gradient optimization i.e. for SGD
local function trainBatch()
  
  f, outputs = optimator:optimize(
     optim.sgd,
     inputs,
     targets,
     criterion)

  if opt.num_gpu > 1 then cutorch.synchronize() end
end -- END: local function trainBatch()

-- Train- epoch
print('Training ...')
local tm = torch.Timer()
for i=1, opt.epoch_size do
  print(i)
  trainBatch()
end
print('Time took: ' .. tm:time().real)
print('DONE')
-- EOF

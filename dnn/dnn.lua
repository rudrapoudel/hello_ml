----------------------------------------------------------------------
-- A complete and minimum example of DNN in Torch.
--
-- Rudra Poudel
----------------------------------------------------------------------
-- Include modules/libraries
require 'torch'
require 'nn'
--require 'nnx'
require 'optim'

-- Command line arguments
cmd = torch.CmdLine()
cmd:text()
cmd:text('DNN Example')
cmd:text()
cmd:text('Options:')
cmd:option('-type',            'float',     'type: float | cuda')
cmd:option('-seed',            1,           'fixed input seed for repeatable experiments')
cmd:option('-threads',         1,           'number of threads')
cmd:option('-gpuid',           1,           'gpu id')
cmd:option('-train_criterion', 'MSE',       'train_criterion: MSE | NLL')
cmd:option('-learning_rate',   5e-2,        'learning rate at t=0')
cmd:option('-momentum',        0.6,         'momentum')
cmd:option('-weight_decay',    1e-5,        'weight decay')
cmd:option('-batch_size',      1,           'mini-batch size (1 = pure stochastic)')
cmd:option('-dropout',         false,        'do dropout with 0.5 probability')
cmd:option('-print_layers_op', false,       'Output the values from each layers')
cmd:text()
opt = cmd:parse(arg or {})

-- Set options
torch.setdefaulttensortype('torch.FloatTensor')
if opt.type == 'cuda' then
   print('... switching to CUDA')
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.gpuid)
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

-- Set DNN parameters
num_classes = 6
batch_size = opt.batch_size
sample_size = {3, 32, 32} -- input: channel/depth , height, width 
feature_size = {3, 16, 32} -- input channel/depth, feature maps ...
filter_size = {5, 5}
pool_size = {2, 2}
pool_step = {2, 2}
classifer_hidden_units = {256}
features_out = feature_size[3] * 5 * 5 -- WARNING: change this if you change feature/filter/pool size/step
-- Dropout
dropout_p = 0.5
if opt.dropout then
  print("... using dropout")
end

-- Configuring optimizer
optim_state = {
  learningRate = opt.learning_rate,
  weightDecay = opt.weight_decay,
  momentum = opt.momentum,
  learningRateDecay = 5e-7
}
optim_method = optim.sgd

print 'Defining DNN model'
model = nn.Sequential()
-- Stage 1
model:add(nn.SpatialConvolutionMM(feature_size[1], feature_size[2], filter_size[1], filter_size[2])) -- 32 - 5 + 1 = 28
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(pool_size[1],pool_size[2],pool_step[1],pool_step[2])) -- 14
-- Stage 2
model:add(nn.SpatialConvolutionMM(feature_size[2], feature_size[3], filter_size[1], filter_size[2])) -- 14 - 5 + 1 = 10
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(pool_size[1],pool_size[2],pool_step[1],pool_step[2])) -- 5
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

-- Apply cuda
if opt.type == 'cuda' then
  model = model:cuda()
  criterion = criterion:cuda()
end

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

if opt.type == 'cuda' then
  inputs = inputs:cuda()
  targets = targets:cuda()
  targets_matrix = targets_matrix:cuda()
end
  
-- Test model
if opt.print_layers_op then
--  model:forward(inputs)
  -- Output the results of each layers/modules
  local o = inputs
  for i=1,#(model.modules) do
    o = model.modules[i]:forward(o)
    print(#o)
    --  print(o) -- WARNING: will print all output matrix values
  end
end

-- The the tensor variables for model params and gradient params 
params,grad_params = model:getParameters()

-- Define the function for gradient optimization i.e. for SGD
-- create closure to evaluate f(X) and df/dX
local feval = function(x)
  -- get new parameters
  if x ~= params then
    params:copy(x)
  end
  
  -- reset gradients
  grad_params:zero()

  -- f is the average of all criterions
  f = 0;
  
  -- evaluate function for complete mini batch  
  local outputs = model:forward(inputs)
  if opt.type == 'cuda' then
    outputs = outputs:cuda()
  else
    outputs = outputs:float()
  end
  print(outputs)
  local df_do = torch.Tensor(outputs:size(1), outputs:size(2)):float()
  for i=1,batch_size do
    local err = 0;
    if opt.train_criterion == 'MSE' then
      -- get error
      err = criterion:forward(outputs[i], targets_matrix[i])
      -- estimate df/dW
      df_do[i]:copy(criterion:backward(outputs[i], targets_matrix[i]))
      -- print(outputs[i])
      -- print(targets_matrix[i])
      -- print(df_do[i])
    elseif opt.train_criterion == 'NLL' then
      -- get error
      err = criterion:forward(outputs[i], targets[i])
      -- estimate df/dW
      df_do[i]:copy(criterion:backward(outputs[i], targets[i]))
      -- print(outputs[i])
      -- print(targets[i])
      -- print(df_do[i])
    end    
    f = f + err
  end
  print (f)
  if opt.type == 'cuda' then
    df_do = df_do:cuda()
  end
  -- back prop error
  model:backward(inputs, df_do)
  -- normalize gradients and f(X)
  grad_params:div(batch_size)
  f = f/batch_size

  -- return f and df/dX
  return f,grad_params
end -- END: local feval = function(x)

-- optimize on current mini-batch
optim_method(feval, params, optim_state)

-- EOF

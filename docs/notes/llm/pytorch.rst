.. meta::
    :description lang=en: PyTorch cheat sheet covering tensors, operations, gradients, GPU usage, neural networks, and NumPy interoperability
    :keywords: Python, Python3, PyTorch, tensor, deep learning, GPU, CUDA, gradient, autograd, numpy, neural network

=======
PyTorch
=======

.. contents:: Table of Contents
    :backlinks: none

PyTorch is an open-source machine learning framework developed by Meta AI. It provides
a flexible and intuitive interface for building and training neural networks with strong
GPU acceleration support. PyTorch uses dynamic computation graphs, making it easier to
debug and experiment with different model architectures compared to static graph frameworks.

Check CUDA
----------

Before running GPU-accelerated computations, verify that CUDA is properly installed and
accessible. These commands help check GPU availability and configure device settings.

.. code-block:: python

    >>> import torch
    >>> torch.cuda.is_available()
    True
    >>> torch.cuda.nccl.version()
    (2, 26, 2)
    >>> torch.cuda.device_count()
    2
    >>> torch.cuda.get_device_name(0)
    'NVIDIA GeForce RTX 3090'
    >>> torch.cuda.set_device(0)
    >>> torch.cuda.current_device()
    0

Check Device
------------

Determine where a tensor is stored (CPU or GPU) to ensure computations run on the intended device.

.. code-block:: python

    >>> tensor = torch.tensor([0, 1, 2, 3, 4, 5])
    >>> tensor.device
    device(type='cpu')
    >>> tensor = tensor.to('cuda')
    >>> tensor.device
    device(type='cuda', index=0)

Create Tensors
--------------

Tensors are the fundamental data structure in PyTorch, similar to NumPy arrays but with
GPU acceleration support. You can create tensors from Python lists, with specific values,
or using various initialization methods.

.. code-block:: python

    >>> x = torch.tensor([0, 1, 2, 3, 4, 5])
    >>> x = torch.empty(2, 2)
    >>> x = torch.rand(2, 2)
    >>> x = torch.randn(2, 2)
    >>> x = torch.ones(2, 2)
    >>> x = torch.zeros(2, 2)
    >>> x = torch.arange(0, 10, 2)
    >>> x = torch.linspace(0, 1, 5)

    >>> device = torch.device("cuda:0")
    >>> x = torch.tensor([1, 2, 3, 4, 5], device=device)
    >>> x
    tensor([1, 2, 3, 4, 5], device='cuda:0')

Tensor Properties
-----------------

Understanding tensor properties like shape, data type, and device location is crucial
for debugging and ensuring compatibility between operations.

.. code-block:: python

    >>> x = torch.randn(2, 3, 4)
    >>> x.shape
    torch.Size([2, 3, 4])
    >>> x.size()
    torch.Size([2, 3, 4])
    >>> x.dtype
    torch.float32
    >>> x.device
    device(type='cpu')
    >>> x.numel()
    24

Contiguous Tensors
------------------

Tensors must be stored in contiguous memory blocks for certain operations. After operations
like ``transpose`` or ``permute``, tensors may not be contiguous. Use ``contiguous()`` to
create a contiguous copy when needed.

.. code-block:: python

    >>> x = torch.randn(2, 3)
    >>> x.is_contiguous()
    True
    >>> y = x.transpose(0, 1)
    >>> y.is_contiguous()
    False
    >>> z = y.contiguous()
    >>> z.is_contiguous()
    True

View vs Reshape
---------------

``view()`` requires tensors to be contiguous and returns a view sharing the same memory.
``reshape()`` works on both contiguous and non-contiguous tensors, creating a copy if needed.
Use ``view()`` when you know the tensor is contiguous for better performance.

.. code-block:: python

    >>> x = torch.randn(2, 3, 4)
    >>> x.is_contiguous()
    True
    >>> x.view(2, 12).shape
    torch.Size([2, 12])

    >>> y = x.transpose(0, 1)
    >>> y.is_contiguous()
    False
    >>> y.view(3, 8)
    RuntimeError: view size is not compatible with input tensor's size and stride

    >>> y.reshape(3, 8).shape
    torch.Size([3, 8])
    >>> y.contiguous().view(3, 8).shape
    torch.Size([3, 8])

Reshape Tensors
---------------

Common reshaping operations for preparing data for neural network layers.

.. code-block:: python

    x = torch.randn(2, 3, 4)
    x.reshape(6, 4)
    x.flatten()
    x.squeeze()
    x.unsqueeze(0)

Move Tensors
------------

Transfer tensors between CPU and GPU, or between different GPU devices. This is necessary
when working with models and data on different devices.

.. code-block:: python

    x = torch.randn(2, 3)
    x_gpu = x.to('cuda')
    x_gpu = x.cuda()
    x_cpu = x_gpu.to('cpu')
    x_cpu = x_gpu.cpu()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)

Arithmetic
----------

PyTorch supports element-wise operations and matrix operations. Most operations have
both functional and method forms, and support broadcasting for tensors of different shapes.

.. code-block:: python

    >>> x = torch.rand(2, 2, device=0)
    >>> y = torch.rand(2, 2, device=0)
    >>> x
    tensor([[0.2171, 0.7797],
            [0.7265, 0.6759]], device='cuda:0')
    >>> y
    tensor([[0.6766, 0.1862],
            [0.2438, 0.0076]], device='cuda:0')

    >>> x + y
    tensor([[0.8937, 0.9659],
            [0.9703, 0.6834]], device='cuda:0')
    >>> torch.add(x, y)
    tensor([[0.8937, 0.9659],
            [0.9703, 0.6834]], device='cuda:0')
    >>> x - y
    tensor([[-0.4594,  0.5935],
            [ 0.4827,  0.6683]], device='cuda:0')
    >>> x * y
    tensor([[0.1469, 0.1452],
            [0.1771, 0.0051]], device='cuda:0')
    >>> x / y
    tensor([[ 0.3209,  4.1880],
            [ 2.9796, 89.4011]], device='cuda:0')
    >>> x ** 2
    tensor([[0.0471, 0.6079],
            [0.5278, 0.4568]], device='cuda:0')
    >>> x @ y
    tensor([[0.3370, 0.0463],
            [0.6563, 0.1404]], device='cuda:0')
    >>> torch.matmul(x, y)
    tensor([[0.3370, 0.0463],
            [0.6563, 0.1404]], device='cuda:0')

In-place Operations
-------------------

Operations ending with ``_`` modify tensors directly without creating new tensors,
saving memory. Use these carefully as they can affect gradient computation.

.. code-block:: python

    x = torch.rand(2, 2, device=0)
    y = torch.rand(2, 2, device=0)
    y.add_(x)
    y.sub_(x)
    y.mul_(x)
    y.div_(x)
    y.zero_()
    y.fill_(5)

Transpose
---------

Swap dimensions of multi-dimensional tensors. This is commonly used when preparing
data for different neural network layers that expect specific input shapes.

.. code-block:: python

    >>> x = torch.randn(2, 6, 2, device=0)
    >>> x.shape
    torch.Size([2, 6, 2])
    >>> y = x.transpose(1, 2)
    >>> y.shape
    torch.Size([2, 2, 6])
    >>> x.T.shape
    torch.Size([2, 2, 6])
    >>> x.permute(2, 0, 1).shape
    torch.Size([2, 2, 6])

Matrix Multiplication
---------------------

Perform batch matrix operations on high-dimensional tensors. This is fundamental for
neural network computations where you process multiple samples simultaneously.

.. code-block:: python

    >>> x = torch.randn(1, 2, 3, 4, device=0)
    >>> x @ x.transpose(2, 3)
    tensor([[[[ 1.0950, -0.1160, -1.4840],
              [-0.1160,  2.4025,  2.8093],
              [-1.4840,  2.8093,  5.9335]],

             [[ 3.2498, -0.3049, -0.5129],
              [-0.3049,  0.1591, -0.2596],
              [-0.5129, -0.2596,  2.9863]]]], device='cuda:0')

    >>> torch.bmm(x, x.transpose(2, 3))
    >>> torch.einsum('bijk,bikl->bijl', x, x.transpose(2, 3))

Aggregation
-----------

Reduce tensors along specified dimensions using aggregation functions. These operations
are essential for computing statistics and reducing dimensionality.

.. code-block:: python

    >>> x = torch.randn(2, 3, 4)
    >>> x.sum()
    tensor(5.2341)
    >>> x.sum(dim=0).shape
    torch.Size([3, 4])
    >>> x.sum(dim=-1).shape
    torch.Size([2, 3])
    >>> x.mean()
    tensor(0.2181)
    >>> x.std()
    tensor(1.0234)
    >>> x.max()
    tensor(2.3456)
    >>> x.min()
    tensor(-1.8765)
    >>> x.argmax()
    tensor(7)
    >>> x.argmin()
    tensor(15)

    >>> x.max(dim=1)
    torch.return_types.max(
    values=tensor([[1.2345, 0.9876, 1.5432, 0.8765],
                   [0.7654, 1.3456, 0.6543, 1.2345]]),
    indices=tensor([[1, 2, 0, 1],
                    [2, 0, 1, 2]]))

Slicing
-------

Extract parts of tensors using NumPy-style indexing. Slicing is essential for accessing
specific elements, rows, columns, or sub-tensors without copying data.

.. code-block:: python

    >>> x = torch.randn(2, 3, device=0)
    >>> x
    tensor([[-1.3921,  0.0475,  0.7571],
            [-0.1469, -0.3882,  0.2149]], device='cuda:0')
    >>> x[:, 1]
    tensor([ 0.0475, -0.3882], device='cuda:0')
    >>> x[1, :]
    tensor([-0.1469, -0.3882,  0.2149], device='cuda:0')
    >>> x[1, 1].item()
    -0.3882044851779938

    >>> x = torch.triu(torch.ones(5, 5))
    >>> x[:3, :3]
    tensor([[1., 1., 1.],
            [0., 1., 1.],
            [0., 0., 1.]])

Advanced Indexing
-----------------

Use boolean masks and fancy indexing to select elements based on conditions or
specific index arrays.

.. code-block:: python

    >>> x = torch.randn(3, 4)
    >>> mask = x > 0
    >>> x[mask]
    tensor([0.5234, 1.2345, 0.8765, 0.3456, 1.5678])

    x[x > 0] = 0

    indices = torch.tensor([0, 2])
    x[indices]
    x[[0, 1], [1, 2]]

Concatenation
-------------

Combine multiple tensors along existing or new dimensions. This is useful for
building batches or combining features from different sources.

.. code-block:: python

    >>> x = torch.randn(2, 3)
    >>> y = torch.randn(2, 3)
    >>> torch.cat([x, y], dim=0).shape
    torch.Size([4, 3])
    >>> torch.cat([x, y], dim=1).shape
    torch.Size([2, 6])
    >>> torch.stack([x, y], dim=0).shape
    torch.Size([2, 2, 3])
    >>> torch.vstack([x, y]).shape
    torch.Size([4, 3])
    >>> torch.hstack([x, y]).shape
    torch.Size([2, 6])

Splitting
---------

Split tensors into multiple chunks or along specific dimensions. Useful for
distributing data across multiple GPUs or processing in smaller batches.

.. code-block:: python

    >>> x = torch.randn(6, 4)
    >>> chunks = torch.split(x, 2, dim=0)
    >>> len(chunks)
    3
    >>> chunks = torch.chunk(x, 3, dim=0)
    >>> len(chunks)
    3
    >>> tensors = x.unbind(dim=0)
    >>> len(tensors)
    6

Gradient
--------

Automatic differentiation is PyTorch's core feature for training neural networks.
Enable gradient tracking on tensors to compute derivatives during backpropagation.

.. code-block:: python

    >>> x = torch.randn(3, requires_grad=True, device=0)
    >>> x
    tensor([-1.1442, -0.8709, -0.2581], device='cuda:0', requires_grad=True)

    >>> y = x.detach()
    >>> y
    tensor([-1.1442, -0.8709, -0.2581], device='cuda:0')

    >>> x.requires_grad_(False)
    tensor([-1.1442, -0.8709, -0.2581], device='cuda:0')

Disable Gradient
----------------

Temporarily disable gradient computation for inference or when you don't need gradients.
This reduces memory usage and speeds up computations.

.. code-block:: python

    >>> x = torch.randn(3, requires_grad=True, device=0)
    >>> with torch.no_grad():
    ...     y = x + 1
    ...     print(y)
    ...
    tensor([1.2969, 1.5251, 0.7915], device='cuda:0')

    with torch.inference_mode():
        y = x * 2

    @torch.no_grad()
    def predict(x):
        return model(x)

Backpropagation
---------------

Compute gradients using automatic differentiation. PyTorch builds a computation graph
during the forward pass and computes gradients during the backward pass using the chain rule.

.. code-block:: python

    >>> x = torch.randn(3, requires_grad=True)
    >>> y = x + 1
    >>> z = y * y * 3
    >>> z = z.mean()
    >>> z.backward()
    >>> x.grad
    tensor([1.2036, 5.0103, 0.5143])

    x.grad.zero_()
    z.backward()

Gradient Accumulation
---------------------

Accumulate gradients over multiple backward passes. This is useful for simulating
larger batch sizes when GPU memory is limited.

.. code-block:: python

    optimizer.zero_grad()
    for i, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

Neural Network Module
---------------------

Define neural networks by subclassing ``nn.Module``. This provides a clean interface
for building complex models with automatic parameter management.

.. code-block:: python

    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = Net()
    model.to('cuda')

Common Layers
-------------

PyTorch provides a wide variety of pre-built layers for constructing neural networks.

.. code-block:: python

    import torch.nn as nn

    linear = nn.Linear(10, 5)
    conv2d = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    maxpool = nn.MaxPool2d(2, 2)
    dropout = nn.Dropout(0.5)
    batchnorm = nn.BatchNorm2d(64)
    relu = nn.ReLU()
    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=1)

    lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
    gru = nn.GRU(input_size=10, hidden_size=20)
    embedding = nn.Embedding(1000, 128)

Loss Functions
--------------

Loss functions measure how well the model's predictions match the target values.
Choose the appropriate loss function based on your task.

.. code-block:: python

    import torch.nn as nn

    mse_loss = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()
    bce_with_logits = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    nll_loss = nn.NLLLoss()

    outputs = model(inputs)
    loss = cross_entropy(outputs, targets)

Optimizers
----------

Optimizers update model parameters based on computed gradients. Different optimizers
use different strategies for parameter updates.

.. code-block:: python

    import torch.optim as optim

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

Learning Rate Scheduler
-----------------------

Adjust the learning rate during training to improve convergence and model performance.

.. code-block:: python

    from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10)

    for epoch in range(num_epochs):
        train(...)
        val_loss = validate(...)
        scheduler.step(val_loss)

Training Loop
-------------

A typical training loop involves forward pass, loss computation, backward pass,
and parameter updates.

.. code-block:: python

    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            print(f'Loss: {loss.item():.4f}')

Evaluation Mode
---------------

Switch between training and evaluation modes to control behavior of layers like
dropout and batch normalization.

.. code-block:: python

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)

Save and Load Models
--------------------

Save trained models to disk and load them later for inference or continued training.

.. code-block:: python

    torch.save(model.state_dict(), 'model.pth')
    model.load_state_dict(torch.load('model.pth'))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'checkpoint.pth')

    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

DataLoader
----------

Efficiently load and batch data for training. DataLoader handles shuffling, batching,
and parallel data loading.

.. code-block:: python

    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    for batch_x, batch_y in loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

Custom Dataset
--------------

Create custom datasets by subclassing ``Dataset`` for loading data from various sources.

.. code-block:: python

    from torch.utils.data import Dataset

    class CustomDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    dataset = CustomDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=32)

NumPy Conversion
----------------

Convert between PyTorch tensors and NumPy arrays for interoperability with other
libraries. Note that tensors on GPU must be moved to CPU first.

.. code-block:: python

    >>> x = torch.randn([1, 2, 3], device=0)
    >>> y = x.cpu().numpy()
    >>> y
    array([[[-0.11979043,  0.13762406, -1.2633433 ],
            [-0.380241  ,  1.5320604 , -1.0828359 ]]], dtype=float32)

    import numpy as np
    arr = np.array([[1, 2], [3, 4]])
    tensor = torch.from_numpy(arr)

Shared Memory
-------------

CPU tensors and NumPy arrays can share the same underlying memory, so modifications
to one will affect the other.

.. code-block:: python

    >>> x = torch.randn(1, 2, 3)
    >>> y = x.numpy()
    >>> x.add_(1)
    tensor([[[ 1.8195,  3.0259,  0.6733],
             [ 2.6539,  1.1562, -0.9821]]])
    >>> y
    array([[[ 1.8194908 ,  3.0258512 ,  0.67326605],
            [ 2.6539469 ,  1.1561831 , -0.98211455]]], dtype=float32)

Random Seed
-----------

Set random seeds for reproducibility across different runs. This ensures consistent
results when debugging or comparing experiments.

.. code-block:: python

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import numpy as np
    import random
    np.random.seed(42)
    random.seed(42)

GPU Memory Management
---------------------

Monitor and manage GPU memory usage to avoid out-of-memory errors and optimize performance.

.. code-block:: python

    >>> torch.cuda.empty_cache()
    >>> torch.cuda.memory_allocated()
    1073741824
    >>> torch.cuda.memory_reserved()
    2147483648
    >>> torch.cuda.max_memory_allocated()
    3221225472

    torch.cuda.reset_peak_memory_stats()
    print(torch.cuda.memory_summary())


Distributed Training
--------------------

NCCL (NVIDIA Collective Communication Library) enables efficient multi-GPU and multi-node
training. Initialize the distributed process group and check its status.

.. code-block:: python

    >>> import torch.distributed as dist
    >>> dist.is_available()
    True
    >>> dist.is_nccl_available()
    True
    >>> dist.is_initialized()
    False

    dist.init_process_group(backend='nccl')

    >>> dist.is_initialized()
    True
    >>> dist.get_rank()
    0
    >>> dist.get_world_size()
    4
    >>> dist.get_backend()
    'nccl'

Launch with torchrun
--------------------

Use ``torchrun`` to launch distributed training across multiple GPUs or nodes. It automatically
sets up environment variables for distributed training.

.. code-block:: bash

    # single node, multiple GPUs
    torchrun --nproc_per_node=4 train.py

    # multiple nodes
    torchrun --nproc_per_node=4 \
             --nnodes=2 \
             --node_rank=0 \
             --master_addr="192.168.1.1" \
             --master_port=29500 \
             train.py

Launch with Slurm
-----------------

Submit distributed training jobs to Slurm clusters. Slurm manages resource allocation
and node assignment. See :doc:`../hpc/slurm` for more Slurm examples.

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=pytorch_ddp
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=4
    #SBATCH --gres=gpu:4
    #SBATCH --time=24:00:00

    export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
    export MASTER_PORT=29500

    srun torchrun --nproc_per_node=4 \
                  --nnodes=$SLURM_NNODES \
                  --node_rank=$SLURM_NODEID \
                  --master_addr=$MASTER_ADDR \
                  --master_port=$MASTER_PORT \
                  train.py

DistributedDataParallel
------------------------

Wrap your model with DDP for multi-GPU training. DDP uses NCCL for efficient gradient
synchronization across GPUs.

.. code-block:: python

    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = Net().to(device)
    model = DDP(model, device_ids=[local_rank])

    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

Collective Operations
---------------------

NCCL provides collective communication operations for distributed training.

.. code-block:: python

    import torch.distributed as dist

    tensor = torch.randn(2, 3).cuda()

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    dist.broadcast(tensor, src=0)
    dist.all_gather(tensor_list, tensor)
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.barrier()

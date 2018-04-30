Tensor-Fu-1
===========


Exercise 1
----------

.. code-block:: python

   import torch
   from torch import nn
   x = torch.randn(9, 10)


Exercise 2
----------

.. code-block:: python

   import torch
   from torch import nn

   x2dim = torch.randn(9, 10)

   # required and default parameters:
   # fc = nn.Linear(in_features, out_features)

Task: Create a linear layer which works wih x2dim


Exercise 3
----------


.. code-block:: python

   import torch
   from torch import nn

   x3dim = torch.randn(9, 10, 11)

   # required and default parameters:
   # conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0)

Task: Create a convolution which works on x3dim



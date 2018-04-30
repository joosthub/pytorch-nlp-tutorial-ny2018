Environment Setup
=================

On this page, you will find not only the list of dependencies to install
for the tutorial, but a description of how to install them. This tutorial assumes
you have a laptop with OSX or Linux. If you use Windows, you might have to install
a virtual machine to get a UNIX-like environment to continue with the rest of this
instruction. A lot of this instruction is more verbose than needed to accomodate
participants of different skill levels.

**Please note that these are only optional.  On the first day of this training, you will be provided with a link to a JupyterHub instance where the environment will be pre-made and ready to go!**

0. Get Anaconda
---------------

Anaconda is a Python (and R) distribution that aims to provide everything
needed for common scientific and machine learning situations out-of-the-box.
We chose Anaconda for this tutorial as it significantly simplifies Python
dependency management.

In practice, Anaconda can be used to manage different environment and packages.
This setup document will assume that you have Anaconda installed as your default
Python distribution.

You can download Anaconda here: https://www.continuum.io/downloads

After installing Anaconda, you can access its command-line interface
with the :code:`conda` command.


1. Create a new environment
---------------------------

Environments are a tool for sanitary software development.  By this, we mean that
you can install specific versions of packages without worrying that it breaks
a dependency elsewhere.

Here is how you can create an environment with Anaconda

.. code-block:: bash

   conda create -n dl4nlp python=3.6


2. Install Dependencies
-----------------------

2a. Activate the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After creating the environment, you need to **activate** the environment:

.. code-block:: bash

   source activate dl4nlp

After an environment is activated, it might prepend/append itself to your
console prompt to let you know it is active.

With the environment activated, any installation commands
(whether it is :code:`pip install X`, :code:`python setup.py install` or using
Anaconda's install command :code:`conda install X`) will only install inside
the environment.

2b. Install IPython and Jupyter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two core dependencies are IPython and Jupyter.  Let's install them first:

.. code-block:: bash

   conda install ipython
   conda install jupyter

To allow a jupyter notebooks to use this environment as their kernel, it
needs to be linked:

.. code-block:: bash

   python -m ipykernel install --user --name dl4nlp

2c. Installing CUDA (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NOTE: CUDA is currently not supported out of the conda package control manager.
Please refer to pytorch's github repository for compilation instructions.

If you have a CUDA compatible GPU, it is worthwhile to take advantage of it as
it can significantly speedup training and make your PyTorch experimentation more
enjoyable.

To install CUDA:

1. Download CUDA appropriate to your OS/Arch from `here <https://developer.nvidia.com/cuda-downloads>`_.
2. Follow installation steps for your architecture/OS. For Ubuntu/x86_64, see `here <http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation>`_.
3. Download and install CUDNN from `here <https://developer.nvidia.com/cudnn>`_.

Make sure you have the latest CUDA (8.0) and CUDNN (7.0).

2d. Install PyTorch
^^^^^^^^^^^^^^^^^^^

There are instructions on http://pytorch.org which detail how to install it.
If you have been following along so far and have Anaconda installed with CUDA enabled, you can simply do:


.. code-block:: bash

   conda install pytorch torchvision cuda80 -c soumith

The widget on PyTorch.org will let you select the right command line for your specific OS/Arch.

PLEASE NOTE.   Make sure you have PyTorch 0.3.0.  PyTorch has recently released version 0.4.0, but it has many code changes that we will not be incorporating at this time.   The Anaconda installation method for this is:

.. code-block:: bash

   conda install pytorch=0.3.1 torchvision -c pytorch

If you would like to install using pips and wheels:

.. code-block:: bash

   pip install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
   pip install torchvision



2e. Clone (or Download) Repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At this point, you may have already cloned the tutorial repository.  But if
you have not, you will need it for the next step.

.. code-block:: bash

   git clone https://github.com/joosthub/pytorch-nlp-tutorial-ny2018.git

If you do not have git or do not want to use it, you can also
`download the repository as a zip file <https://github.com/joosthub/pytorch-nlp-tutorial-ny2018/archive/master.zip>`_

2f. Install Dependencies from Repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming the you have cloned (or downloaded and unzipped) the repository,
please navigate to the directory in your terminal.  Then, you can do the following:

.. code-block:: bash

   pip install -r requirements.txt

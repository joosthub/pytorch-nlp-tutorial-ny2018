Tensor-Fu-2
===========

Exercise 1
----------

.. code-block:: python

   indices = torch.arange(10).long()
   indices = torch.from_numpy(np.random.randint(0, 10, size=(10,)))

   emb = nn.Embedding(num_embeddings=100, embedding_dim=16)
   emb(indices)

Task: Get the above code to work.
Use the second indices method and change the size to a matrix (such as (10,11)).

Exercise 2
----------

Task: Create a MultiEmbedding class which can input two sets of indices, embed them, and concat the results!

.. code-block:: python

   class MultiEmbedding(nn.Module):
       def __init__(self, num_embeddings1, num_embeddings2, embedding_dim1, embedding_dim2):
           pass

       def forward(self, indices1, indices2):
           # use something like
           # z = torch.concat([x, y], dim=1)

           pass
Solutions 
=========

Problem 1
---------

.. code-block:: python

   def f(x):
       if x.data[0] > 0:
           return torch.sin(x)
       else:
           return torch.cos(x)

   x = torch.autograd.Variable(torch.FloatTensor([1]), 
                               requires_grad=True)

   y = f(x)
   print(y)

   y.backward()

   x.grad

   y.grad_fn

Problem 2
---------

.. code-block:: python

   def cbow(phrase):
       words = phrase.split(" ")
       embeddings = []
       for word in words:
           if word in glove.word_to_index:
               embeddings.append(glove.get_embedding(word))
       embeddings = np.stack(embeddings)
       return np.mean(embeddings, axis=0)

   cbow("the dog flew over the moon").shape

   # >> (100,)

   def cbow_sim(phrase1, phrase2):
       vec1 = cbow(phrase1)
       vec2 = cbow(phrase2)
       return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

   cbow_sim("green apple", "green apple")
   # >> 1.0

   cbow_sim("green apple", "apple green")
   # >> 1.0

   cbow_sim("green apple", "red potato")
   # >> 0.749

   cbow_sim("green apple", "green alien")
   # >> 0.683

   cbow_sim("green apple", "blue alien")
   # >> 0.5799815958114477

   cbow_sim("eat an apple", "ingest an apple")
   # >> 0.9304712574359718
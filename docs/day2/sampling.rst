Exercise: Sampling from an RNN
==============================

The goal of sampling from an RNN is to initialize the sequence in some way, feed it into the recurrent computation, and retrieve the next prediction. 

To start, we create the initial vectors:

.. code-block:: python

   start_index = vectorizer.surname_vocab.start_index
   batch_size = 2
   # hidden_size = whatever hidden size the model is set to

   initial_h = Variable(torch.ones(batch_size, hidden_size))
   initial_x_index = Variable(torch.ones(batch_size).long()) * start_index

Then, we need to use these vectors to retrieve the next prediction:

.. code-block:: python

   # model is stored in variable called `net`

   x_t = net.emb(initial_x_index)
   print(x_t.shape)
   h_t = net.rnn._compute_next_hidden(x_t, initial_h)

   y_t = net.fc(h_t)


Now that we have a prediction vector, we can create a probability distribution and sample from it.  Note we include a temperature hyper parameter for controlling how strongly we sample from the distribution (at high temperatures, everything is uniform, at low temperatures below 1, small differences are magnified).  The temperature is always greater than 0. 

.. code-block:: python
	
   temperature = 1.0
   prediction_vector = F.softmax(y_t / temperature, dim=1)
   x_index_t = torch.multinomial(y_t, 1)[:, 0]


Now we can start the cycle over again:

.. code-block:: python

   x_t = net.emb(x_index_t)
   h_t = net.rnn._compute_next_hidden(x_t, h_t)

   y_t = net.fc(h_t)

Write a for loop which repeats this sequence and appends the x_t variable to a list.

Then, we can do the following:

.. code-block:: python

   final_x_indices = torch.stack(x_indices).squeeze().permute(1, 0)

   # stop here if you don't know what cpu, data, and numpy do. Ask away!
   final_x_indices = final_x_indices.cpu().data.numpy()

   # loop over the items in the batch
   results = []
   for i in range(len(final_x_indices)):
       tokens = []
       index_vector = final_x_indices[i]
       for x_index in index_vector:
           if vectorizer.surname_vocab.start_index == x_index:
               continue
           elif vectorizer.surname_vocab.end_index == x_index:
               break
           else:
               token = vectorizer.surname_vocab.lookup(x_index)
               tokens.append(token)

   sampled_surname = "".join(tokens)
   results.append(sampled_surname)
   tokens = []
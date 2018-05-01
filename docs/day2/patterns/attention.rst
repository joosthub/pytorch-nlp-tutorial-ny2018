Design Pattern: Attention
=========================

Attention is a useful pattern for when you want to take a collection of vectors---whether it be a sequence of vectors representing a sequence of words, or an unordered collections of vectors representing a collection of attributes---and summarize them into a single vector.  This has similar analogs to the CBOW examples we saw on Day 1, but instead of just averaging or using max pooling, we are learning a function which learns to compute the weights for each of the vectors before summing them together.

Importantly, the weights that the attention module is learning is a valid probability distribution.  This means that weighting the vectors by the value the attention module learns can additionally be seen as computing the Expection. Or, it could as interpolating. In any case, attention's main use is to select 'softly' amongst a set of vectors.

The attention vector has several different published forms. The one below is very simple and just learns a single vector as the attention mechanism.

Using the :code:`new_parameter` function we have been using for the RNN notebooks:

.. code-block:: python

   def new_parameter(*size):
       out = Parameter(FloatTensor(*size))
       torch.nn.init.xavier_normal(out)
       return out

We can then do:

.. code-block:: python

   class Attention(nn.Module):
       def __init__(self, attention_size):
           super(Attention, self).__init__()
           self.attention = new_parameter(attention_size, 1)

       def forward(self, x_in):
           # after this, we have (batch, dim1) with a diff weight per each cell
           attention_score = torch.matmul(x_in, self.attention).squeeze()
           attention_score = F.softmax(attention_score).view(x_in.size(0), x_in.size(1), 1)
           scored_x = x_in * attention_score

           # now, sum across dim 1 to get the expected feature vector
           condensed_x = torch.sum(scored_x, dim=1)

           return condensed_x



   attn = Attention(100)
   x = Variable(torch.randn(16,30,100))
   attn(x).size() == (16,100)



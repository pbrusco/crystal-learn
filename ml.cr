module ML
  def ML.train_test_split(size, percentage=0.8)
    idx = (0..size-1).to_a.shuffle
    {idx.take(percentage), idx.drop(percentage)}
  end

  def ML.accuracy(actual, predicted)
    right = actual.zip(predicted).map {|(x,y)| x == y ? 1 : 0}.sum
    right.to_f / actual.size
  end
end

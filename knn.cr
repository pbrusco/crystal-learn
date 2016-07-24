class KNeighborsClassifier(T1, T2)
  def initialize(@n_neighbors=5)
    @xs = [] of Array(T1)
    @ys = [] of T2
  end

  def dist(x, y)
    Math.sqrt(x.zip(y).map {|(x, y)| (x - y).abs2}.sum)
  end

  def fit(xs, ys)
    @xs = xs
    @ys = ys
  end

  def predict(xs_new)
    xs_new.map { |x_new|
      distances = @xs.map {|x| dist(x, x_new)}
      top_n = distances.zip(@ys).sort.take(@n_neighbors).map {|x| x[1]}
      top_n.mode
    }
  end
end

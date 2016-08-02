module ML
  module Classifiers
    class KNeighborsClassifier(T1, T2)
      def initialize(n_neighbors @k=5)
        @neighbors = [] of T1
        @tags = [] of T2
      end

      def dist(x, y)
        Math.sqrt(x.zip(y).map {|(x, y)| (x - y).abs2}.sum)
      end

      def fit(xs, tags)
        @neighbors = xs
        @tags = tags
      end

      def predict(instances)
        instances.map { |x|
          dist_to_x = ->(x : T1, y : T1){dist(x, y)}.partial(x)

          distances = @neighbors.map(&dist_to_x)
          top_n = distances.zip(@tags).sort.take(@k).map(&.last)
          top_n.mode
        }
      end
    end
  end
end

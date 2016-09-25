module ML
  module Classifiers
    class KNeighborsClassifier(XType, YType)
      def initialize(n_neighbors @k=5)
        @neighbors = [] of Array(XType)
        @tags = [] of YType
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
          dist_to_x = ->(x : Array(XType), y : Array(XType)){dist(x, y)}.partial(x)

          distances = @neighbors.map(&dist_to_x)
          top_n = distances.zip(@tags).sort.take(@k).map(&.last)
          top_n.mode
        }
      end
    end
  end
end

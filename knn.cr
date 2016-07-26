module ML
  module Classifiers
    class KNeighborsClassifier(F, T)
      def initialize(@n_neighbors=5)
        @neighbors = [] of F
        @tags = [] of T
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
          distances = @neighbors.map {|neighbor| dist(neighbor, x)}
          top_n = distances.zip(@tags).sort.take(@n_neighbors).map {|x| x[1]}
          top_n.mode
        }
      end
    end
  end
end

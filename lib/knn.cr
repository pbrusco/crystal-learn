require "./classifier"

module ML
  module Classifiers
    class KNeighborsClassifier(XType, YType) < Classifier(XType, YType)

      def fit(x : Array(Array(XType)), y : Array(YType))
        @neighbors = x
        @tags = y
        self
      end

      def initialize(n_neighbors @k= 5)
        @neighbors = [] of Array(XType)
        @tags = [] of YType
      end

      def dist(x, y)
        Math.sqrt(x.zip(y).map { |(x, y)| (x - y).abs2 }.sum)
      end

      def predict(instance : Array(XType))
        dist_to_x = ->(x : Array(XType), y : Array(XType)) { dist(x, y) }.partial(instance)

        distances = @neighbors.map(&dist_to_x)
        top_n = distances.zip(@tags).sort.take(@k).map(&.last)
        top_n.mode
      end
    end
  end
end

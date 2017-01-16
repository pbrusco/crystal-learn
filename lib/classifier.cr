module ML
  module Classifiers
    abstract class Classifier(XType, YType)
      abstract def fit(x : Array(Array(XType)), y : Array(YType))
      abstract def predict(instance : Array(XType)) : YType

      def predict(instances : Array(Array(XType)))
        instances.map { |instance| predict(instance) }
      end
      
    end
  end
end

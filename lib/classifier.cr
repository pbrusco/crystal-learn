module ML
  module Classifiers
    abstract class Classifier(XType, YType)
      abstract def fit(x : Array(Array(XType)), y : Array(YType))
      abstract def predict(instances : Array(Array(XType))) : Array(YType)
      abstract def predict(instance : Array(XType)) : YType
    end
  end
end

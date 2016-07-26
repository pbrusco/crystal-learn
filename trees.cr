module ML
  module Classifiers
    class CategoricalDecisionTree
      def initialize()
      end

      def fit(xs, tags)
        data = xs.transpose
        gains = (0 .. data.size-1).map {|feature_idx| {feature_idx, ML.gain(tags, given: data[feature_idx])} }
        min_gain = gains.min_by { |feature, gain| gain }
      end

      def predict(instances)
      end


    end
  end
end

# http://www.saedsayad.com/decision_tree.htm

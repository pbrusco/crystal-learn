require "./tree_structure"
module ML
  module Classifiers

    abstract class DecisionTree
      @tree : Tree

      def initialize(@tree = EmptyTree.new)
      end

      def fit(x, y)
        @tree = build_tree(x, y)
        self
      end

      def predict(instances)
        instances.map do |new_instance|
          navigate_tree(@tree, new_instance)
        end
      end

      def build_tree(xs, tags)
        data = xs.transpose
        metric_by_feature = (0 .. data.size-1).map {|feature_idx| {feature_idx, metric_function(tags, given: data[feature_idx])} }
        selected_feature, max_metric_value = metric_by_feature.max_by { |feature, metric_value| metric_value }

        if threshold_achieved(max_metric_value)
          Leaf.new(tags: tags)
        else
          feature_values = data[selected_feature].uniq
          node = Node(typeof(feature_values[0])).new(feature_index: selected_feature, values: feature_values)
          node.children = feature_values.map { |feature_value|
            selected_rows, indices = xs.select_with_indices {|row| row[selected_feature] == feature_value}
            build_tree(selected_rows, tags[indices]) as Tree
          }
          node
        end
      end

      def show_tree(column_names)
        @tree.show(column_names, level: 0)
      end

      def navigate_tree(tree, new_instance)
        case tree
        when Leaf
          select_final_value(tree.tags)
        when Node
          split = tree.feature_index
          new_instance_feature_value = new_instance[split]
          child = tree.children_with_value(new_instance_feature_value)
          navigate_tree(child, new_instance)
        else
          raise "fit before predicting"
        end
      end
    end

    class DecisionTreeClassifier < DecisionTree
      def metric_function(tags, *, given x)
        ML.gain(tags, given: x)
      end

      def threshold_achieved(gain)
         gain == 0
      end

      def select_final_value(values)
        raise "values must be all equal" if values.uniq.size != 1
        values.first
      end
    end

    class DecisionTreeRegresor < DecisionTree
      def initialize
        @full_dataset_std=0.0f32
        super
      end

      def fit(x, y)
        @full_dataset_std = y.std
        super
      end

      def metric_function(tags, *, given x)
        ML.std_reduction(tags, given: x)
      end

      def threshold_achieved(std_reduction)
         puts std_reduction, @full_dataset_std
         std_reduction <= (0.05 * @full_dataset_std)
      end

      def select_final_value(values)
        if values.is_a? Array(Float32)
          values.mean
        else
          raise "invalid type for regresion"
        end
      end
    end
  end
end

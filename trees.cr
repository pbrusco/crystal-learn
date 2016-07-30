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
          node = build_node(xs, data, selected_feature, tags)
        end
      end

      def build_node(xs : Array(Array(String)), data, selected_feature, tags)
        feature_values = data[selected_feature].uniq
        split_feature_value = better_split(feature_values, xs)

        node = Node.new(feature_index: selected_feature, split_value: split_feature_value)

        selected_rows, indices = xs.select_with_indices {|row| row[selected_feature] == split_feature_value}
        node.left_child = build_tree(selected_rows, tags[indices])

        other_rows, other_indices = xs.select_with_indices {|row| row[selected_feature] != split_feature_value}
        node.right_child = build_tree(other_rows, tags[other_indices])
        node
      end

      def build_node(xs : Array(Array(Float32)), data, selected_feature, tags)
        split_feature_value = data[selected_feature].mean
        node = Node.new(feature_index: selected_feature, split_value: split_feature_value)

        selected_rows, indices = xs.select_with_indices {|row| row[selected_feature] <= split_feature_value}
        node.left_child = build_tree(selected_rows, tags[indices])

        other_rows, other_indices = xs.select_with_indices {|row| row[selected_feature] > split_feature_value}
        node.right_child = build_tree(other_rows, tags[other_indices])
        node
      end

      def better_split(features, xs)
        features[0] # TODO: select better partition
      end

      def show_tree(column_names)
        @tree.show(column_names, level: 0)
      end

      def decide_child(split_val : Float32, new_instance_feature_value : Float32, tree)
        child = split_val <= new_instance_feature_value ? tree.left_child : tree.right_child
      end

      def decide_child(split_val : String, new_instance_feature_value : String, tree)
        child = split_val == new_instance_feature_value ? tree.left_child : tree.right_child
      end

      def decide_child(a, b, c)
        raise "Type error"
      end

      def navigate_tree(tree, new_instance)
        case tree
        when Leaf
          select_final_value(tree.tags)
        when Node
          split = tree.feature_index
          new_instance_feature_value = new_instance[split]
          split_val = tree.split_value
          if split_val.is_a?(String)
            child = decide_child(split_val, new_instance_feature_value, tree)
          elsif split_val.is_a?(Float32)
            child = decide_child(split_val, new_instance_feature_value, tree)
          end
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
        values.mode
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

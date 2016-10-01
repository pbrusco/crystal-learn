require "./tree_structure"
require "./ml"

module ML
  module Classifiers
    abstract class DecisionTree(XType, YType)
      @tree : Tree(XType, YType)

      def initialize(@tree = EmptyTree(XType, YType).new, *, @max_depth = 10)
      end

      def depth
        @tree.depth
      end

      def fit(x : Array(Array(XType)), y : Array(YType))
        @tree = build_tree(x, y, 1)
        self
      end

      def predict(instances : Array(Array(XType)))
        instances.map do |new_instance|
          predict(new_instance)
        end
      end

      def predict(instance : Array(XType))
        navigate_tree(@tree, instance)
      end

      def build_tree(xs, tags, tree_depth)
        data = xs.transpose
        metric_by_feature = (0..data.size - 1).map { |feature_idx| {feature_idx, metric_function(tags, given: data[feature_idx])} }
        selected_feature, max_metric_value = metric_by_feature.max_by { |feature, metric_value| metric_value }
        if threshold_achieved(max_metric_value) || tree_depth > @max_depth
          Leaf(XType, YType).new(tags: tags)
        else
          node = build_node(xs, data, selected_feature, tags, tree_depth)
        end
      end

      def build_node(xs : Array(Array(Number)), data, selected_feature, tags, tree_depth)
        split_feature_value = data[selected_feature].mean
        node = Node(XType, YType).new(feature_index: selected_feature, split_value: split_feature_value)

        selected_rows, indices = xs.select_with_indices { |row| row[selected_feature] <= split_feature_value }
        node.left_child = build_tree(selected_rows, tags[indices], tree_depth + 1)

        other_rows, other_indices = xs.select_with_indices { |row| row[selected_feature] > split_feature_value }
        node.right_child = build_tree(other_rows, tags[other_indices], tree_depth + 1)
        node
      end

      def build_node(xs : Array(Array(XType)), data, selected_feature, tags, tree_depth)
        feature_values = data[selected_feature].uniq
        split_feature_value = select_best_feature_value_to_split(feature_values, xs)

        node = Node(XType, YType).new(feature_index: selected_feature, split_value: split_feature_value)

        selected_rows, indices = xs.select_with_indices { |row| row[selected_feature] == split_feature_value }
        node.left_child = build_tree(selected_rows, tags[indices], tree_depth + 1)

        other_rows, other_indices = xs.select_with_indices { |row| row[selected_feature] != split_feature_value }
        node.right_child = build_tree(other_rows, tags[other_indices], tree_depth + 1)
        node
      end

      def select_best_feature_value_to_split(features, xs)
        features[0] # TODO: select better partition
      end

      def show_tree(column_names)
        @tree.show(column_names, level: 0)
      end

      def decide_child(split_val : Number, new_instance_feature_value : Number, tree)
        child = split_val > new_instance_feature_value ? tree.left_child : tree.right_child
      end

      def decide_child(split_val : XType, new_instance_feature_value : XType, tree)
        child = split_val == new_instance_feature_value ? tree.left_child : tree.right_child
      end

      def navigate_tree(tree, new_instance)
        case tree
        when Leaf
          select_final_value(tree.tags)
        when Node
          split = tree.feature_index
          new_instance_feature_value = new_instance[split]
          split_val = tree.split_value
          child = decide_child(split_val, new_instance_feature_value, tree)
          navigate_tree(child, new_instance)
        else
          raise "fit before predicting"
        end
      end
    end

    class DecisionTreeClassifier(XType, YType) < DecisionTree(XType, YType)
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

    class DecisionTreeRegressor(XType, YType) < DecisionTree(XType, YType)
      @full_dataset_std : Float32 | Float64 | Nil

      def fit(x : Array(Array(XType)), y : Array(Float))
        @full_dataset_std = y.std
        super
      end

      def metric_function(tags, *, given x)
        ML.std_reduction(tags, given: x)
      end

      def threshold_achieved(stdr)
        stdr <= (0.05 * @full_dataset_std.not_nil!)
      end

      def select_final_value(values)
        if values.is_a? Array(Float32) | Array(Float64)
          values.mean
        else
          raise "invalid type for regresion"
        end
      end
    end
  end
end

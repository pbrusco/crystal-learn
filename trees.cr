module ML
  module Classifiers

    abstract class DecisionTree
      @tree : Tree
      @column_names : Array(String)?

      def initialize(@tree = EmptyTree.new)
      end

      def fit(xs, tags, *, @column_names = nil)
        @tree = build_tree(xs, tags)
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
          node = Node.new(feature: selected_feature, values: feature_values)
          node.children = feature_values.map { |feature_value|
            selected_rows, indices = xs.select_with_indices {|row| row[selected_feature] == feature_value}
            build_tree(selected_rows, tags[indices]) as Tree
          }
          node
        end
      end

      def show_tree
        @tree.show(@column_names, level: 0)
      end

      def navigate_tree(tree, new_instance)
        case tree
        when Leaf
          select_final_value(tree.tags)
        when Node
          split = tree.feature
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
        values.first
      end
    end

    class DecisionTreeRegresor < DecisionTree
      @full_dataset_std : Float32?

      def fit(x, y)
        @full_dataset_std = y.std
        super
      end

      def metric_function(tags, *, given x)
        ML.std_reduction(tags, given: x)
      end

      def threshold_achieved(std_reduction)
         std_reduction < 0.05 * @full_dataset_std.not_nil!
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

# http://www.saedsayad.com/decision_tree.htm

abstract class Tree
end

class Node < Tree
  property :children
  property :feature

  @children : Array(Tree)

  def initialize(@feature : Int32, @values : Array(String))
    @children = Array(Tree).new(@values.size)
  end

  def children_with_value(value)
    @children[@values.index(value).not_nil!]
  end

  def show(column_names, level)
    tabs = "\t" * level
    feature_name = column_names ? column_names[@feature] : "F#{@feature}"

    puts tabs + "Node(feature=#{feature_name})"
    @children.zip(@values).each do |c, v|
      puts tabs + "value: #{v}"
      c.show(column_names, level+1)
    end
  end

end

class Leaf < Tree
  property :tags

  def initialize(@tags : Array(String) | Array(Float32))
  end

  def show(column_names, level)
    tabs = "\t" * level
    class_name = column_names ? column_names.last : "class: "
    puts tabs + "Hoja(#{class_name}: #{@tags})"
  end
end

class EmptyTree < Tree
  def show(column_names, level)
  end
end

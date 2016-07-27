module ML
  module Classifiers
    class DecisionTreeClassifier
      @tree : Tree
      @column_names : Array(Symbol) | Nil

      def initialize(@tree = EmptyTree.new)
      end

      def fit(xs, tags, *, @column_names)
        @tree = build_tree(xs, tags)
        self
      end

      def build_tree(xs, tags)
        data = xs.transpose
        gains_by_feature = (0 .. data.size-1).map {|feature_idx| {feature_idx, ML.gain(tags, given: data[feature_idx])} }
        selected_feature, max_gain = gains_by_feature.max_by { |feature, gain| gain }

        if max_gain == 0
          Leaf.new(tag: tags[0])
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

      def predict(instances)
        raise "Fit before predicting" if @tree.nil?
        instances.map do |i|
          navigate_tree(i)
        end
      end

      def show_tree
        @tree.show(@column_names, level: 0)
      end

      def navigate_tree(i)
        case @tree
        when Leaf
          @tree.tag
        when Node
          split = @tree.feature
          navigate(@tree.children(i[split]))
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

  def initialize(@feature : Int32, @values : Array(Symbol))
    @children = Array(Tree).new(@values.size)
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
  property :tag

  def initialize(@tag : Symbol)
  end

  def show(column_names, level)
    tabs = "\t" * level
    class_name = column_names ? column_names.last : "class: "
    puts tabs + "Hoja(#{class_name}: #{@tag})"
  end
end

class EmptyTree < Tree
  def show(column_names, level)
  end
end

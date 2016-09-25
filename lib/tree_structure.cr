# http://www.saedsayad.com/decision_tree.htm

abstract class Tree(XType, YType)
end

class Node(XType, YType) < Tree(XType, YType)
  property :left_child
  property :right_child
  property :feature_index
  property :split_value

  @left_child : Tree(XType, YType)
  @right_child : Tree(XType, YType)
  @split_value : XType

  def initialize(@feature_index : Int32, @split_value)
    @right_child = EmptyTree(XType, YType).new
    @left_child = EmptyTree(XType, YType).new
  end

  def depth
    Math.max(@left_child.depth, @right_child.depth) + 1
  end

  def children
    [@left_child, @right_child]
  end

  def show(column_names, level)
    tabs = "\t" * level
    feature_name = column_names ? column_names[@feature_index] : "F#{@feature_index}"

    puts tabs + "Node(feature=#{feature_name})"
    children.each do |c|
      puts tabs + "split_by: #{@split_value}"
      c.show(column_names, level+1)
    end
  end
end

class Leaf(XType, YType) < Tree(XType, YType)
  property :tags

  def initialize(@tags : Array(YType) )  # TODO: Change for generic type when crystal is ready :)
  end

  def depth
    0
  end

  def show(column_names, level)
    tabs = "\t" * level
    class_name = column_names ? column_names.last : "class: "
    puts tabs + "Hoja(#{class_name}: #{@tags.mode})"
  end
end

class EmptyTree(XType, YType) < Tree(XType, YType)
  def show(column_names, level)
  end

  def depth
    0
  end
end
